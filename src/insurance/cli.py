"""Command line: skew check and release gate on files, for CI and on-call.

    python -m insurance skew training.jsonl serving.jsonl [--json]
    python -m insurance gate --card card.json --metrics metrics.json \\
        --fairness fairness.json --skew skew.json [--json]

Exit codes: 0 safe / approved, 1 not safe / refused, 2 bad input.
Rows may be JSON Lines (one object per line), a JSON array of objects, or CSV. CSV
cells are read as numbers where they parse as numbers, and blank / NA / NaN / null
cells as missing, so a CSV and a JSONL export of the same data compare equal.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .governance import GovernanceError, ModelCard, ReleaseGate
from .skew import detect_skew

_MISSING_CELLS = {"", "na", "n/a", "nan", "null", "none"}


class InputError(Exception):
    pass


def _cell(text: str):
    if text.strip().lower() in _MISSING_CELLS:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def read_rows(path: str) -> list[dict]:
    """Read feature rows from .jsonl / .ndjson, .json (array) or .csv."""
    p = Path(path)
    if not p.is_file():
        raise InputError(f"{path}: no such file")
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8-sig")
    if suffix == ".csv":
        return [
            {k: _cell(v or "") for k, v in row.items()} for row in csv.DictReader(text.splitlines())
        ]
    if suffix == ".json":
        data = _loads(text, path)
        if not isinstance(data, list):
            raise InputError(f"{path}: expected a JSON array of row objects")
        rows = data
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                rows.append(_loads(line, f"{path} line {n}"))
    else:
        raise InputError(f"{path}: unsupported file type {suffix!r} (use .jsonl, .json or .csv)")
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InputError(f"{path}: row {i + 1} is not a JSON object")
    return rows


def _loads(text: str, where: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{where}: invalid JSON ({exc.msg} at line {exc.lineno})") from None


def _read_json_object(path: str | None, what: str) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        raise InputError(f"{what} {path}: no such file")
    data = _loads(p.read_text(encoding="utf-8-sig"), path)
    if not isinstance(data, dict):
        raise InputError(f"{what} {path}: expected a JSON object")
    return data


def _emit(payload: dict, as_json: bool, text_lines: list[str]) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, allow_nan=True))
    else:
        print("\n".join(text_lines))


def cmd_skew(args: argparse.Namespace) -> int:
    report = detect_skew(
        read_rows(args.training), read_rows(args.serving), tolerance=args.tolerance
    )
    lines = [
        f"training rows: {report['rows']['training']}   serving rows: {report['rows']['serving']}"
    ]
    for f in report["findings"]:
        lines.append(
            f"  {f['severity'].upper():8} {f['feature']:<20} {f['kind']:<16} {f['detail']}"
        )
    verdict = "SAFE TO SERVE" if report["safe_to_serve"] else "NOT SAFE TO SERVE"
    lines.append(f"{verdict}: {report['critical']} critical, {report['warnings']} warning(s)")
    _emit(report, args.json, lines)
    return 0 if report["safe_to_serve"] else 1


def cmd_gate(args: argparse.Namespace) -> int:
    card_data = _read_json_object(args.card, "card")
    known = set(ModelCard.__dataclass_fields__)
    unknown = sorted(set(card_data) - known)
    if unknown:
        raise InputError(f"card {args.card}: unknown field(s) {', '.join(unknown)}")
    for required in ("name", "version"):
        if required not in card_data:
            raise InputError(f"card {args.card}: missing {required!r}")
    gate = ReleaseGate(
        require_calibration=not args.no_calibration,
        require_fairness=not args.no_fairness,
        require_no_skew=not args.no_skew,
    )
    result = gate.evaluate(
        card=ModelCard(**card_data),
        metrics=_read_json_object(args.metrics, "metrics"),
        fairness=_read_json_object(args.fairness, "fairness"),
        skew=_read_json_object(args.skew, "skew"),
    )
    lines = ["APPROVED" if result["approved"] else "REFUSED"]
    lines += [f"  - {f}" for f in result["failures"]]
    lines.append(f"gate: {result['gate']}")
    _emit(result, args.json, lines)
    return 0 if result["approved"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m insurance",
        description="Training/serving skew check and release gate. "
        "Exit 0 = safe/approved, 1 = not, 2 = bad input.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    skew = sub.add_parser("skew", help="compare a training batch with a serving batch")
    skew.add_argument("training", help="training rows (.jsonl, .json or .csv)")
    skew.add_argument("serving", help="serving rows (.jsonl, .json or .csv)")
    skew.add_argument("--tolerance", type=float, default=0.25, help="relative mean shift warning")
    skew.add_argument("--json", action="store_true", help="print the report as JSON")
    skew.set_defaults(func=cmd_skew)

    gate = sub.add_parser("gate", help="run the release gate on JSON evidence files")
    gate.add_argument("--card", required=True, help="model card JSON (ModelCard fields)")
    gate.add_argument("--metrics", help='metrics JSON, e.g. {"gini": 0.45, "brier": 0.12}')
    gate.add_argument("--fairness", help='fairness JSON, e.g. {"worst_disparate_impact": 0.9}')
    gate.add_argument("--skew", help="skew report JSON, as written by `skew --json`")
    gate.add_argument("--no-calibration", action="store_true", help="do not require brier")
    gate.add_argument("--no-fairness", action="store_true", help="do not require fairness")
    gate.add_argument("--no-skew", action="store_true", help="do not require a skew report")
    gate.add_argument("--json", action="store_true", help="print the decision as JSON")
    gate.set_defaults(func=cmd_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    # A category name the console code page cannot encode must not crash a report.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (InputError, GovernanceError, OSError, UnicodeDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
