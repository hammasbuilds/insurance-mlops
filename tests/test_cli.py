"""The command line, as CI and on-call would run it."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from insurance.cli import main, read_rows

SRC = str(Path(__file__).resolve().parents[1] / "src")


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def batches(tmp_path):
    training = [{"vehicle_age": float(1 + i % 12), "claims": i % 3} for i in range(120)]
    ok = [{"vehicle_age": r["vehicle_age"], "claims": float(r["claims"])} for r in training]
    broken = [{**r, "vehicle_age": r["vehicle_age"] * 12} for r in ok]
    return (
        write_jsonl(tmp_path / "train.jsonl", training),
        write_jsonl(tmp_path / "ok.jsonl", ok),
        write_jsonl(tmp_path / "broken.jsonl", broken),
    )


def test_skew_clean_exits_zero(batches, capsys):
    train, ok, _ = batches
    assert main(["skew", str(train), str(ok)]) == 0
    assert "SAFE TO SERVE" in capsys.readouterr().out


def test_skew_broken_exits_one_and_names_the_feature(batches, capsys):
    train, _, broken = batches
    assert main(["skew", str(train), str(broken)]) == 1
    out = capsys.readouterr().out
    assert "NOT SAFE TO SERVE" in out and "vehicle_age" in out and "years vs months" in out


def test_skew_json_output_parses(batches, capsys):
    train, _, broken = batches
    main(["skew", str(train), str(broken), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["safe_to_serve"] is False and report["critical"] >= 1


def test_csv_and_jsonl_read_the_same(tmp_path):
    (tmp_path / "a.csv").write_text("x,g\n1,M\n2.5,F\n,M\nNaN,F\n", encoding="utf-8")
    rows = read_rows(str(tmp_path / "a.csv"))
    assert rows == [
        {"x": 1, "g": "M"},
        {"x": 2.5, "g": "F"},
        {"x": None, "g": "M"},
        {"x": None, "g": "F"},
    ]


def test_json_array_and_nan_literals(tmp_path):
    (tmp_path / "a.json").write_text('[{"x": 1}, {"x": NaN}]', encoding="utf-8")
    rows = read_rows(str(tmp_path / "a.json"))
    assert rows[0] == {"x": 1} and rows[1]["x"] != rows[1]["x"]


@pytest.mark.parametrize(
    "name,content,message",
    [
        ("bad.jsonl", '{"x": 1}\n{oops}\n', "line 2"),
        ("list.jsonl", "[1, 2]\n", "not a JSON object"),
        ("rows.txt", "x\n1\n", "unsupported file type"),
    ],
)
def test_bad_input_exits_two_with_a_message(tmp_path, capsys, name, content, message):
    bad = tmp_path / name
    bad.write_text(content, encoding="utf-8")
    good = write_jsonl(tmp_path / "good.jsonl", [{"x": 1}])
    assert main(["skew", str(good), str(bad)]) == 2
    assert message in capsys.readouterr().err


def test_a_missing_file_exits_two(tmp_path, capsys):
    assert main(["skew", str(tmp_path / "nope.jsonl"), str(tmp_path / "nope2.jsonl")]) == 2
    assert "no such file" in capsys.readouterr().err


def _gate_files(tmp_path, metrics):
    card = {
        "name": "fraud",
        "version": "1",
        "intended_use": "triage",
        "out_of_scope": ["pricing"],
        "training_data": "claims 2023-25",
        "limitations": ["no fleet"],
        "owner": "risk",
        "reviewed_by": "mrm",
    }
    paths = {}
    for name, data in (
        ("card", card),
        ("metrics", metrics),
        ("fairness", {"worst_disparate_impact": 0.9}),
        ("skew", {"safe_to_serve": True, "critical": 0, "findings": []}),
    ):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data), encoding="utf-8")
    return [f"--{k}={v}" for k, v in paths.items()]


def test_gate_approves_and_refuses(tmp_path, capsys):
    assert main(["gate", *_gate_files(tmp_path, {"gini": 0.45, "brier": 0.1})]) == 0
    assert capsys.readouterr().out.startswith("APPROVED")
    assert main(["gate", *_gate_files(tmp_path, {"gini": 0.45}), "--json"]) == 1
    decision = json.loads(capsys.readouterr().out)
    assert decision["failures"] == ["brier not supplied"]


def test_gate_rejects_unknown_card_fields(tmp_path, capsys):
    args = _gate_files(tmp_path, {"gini": 0.45, "brier": 0.1})
    (tmp_path / "card.json").write_text('{"name": "x", "version": "1", "ownr": "y"}', "utf-8")
    assert main(["gate", *args]) == 2
    assert "ownr" in capsys.readouterr().err


def test_python_dash_m_runs(batches):
    train, _, broken = batches
    proc = subprocess.run(
        [sys.executable, "-m", "insurance", "skew", str(train), str(broken)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": SRC},
    )
    assert proc.returncode == 1, proc.stderr
    assert "vehicle_age" in proc.stdout
