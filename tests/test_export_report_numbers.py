"""scripts/export_report_numbers.py: the snapshot the slide deck and the
report artifact are built from. Runs against the real results/ and sibling
repo (same convention as tests/test_config.py's "reads real file" tests),
since the point of this script is that it can't disagree with what
app/pages/study_overview.py itself computes.
"""
import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_report_numbers.py"
_spec = importlib.util.spec_from_file_location("export_report_numbers", _SCRIPT)
export_report_numbers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_report_numbers)


def test_build_snapshot_has_the_expected_sections():
    snapshot = export_report_numbers.build_snapshot()
    assert snapshot["study"] == "v8"
    assert set(snapshot) == {
        "study", "test_window", "battery_headline", "markets", "forecast_models", "optimizer", "results", "lookahead",
    }
    assert {m["market"] for m in snapshot["markets"]} == {"day_ahead", "fcr", "afrr_pos", "afrr_neg"}
    assert snapshot["optimizer"]["battery"]["energy_mwh"] == 10.0


def test_build_snapshot_is_json_serializable_with_no_nan():
    snapshot = export_report_numbers.build_snapshot()
    text = json.dumps(snapshot)
    assert "NaN" not in text


def test_build_snapshot_writes_a_file(tmp_path):
    snapshot = export_report_numbers.build_snapshot()
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    (out_dir / f"report_numbers_{snapshot['study']}.json").write_text(json.dumps(snapshot), encoding="utf-8")
    export_report_numbers.main(["--out-dir", str(out_dir)])
    written = json.loads((out_dir / f"report_numbers_{snapshot['study']}.json").read_text(encoding="utf-8"))
    assert written["study"] == snapshot["study"]
