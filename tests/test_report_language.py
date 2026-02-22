import argparse
from types import SimpleNamespace

from core.report import DecisionEntry, _decision_paragraphs
from core.report_pipeline import _replicate_dataset_dir_labels, build_decisions, build_report_artifacts


def test_build_decisions_uses_provisional_language():
    args = argparse.Namespace(bootstrap_ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": SimpleNamespace(model=SimpleNamespace(name="11", n_logk=1), bic=10.0, bootstrap=None),
            "12": SimpleNamespace(model=SimpleNamespace(name="12", n_logk=2), bic=10.9, bootstrap=None),
        }
    }
    failures_by_key = {}

    decisions, entries = build_decisions(
        args,
        ordered_keys,
        results_by_key,
        failures_by_key,
        display_model_name=lambda name: name,
    )

    assert any("Tentative working model among tested candidates" in line for line in decisions)
    assert any("delta BIC to next candidate" in line for line in decisions)
    assert any("model selection as provisional" in line for line in decisions)
    assert len(entries) == 1
    assert "relative support only" in entries[0].reasons[0]


def test_decision_paragraphs_use_provisional_working_model_language():
    paragraphs = _decision_paragraphs(
        [DecisionEntry(dataset="sample", recommended_model="H : G = 1 : 1", reasons=["Lowest BIC among tested models"])]
    )

    assert len(paragraphs) == 1
    assert "provisional working model" in paragraphs[0]
    assert "best supported" not in paragraphs[0]


def test_replicate_dataset_dir_labels_are_collision_free():
    ds1 = SimpleNamespace(name="sample", path="a/sample.csv")
    ds2 = SimpleNamespace(name="sample", path="b/sample.csv")
    ds3 = SimpleNamespace(name="sample", path="b/sample.csv")

    labels = _replicate_dataset_dir_labels([ds1, ds2, ds3])

    assert len(labels) == 3
    assert len(set(labels)) == 3
    assert labels[0].startswith("01_")
    assert labels[1].startswith("02_")
    assert labels[2].startswith("03_")


def test_build_decisions_uses_fit_failed_wording_for_exclusions():
    args = argparse.Namespace(bootstrap_ci_width=None)
    decisions, entries = build_decisions(
        args,
        ordered_keys=["dataset_a"],
        results_by_key={"dataset_a": {}},
        failures_by_key={"dataset_a": [("11", "ModelFitError: forced model crash")]},
        display_model_name=lambda name: name,
    )

    assert len(entries) == 0
    assert any("fit failed: ModelFitError: forced model crash" in line for line in decisions)


def test_build_report_artifacts_uses_fit_failed_wording_for_exclusions(tmp_path):
    summary_rows, model_entries, warnings = build_report_artifacts(
        args=argparse.Namespace(),
        ordered_keys=["dataset_a"],
        results_by_key={"dataset_a": {}},
        failures_by_key={"dataset_a": [("11", "ModelFitError: forced model crash")]},
        out_dir=tmp_path,
        display_model_name=lambda name: name,
    )

    assert summary_rows == []
    assert model_entries == []
    assert warnings == ["dataset_a: excluded 11 (fit failed: ModelFitError: forced model crash)"]
