import argparse
from types import SimpleNamespace

import numpy as np

from nmr_bind_fit.report import DecisionEntry, _decision_paragraphs
from nmr_bind_fit.report_pipeline import (
    _build_summary_row,
    _replicate_dataset_dir_labels,
    build_decisions,
    build_methods_sections,
    build_report_artifacts,
)


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


def test_build_methods_sections_uses_brent_and_molar_k_units():
    args = SimpleNamespace(
        replicates=False,
        bootstrap=0,
        bootstrap_method="residual",
        bootstrap_logk_jitter=0.1,
    )
    ds = SimpleNamespace(name="sample", path="sample.csv")

    sections = build_methods_sections(args, [ds])
    param_section = next(section for section in sections if section["title"] == "Parameter Estimation")
    content = param_section["content"]

    assert "Brent's method" in content
    assert "scipy.optimize.brentq" in content
    assert "xtol=1e-50, rtol=1e-15" in content
    assert "M⁻¹" in content
    assert "Newton" not in content


def test_build_methods_sections_mentions_bca_when_selected():
    args = SimpleNamespace(
        replicates=False,
        bootstrap=100,
        bootstrap_method="residual",
        bootstrap_ci_method="bca",
        bootstrap_logk_jitter=0.1,
    )
    ds = SimpleNamespace(name="sample", path="sample.csv")

    sections = build_methods_sections(args, [ds])
    uq_section = next(section for section in sections if section["title"] == "Uncertainty Quantification")
    assert "BCa-adjusted bootstrap quantiles" in uq_section["content"]


def test_build_summary_row_uses_selected_ci_and_reports_logk_se():
    bootstrap = SimpleNamespace(
        param_samples=np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float),
        logk_samples=np.array([[1.0], [2.0], [3.0]], dtype=float),
        ci_low=np.array([1.2], dtype=float),
        ci_high=np.array([2.8], dtype=float),
        ci_low_percentile=np.array([1.05], dtype=float),
        ci_high_percentile=np.array([2.95], dtype=float),
        ci_low_bca=np.array([1.2], dtype=float),
        ci_high_bca=np.array([2.8], dtype=float),
        ci_method="bca",
        n_success=3,
        n_boot=3,
    )
    res = SimpleNamespace(
        model=SimpleNamespace(n_logk=1),
        params=np.array([2.0], dtype=float),
        bootstrap=bootstrap,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=11.0,
        penalty_count=0,
    )

    row = _build_summary_row(res, "sample", "11")

    assert row["95 % CI"] == f"[{10**1.2:.6g}, {10**2.8:.6g}]"
