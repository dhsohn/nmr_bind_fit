import argparse
from types import SimpleNamespace

import numpy as np

from nmr_bind_fit.models import MODEL_SPECS
from nmr_bind_fit.report import DecisionEntry, _decision_paragraphs
from nmr_bind_fit.report_pipeline import (
    _build_model_warnings,
    _build_summary_row,
    build_decisions,
    build_methods_sections,
    build_report_artifacts,
)


def _result(**overrides):
    # A complete FitResult-shaped stand-in; override only what a test exercises.
    base = dict(
        model=SimpleNamespace(name="11", n_logk=1),
        datasets=[],
        params=np.array([2.0, 7.0, 7.5], dtype=float),
        param_names=["logK", "H", "HG"],
        bootstrap=None,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=11.0,
        penalty_count=0,
        species=[],
        residual_diagnostics={},
        n=10,
        p=3,
        dof=7,
        jacobian_rank=3,
        jacobian_condition=100.0,
        jacobian_logk_sensitivity=0.5,
        logk_bounds=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_decisions_uses_provisional_language():
    args = argparse.Namespace(bootstrap_ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": _result(model=SimpleNamespace(name="11", n_logk=1), bic=10.0),
            "12": _result(model=SimpleNamespace(name="12", n_logk=2), bic=10.9),
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


def test_build_decisions_propagates_unavailable_bootstrap_uncertainty():
    args = argparse.Namespace(bootstrap_ci_width=None)
    result = _result(
        bic=10.0,
        bootstrap=SimpleNamespace(
            ci_valid=False,
            ci_message="Bootstrap uncertainty unavailable: 1/1000 refits succeeded.",
            n_boot=1000,
            n_success=1,
            ci_method_used="unavailable",
        ),
    )

    decisions, entries = build_decisions(
        args,
        ordered_keys=["dataset_a"],
        results_by_key={"dataset_a": {"11": result}},
        failures_by_key={},
        display_model_name=lambda name: name,
    )

    assert any("1/1000 refits succeeded" in line for line in decisions)
    assert any("1/1000 refits succeeded" in reason for reason in entries[0].reasons)


def test_build_report_artifacts_uses_fit_failed_wording_for_exclusions(tmp_path):
    summary_rows, model_entries, warnings = build_report_artifacts(
        args=argparse.Namespace(),
        ordered_keys=["dataset_a"],
        results_by_key={"dataset_a": {}},
        failures_by_key={"dataset_a": [("11", "ModelFitError: forced model crash")]},
        out_dir=tmp_path,
        display_model_name=lambda name: name,
    )

    assert len(summary_rows) == 1
    assert summary_rows[0]["Status"] == "failed"
    assert "fit failed: ModelFitError: forced model crash" in summary_rows[0]["Notes"]
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
    assert "physical free-guest bracket [0, [G]ₜ]" in content
    assert "scale-adaptive xtol = 10⁻¹³" in content
    assert "rtol = 8 machine epsilons" in content
    assert "bracket-scale-adaptive iteration budget with a minimum of 200" in content
    assert "condition number at most 10⁶" in content
    assert "no active log₁₀(K) bound" in content
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
    assert "BCa-style adjusted bootstrap quantiles" in uq_section["content"]
    assert "local warm-start refit estimator" in uq_section["content"]
    assert "leave-one-out jackknife fits" in uq_section["content"]
    assert "95% profile-likelihood RSS window" in uq_section["content"]
    assert "at least 20 refits were requested" in uq_section["content"]
    assert "every requested pseudo-dataset yielded an uncensored acceptable refit" in uq_section["content"]
    assert "selected interval method to succeed" in uq_section["content"]
    assert "BCa-only failure" in uq_section["content"]
    assert "complete raw-distribution summaries available" in uq_section["content"]


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
        ci_valid=True,
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
    assert row["Status"] == "success"


def test_build_summary_row_labels_sequential_k_values_and_ci():
    bootstrap = SimpleNamespace(
        param_samples=np.array([[1.0, 2.0], [1.1, 2.1]], dtype=float),
        logk_samples=np.array([[1.0, 2.0], [1.1, 2.1]], dtype=float),
        ci_low=np.array([0.9, 1.9], dtype=float),
        ci_high=np.array([1.2, 2.2], dtype=float),
        ci_low_percentile=np.array([0.9, 1.9], dtype=float),
        ci_high_percentile=np.array([1.2, 2.2], dtype=float),
        ci_low_bca=np.array([np.nan, np.nan], dtype=float),
        ci_high_bca=np.array([np.nan, np.nan], dtype=float),
        ci_method="percentile",
        ci_valid=True,
        n_success=2,
        n_boot=2,
    )
    res = SimpleNamespace(
        model=SimpleNamespace(n_logk=2),
        params=np.array([1.0, 2.0], dtype=float),
        bootstrap=bootstrap,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=11.0,
        penalty_count=0,
    )

    row = _build_summary_row(res, "sample", "12")

    assert row["Binding constant (M⁻¹)"] == "K1=10; K2=100"
    assert row["95 % CI"] == f"K1=[{10**0.9:.6g}, {10**1.2:.6g}]; K2=[{10**1.9:.6g}, {10**2.2:.6g}]"


def _pinned_k_result(logk_bounds):
    return SimpleNamespace(
        model=MODEL_SPECS["11"],
        datasets=[],
        params=np.array([12.0, 7.0, 7.5], dtype=float),
        param_names=["logK", "H", "HG"],
        bootstrap=None,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=11.0,
        penalty_count=0,
        species=[],
        residual_diagnostics={},
        n=10,
        p=3,
        logk_bounds=logk_bounds,
    )


def test_bound_pinned_k_is_reported_in_warnings_and_summary_notes():
    res = _pinned_k_result((0.0, 12.0))

    warnings = _build_model_warnings(argparse.Namespace(bootstrap_ci_width=None), res, None)
    row = _build_summary_row(res, "sample", "11", warnings)

    assert any("upper log10(K) bound" in warning for warning in warnings)
    assert "upper log10(K) bound" in row["Notes"]


def test_bound_pinned_warning_uses_actual_bounds_not_cli_constants():
    # A logK of 12 is only "pinned" if 12 was the active upper bound. With no
    # bounds (a programmatic fit) or a wider bound, K=1e12 is a valid estimate
    # and must not be flagged.
    unbounded = _build_model_warnings(
        argparse.Namespace(bootstrap_ci_width=None), _pinned_k_result(None), None
    )
    wider = _build_model_warnings(
        argparse.Namespace(bootstrap_ci_width=None), _pinned_k_result((0.0, 15.0)), None
    )

    assert not any("log10(K) bound" in warning for warning in unbounded)
    assert not any("log10(K) bound" in warning for warning in wider)


def test_aicc_only_unavailable_is_explained_in_warnings():
    # An underpowered fit can keep a finite BIC while the AICc small-sample
    # correction is undefined (NaN). The report must explain the resulting
    # AICc=N/A instead of showing it silently.
    res = SimpleNamespace(
        model=MODEL_SPECS["11"],
        datasets=[],
        params=np.array([3.0, 7.0, 7.5], dtype=float),
        param_names=["logK", "H", "HG"],
        bootstrap=None,
        r2=0.9,
        r2_per_peak=[0.9],
        rss=1.0,
        rmse=0.5,
        bic=10.0,
        aicc=float("nan"),
        penalty_count=0,
        species=[],
        residual_diagnostics={},
        n=5,
        p=4,
        logk_bounds=(0.0, 12.0),
    )

    warnings = _build_model_warnings(argparse.Namespace(bootstrap_ci_width=None), res, None)
    row = _build_summary_row(res, "sample", "11", warnings)

    assert any("AICc unavailable: too few observations" in warning for warning in warnings)
    assert not any("BIC/AICc unavailable" in warning for warning in warnings)
    assert "AICc unavailable" in row["Notes"]


def test_build_decisions_excludes_nonfinite_bic_from_ranking():
    args = argparse.Namespace(bootstrap_ci_width=None)
    ordered_keys = ["dataset_a"]
    results_by_key = {
        "dataset_a": {
            "11": _result(bic=float("nan")),
        }
    }

    decisions, entries = build_decisions(
        args,
        ordered_keys,
        results_by_key,
        failures_by_key={},
        display_model_name=lambda name: name,
    )

    assert entries == []
    assert any("No model had a finite BIC" in line for line in decisions)
