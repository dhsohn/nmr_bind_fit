import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import nmr_bind_fit.cli as cli_module
from nmr_bind_fit.cli import (
    _index_results,
    _reserve_output_dir,
    _resolve_inputs,
    _resolve_logk_config,
    _safe_output_name,
    build_parser,
    main,
    run_fit,
)


def test_resolve_logk_config_defaults_and_fixed_bounds():
    args = argparse.Namespace(
        k_starts=None,
        bootstrap_logk_jitter=0.1,
    )
    starts, bounds = _resolve_logk_config(args)
    assert len(starts) == 8
    assert bounds == (0.0, 12.0)

    args.k_starts = "10,100"
    starts, bounds = _resolve_logk_config(args)
    assert starts == [1.0, 2.0]
    assert bounds == (0.0, 12.0)


def test_resolve_logk_config_rejects_negative_jitter():
    args = argparse.Namespace(
        k_starts="10,100",
        bootstrap_logk_jitter=-0.1,
    )
    with pytest.raises(ValueError, match="--bootstrap-logk-jitter must be non-negative."):
        _resolve_logk_config(args)


def test_resolve_logk_config_rejects_nan_k_starts():
    args = SimpleNamespace(
        k_starts="nan",
        bootstrap_logk_jitter=0.1,
    )

    with pytest.raises(ValueError, match="All K starts must be finite."):
        _resolve_logk_config(args)


def test_resolve_logk_config_rejects_nan_jitter():
    args = SimpleNamespace(
        k_starts="10",
        bootstrap_logk_jitter=float("nan"),
    )

    with pytest.raises(ValueError, match="--bootstrap-logk-jitter must be finite."):
        _resolve_logk_config(args)


def test_resolve_logk_config_rejects_empty_k_starts_list():
    args = argparse.Namespace(
        k_starts=",",
        bootstrap_logk_jitter=0.1,
    )
    with pytest.raises(ValueError, match="--k-starts must include at least one positive value."):
        _resolve_logk_config(args)


def test_resolve_logk_config_rejects_k_starts_out_of_strict_bounds():
    args = argparse.Namespace(
        k_starts="10,1e13",
        bootstrap_logk_jitter=0.1,
    )
    with pytest.raises(ValueError, match=r"All K starts must be within \[1e\+00, 1e\+12\]\."):
        _resolve_logk_config(args)


def test_safe_output_name_is_bounded_and_collision_resistant():
    first = _safe_output_name("sample/" + "a" * 300)
    second = _safe_output_name("sample?" + "a" * 300)

    assert len(first) <= 80
    assert len(second) <= 80
    assert first != second


def test_reserve_output_dir_uses_atomic_suffixes(tmp_path, monkeypatch):
    base = tmp_path / "fixed-analysis"
    monkeypatch.setattr(cli_module, "_auto_output_dir", lambda _paths: base)

    first = _reserve_output_dir([Path("sample.csv")])
    second = _reserve_output_dir([Path("sample.csv")])

    assert first == base
    assert second == tmp_path / "fixed-analysis_02"
    assert first.is_dir()
    assert second.is_dir()


@pytest.mark.parametrize("k_starts", ["nan", "inf", "10,-inf"])
def test_resolve_logk_config_rejects_nonfinite_k_starts(k_starts):
    args = argparse.Namespace(
        k_starts=k_starts,
        bootstrap_logk_jitter=0.1,
    )
    with pytest.raises(ValueError, match="All K starts must be finite."):
        _resolve_logk_config(args)


@pytest.mark.parametrize("jitter", [float("nan"), float("inf")])
def test_resolve_logk_config_rejects_nonfinite_jitter(jitter):
    args = argparse.Namespace(
        k_starts="10,100",
        bootstrap_logk_jitter=jitter,
    )
    with pytest.raises(ValueError, match="--bootstrap-logk-jitter must be finite."):
        _resolve_logk_config(args)


def test_build_parser_rejects_negative_bootstrap(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "sample.csv", "--bootstrap", "-1"])
    err = capsys.readouterr().err
    assert "--bootstrap must be non-negative." in err


def test_build_parser_sets_default_flags():
    parser = build_parser()
    args = parser.parse_args(["--input", "sample.csv", "--bootstrap", "0"])
    assert args.bootstrap_ci_method == "percentile"
    assert args.residual_diagnostics is False


def test_build_parser_removed_concentration_unit_flag():
    parser = build_parser()
    # The flag no longer exists on the parsed namespace.
    args = parser.parse_args(["--input", "sample.csv", "--bootstrap", "0"])
    assert not hasattr(args, "concentration_unit")
    # Supplying it is now an error rather than being silently accepted.
    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "sample.csv", "--concentration-unit", "mM"])


def test_run_fit_rejects_negative_bootstrap_when_parser_is_bypassed():
    args = argparse.Namespace(
        input=["sample.csv"],
        ppm_cols=None,
        bootstrap=-1,
        bootstrap_method="residual",
        bootstrap_logk_jitter=0.1,
        k_starts=None,
        replicates=False,
        max_nfev=100,
        seed=None,
        bootstrap_ci_width=None,
    )
    with pytest.raises(ValueError, match="--bootstrap must be non-negative."):
        run_fit(args)


def test_run_fit_rejects_unknown_bootstrap_ci_method_when_parser_is_bypassed():
    args = argparse.Namespace(
        input=["sample.csv"],
        ppm_cols=None,
        bootstrap=0,
        bootstrap_ci_method="unknown",
        residual_diagnostics=False,
        bootstrap_method="residual",
        bootstrap_logk_jitter=0.1,
        k_starts=None,
        replicates=False,
        max_nfev=100,
        seed=None,
        bootstrap_ci_width=None,
    )
    with pytest.raises(ValueError, match="--bootstrap-ci-method must be one of: percentile, bca."):
        run_fit(args)


def test_run_fit_rejects_nonpositive_max_nfev_before_loading_input():
    args = argparse.Namespace(
        input=["missing.csv"],
        ppm_cols=None,
        bootstrap=0,
        bootstrap_method="residual",
        bootstrap_logk_jitter=0.1,
        k_starts=None,
        replicates=False,
        max_nfev=0,
        seed=None,
        bootstrap_ci_width=None,
    )
    with pytest.raises(ValueError, match="--max-nfev must be a positive integer."):
        run_fit(args)


def test_main_returns_nonzero_for_zero_max_nfev(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["nmr_bind_fit", "--input", "missing.csv", "--max-nfev", "0"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "--max-nfev must be a positive integer." in capsys.readouterr().err


def test_main_returns_nonzero_for_nan_k_start_before_loading_input(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["nmr_bind_fit", "--input", "missing.csv", "--k-starts", "nan"],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "All K starts must be finite." in capsys.readouterr().err


def test_run_fit_rejects_nonfinite_bootstrap_ci_width():
    args = argparse.Namespace(
        input=["missing.csv"],
        ppm_cols=None,
        bootstrap=0,
        bootstrap_method="residual",
        bootstrap_logk_jitter=0.1,
        k_starts=None,
        replicates=False,
        max_nfev=100,
        seed=None,
        bootstrap_ci_width=float("nan"),
    )
    with pytest.raises(ValueError, match="--bootstrap-ci-width must be finite."):
        run_fit(args)


def test_run_fit_rejects_nonpositive_bootstrap_ci_width():
    args = argparse.Namespace(
        input=["missing.csv"],
        ppm_cols=None,
        bootstrap=0,
        bootstrap_method="residual",
        bootstrap_logk_jitter=0.1,
        k_starts=None,
        replicates=False,
        max_nfev=100,
        seed=None,
        bootstrap_ci_width=0,
    )
    with pytest.raises(ValueError, match="--bootstrap-ci-width must be positive."):
        run_fit(args)


def test_index_results_groups_success_and_failures_by_dataset():
    ds1 = SimpleNamespace(name="a")
    ds2 = SimpleNamespace(name="b")
    labels = {id(ds1): "Dataset A", id(ds2): "Dataset B"}
    results = [
        SimpleNamespace(datasets=[ds1], success=True, message="ok", model=SimpleNamespace(name="11")),
        SimpleNamespace(datasets=[ds1], success=False, message="fail", model=SimpleNamespace(name="12")),
        SimpleNamespace(datasets=[ds2], success=True, message="ok", model=SimpleNamespace(name="11")),
    ]

    ordered_keys, result_map, failures = _index_results(results, labels)

    assert ordered_keys == ["Dataset A", "Dataset B"]
    assert set(result_map["Dataset A"].keys()) == {"11"}
    assert set(result_map["Dataset B"].keys()) == {"11"}
    assert failures["Dataset A"] == [("12", "fail")]


def test_resolve_inputs_rejects_duplicate_glob_and_explicit_path(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("[H]t,[G]t,ppm\n1e-3,0,7.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate input files detected:"):
        _resolve_inputs(
            [
                str(tmp_path / "*.csv"),
                str(csv_path),
                str(tmp_path / "sample.*"),
            ]
        )


def test_resolve_inputs_rejects_missing_pattern_even_when_another_input_exists(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("[H]t,[G]t,ppm\n1e-3,0,7.1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Input file or pattern not found"):
        _resolve_inputs([str(csv_path), str(tmp_path / "missing*.csv")])


def test_resolve_inputs_sorts_glob_matches(tmp_path):
    b_path = tmp_path / "b.csv"
    a_path = tmp_path / "a.csv"
    b_path.write_text("[H]t,[G]t,ppm\n1e-3,0,7.1\n", encoding="utf-8")
    a_path.write_text("[H]t,[G]t,ppm\n1e-3,0,7.1\n", encoding="utf-8")

    paths = _resolve_inputs([str(tmp_path / "*.csv")])

    assert [path.name for path in paths] == ["a.csv", "b.csv"]


def test_run_fit_rejects_replicates_with_one_dataset(monkeypatch):
    args = SimpleNamespace(
        bootstrap=0,
        bootstrap_ci_method="percentile",
        residual_diagnostics=False,
        input=["one.csv"],
        ppm_cols=None,
        k_starts="10",
        max_nfev=100,
        bootstrap_method="residual",
        seed=None,
        replicates=True,
        bootstrap_logk_jitter=0.1,
        bootstrap_ci_width=None,
    )
    monkeypatch.setattr(cli_module, "_resolve_inputs", lambda _patterns: [Path("one.csv")])
    monkeypatch.setattr(
        cli_module,
        "load_datasets",
        lambda _paths, ppm_cols, missing_policy: [
            SimpleNamespace(name="one", path=Path("one.csv"))
        ],
    )

    with pytest.raises(ValueError, match="--replicates requires at least two input datasets"):
        run_fit(args)


def test_resolve_inputs_rejects_directories(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="Input path is not a regular file"):
        _resolve_inputs([str(input_dir)])


def test_run_fit_fails_without_creating_reports_when_every_model_fails(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm_H1": [7.1, 7.2, 7.3],
        }
    ).to_csv(data_path, index=False)

    def fail_every_model(datasets, model_names, **_kwargs):
        return [
            SimpleNamespace(
                datasets=[datasets[0]],
                success=False,
                message="optimizer failed",
                model=SimpleNamespace(name=model_name),
            )
            for model_name in model_names
        ]

    monkeypatch.setattr(cli_module, "fit_models", fail_every_model)
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(
        [
            "--input",
            str(data_path),
            "--bootstrap",
            "0",
            "--k-starts",
            "10",
            "--max-nfev",
            "100",
        ]
    )

    with pytest.raises(ValueError, match="All model fits failed; no report was generated"):
        run_fit(args)

    assert not any(path.is_dir() for path in tmp_path.iterdir())
