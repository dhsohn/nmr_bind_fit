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


def test_run_fit_writes_report_artifacts(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 2.5e-4, 5e-4, 1e-3],
            "ppm_H1": [7.10, 7.15, 7.20, 7.32],
        }
    ).to_csv(data_path, index=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            str(data_path),
            "--k-starts",
            "10",
            "--max-nfev",
            "200",
        ]
    )

    monkeypatch.chdir(tmp_path)
    run_fit(args)

    output_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(output_dirs) == 1

    out_dir = output_dirs[0]
    assert not (out_dir / "summary.csv").exists()
    assert not (out_dir / "decision.txt").exists()
    assert (out_dir / "report.html").is_file()


def test_run_fit_replicates_and_continue_mode(tmp_path, monkeypatch):
    data_path_1 = tmp_path / "sample1.csv"
    data_path_2 = tmp_path / "sample2.csv"

    pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 2.5e-4, 5e-4, 1e-3],
            "ppm_H1": [7.10, 7.15, 7.20, 7.32],
            "ppm_H2": [8.00, 8.03, 8.09, 8.18],
        }
    ).to_csv(data_path_1, index=False)

    pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 2.5e-4, 5e-4, 1e-3],
            "ppm_H1": [7.08, 7.14, 7.22, 7.35],
            "ppm_H2": [7.99, 8.02, 8.08, 8.17],
        }
    ).to_csv(data_path_2, index=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            str(data_path_1),
            str(data_path_2),
            "--replicates",
            "--k-starts",
            "10",
            "--max-nfev",
            "200",
        ]
    )

    monkeypatch.chdir(tmp_path)
    run_fit(args)

    output_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(output_dirs) == 1

    out_dir = output_dirs[0]
    assert not (out_dir / "summary.csv").exists()
    assert not (out_dir / "decision.txt").exists()
    report_path = out_dir / "report.html"
    assert report_path.is_file()

    report_text = report_path.read_text(encoding="utf-8")
    # Replicate fits are reported under one shared key rather than per input.
    assert "Simultaneous Fitting" in report_text
    assert "fail-fast behavior" in report_text
    assert "dropped before fitting" in report_text


def test_run_fit_keeps_same_named_independent_inputs_and_artifacts_separate(tmp_path, monkeypatch):
    input_dirs = [tmp_path / "private-a", tmp_path / "private-b"]
    for idx, input_dir in enumerate(input_dirs):
        input_dir.mkdir()
        pd.DataFrame(
            {
                "[H]t": [1e-3] * 5,
                "[G]t": [0.0, 2.5e-4, 5e-4, 1e-3, 2e-3],
                "ppm_H1": [
                    7.10 + 0.01 * idx,
                    7.14 + 0.01 * idx,
                    7.19 + 0.01 * idx,
                    7.27 + 0.01 * idx,
                    7.34 + 0.01 * idx,
                ],
            }
        ).to_csv(input_dir / "sample.csv", index=False)

    args = build_parser().parse_args(
        [
            "--input",
            *(str(input_dir / "sample.csv") for input_dir in input_dirs),
            "--k-starts",
            "10",
            "--max-nfev",
            "200",
        ]
    )

    monkeypatch.chdir(tmp_path)
    run_fit(args)

    output_dirs = [
        path
        for path in tmp_path.iterdir()
        if path.is_dir() and path not in input_dirs
    ]
    assert len(output_dirs) == 1
    out_dir = output_dirs[0]

    report_text = (out_dir / "report.html").read_text(encoding="utf-8")
    # Same-named inputs get distinct numbered labels, and no input path leaks.
    assert "1. sample" in report_text
    assert "2. sample" in report_text
    assert str(tmp_path) not in report_text

    dataset_dirs = sorted((out_dir / "model_nb").glob("dataset_*"))
    assert len(dataset_dirs) == 2
    assert len({path.name.casefold() for path in dataset_dirs}) == 2
    assert all(list(path.glob("*.png")) for path in dataset_dirs)

    for png_path in (path for directory in dataset_dirs for path in directory.glob("*.png")):
        assert png_path.relative_to(out_dir).as_posix() in report_text


def test_run_fit_rejects_out_of_bounds_k_starts_before_fitting(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "[H]t": [1e-3, 1e-3, 1e-3],
            "[G]t": [0.0, 5e-4, 1e-3],
            "ppm_H1": [7.10, 7.20, 7.32],
        }
    ).to_csv(data_path, index=False)

    parser = build_parser()
    args = parser.parse_args(
        [
            "--input",
            str(data_path),
            "--k-starts",
            "1e13",
        ]
    )

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=r"All K starts must be within \[1e\+00, 1e\+12\]\."):
        run_fit(args)

    output_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert output_dirs == []


def test_resolve_logk_config_defaults_and_fixed_bounds():
    args = argparse.Namespace(
        k_starts=None,
    )
    starts, bounds = _resolve_logk_config(args)
    assert starts
    assert all(bounds[0] <= start <= bounds[1] for start in starts)
    assert bounds == (0.0, 12.0)

    args.k_starts = "10,100"
    starts, bounds = _resolve_logk_config(args)
    assert starts == [1.0, 2.0]
    assert bounds == (0.0, 12.0)


def test_resolve_logk_config_rejects_empty_k_starts_list():
    args = argparse.Namespace(
        k_starts=",",
    )
    with pytest.raises(ValueError, match="--k-starts must include at least one positive value."):
        _resolve_logk_config(args)


def test_resolve_logk_config_rejects_k_starts_out_of_strict_bounds():
    args = argparse.Namespace(
        k_starts="10,1e13",
    )
    with pytest.raises(ValueError, match=r"All K starts must be within \[1e\+00, 1e\+12\]\."):
        _resolve_logk_config(args)


def test_safe_output_name_is_normalized_and_bounded():
    safe = _safe_output_name("../sample? " + "a" * 300)

    assert len(safe) <= 80
    assert safe.startswith("sample_")
    assert not {"/", "?", " "}.intersection(safe)
    assert _safe_output_name("...") == "output"


def test_reserve_output_dir_uses_atomic_suffixes(tmp_path, monkeypatch):
    base = tmp_path / "fixed-analysis"
    monkeypatch.setattr(cli_module, "_auto_output_dir", lambda _paths: base)

    first = _reserve_output_dir([Path("sample.csv")])
    second = _reserve_output_dir([Path("sample.csv")])

    assert first == base
    assert second != first
    assert first.is_dir()
    assert second.is_dir()


@pytest.mark.parametrize("k_starts", ["nan", "inf", "10,-inf"])
def test_resolve_logk_config_rejects_nonfinite_k_starts(k_starts):
    args = argparse.Namespace(
        k_starts=k_starts,
    )
    with pytest.raises(ValueError, match="All K starts must be finite."):
        _resolve_logk_config(args)


def test_build_parser_sets_default_flags():
    parser = build_parser()
    args = parser.parse_args(["--input", "sample.csv"])
    assert args.ci_width is None
    assert args.residual_diagnostics is False


@pytest.mark.parametrize("bad_option", [["--max-nfev", "0"], ["--k-starts", "nan"]])
def test_main_exits_nonzero_with_a_message_for_invalid_options(monkeypatch, capsys, bad_option):
    # The input file does not exist, so failing cleanly here also shows the
    # options are rejected before any data is read.
    monkeypatch.setattr(
        sys,
        "argv",
        ["nmr_bind_fit", "--input", "missing.csv", *bad_option],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code != 0
    assert capsys.readouterr().err.strip()


@pytest.mark.parametrize("value", ["0", "-1", "nan"])
def test_build_parser_rejects_invalid_ci_width(value):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "sample.csv", "--ci-width", value])


def test_index_results_groups_success_and_failures_by_dataset():
    ds1 = SimpleNamespace(name="a")
    ds2 = SimpleNamespace(name="b")
    labels = {id(ds1): "Dataset A", id(ds2): "Dataset B"}
    results = [
        SimpleNamespace(datasets=[ds1], success=True, message="ok", model=SimpleNamespace(name="11")),
        SimpleNamespace(datasets=[ds1], success=False, message="fail", model=SimpleNamespace(name="12")),
        SimpleNamespace(datasets=[ds2], success=True, message="ok", model=SimpleNamespace(name="11")),
    ]

    result_map, failures = _index_results(results, labels)

    assert list(result_map) == ["Dataset A", "Dataset B"]
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


def test_resolve_inputs_accepts_literal_path_with_glob_characters(tmp_path):
    csv_path = tmp_path / "sample[1].csv"
    csv_path.write_text("[H]t,[G]t,ppm\n1e-3,0,7.1\n", encoding="utf-8")

    paths = _resolve_inputs([str(csv_path)])

    assert paths == [csv_path]


def test_resolve_inputs_prefers_literal_path_over_glob_metacharacter_match(tmp_path):
    # A filename containing glob metacharacters must resolve to itself, never to
    # a different file the pattern happens to match, or the tool would silently
    # analyze data the user did not request.
    literal = tmp_path / "sample[1].csv"
    decoy = tmp_path / "sample1.csv"
    literal.write_text("[H]t,[G]t,ppm\n1e-3,0,7.1\n", encoding="utf-8")
    decoy.write_text("[H]t,[G]t,ppm\n1e-3,0,7.1\n", encoding="utf-8")

    paths = _resolve_inputs([str(literal)])

    assert paths == [literal]


def test_run_fit_rejects_replicates_with_one_dataset(monkeypatch):
    args = SimpleNamespace(
        residual_diagnostics=False,
        input=["one.csv"],
        ppm_cols=None,
        k_starts="10",
        max_nfev=100,
        replicates=True,
        ci_width=None,
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
            "--k-starts",
            "10",
            "--max-nfev",
            "100",
        ]
    )

    with pytest.raises(ValueError, match="All model fits failed; no report was generated"):
        run_fit(args)

    assert not any(path.is_dir() for path in tmp_path.iterdir())
