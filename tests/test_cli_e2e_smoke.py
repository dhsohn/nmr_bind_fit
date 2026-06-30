import pandas as pd
import pytest

from nmr_bind_fit.cli import build_parser, run_fit


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
            "--bootstrap",
            "0",
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
    assert (out_dir / "summary.csv").is_file()
    assert (out_dir / "decision.txt").is_file()
    assert (out_dir / "report.html").is_file()


def test_run_fit_replicates_bootstrap_and_continue_mode(tmp_path, monkeypatch):
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
            "--bootstrap",
            "2",
            "--seed",
            "0",
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
    assert (out_dir / "summary.csv").is_file()
    assert (out_dir / "decision.txt").is_file()
    report_path = out_dir / "report.html"
    assert report_path.is_file()

    report_text = report_path.read_text(encoding="utf-8")
    assert "fail-fast behavior" in report_text
    assert "dropped before fitting" in report_text


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
            "--bootstrap",
            "0",
            "--k-starts",
            "1e13",
        ]
    )

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=r"All K starts must be within \[1e\+00, 1e\+12\]\."):
        run_fit(args)

    output_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert output_dirs == []
