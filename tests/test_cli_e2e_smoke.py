import pandas as pd

from nmrbindfit.cli import build_parser, run_fit


def test_run_fit_writes_report_artifacts(tmp_path, monkeypatch):
    data_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "Host Conc.": [1e-3, 1e-3, 1e-3, 1e-3],
            "Guest Conc.": [0.0, 2.5e-4, 5e-4, 1e-3],
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
