import csv

import pytest

from nmr_bind_fit.report import write_summary_csv


def test_write_summary_csv_rejects_empty_rows_explicitly(tmp_path):
    output_path = tmp_path / "summary.csv"

    with pytest.raises(ValueError, match="without at least one successful fit result"):
        write_summary_csv([], output_path)

    assert not output_path.exists()


def test_write_summary_csv_neutralizes_formulas_but_preserves_numeric_negatives(tmp_path):
    # A dataset label is the input file stem, so an input named "=cmd.csv" puts
    # that text straight into the first column and a spreadsheet would evaluate
    # it. Negative and exponent-form statistics must survive unchanged.
    output_path = tmp_path / "summary.csv"
    rows = [
        {
            "Dataset": '=HYPERLINK("https://example.invalid", "sample")',
            "Model": "@malicious-name",
            "BIC": "-12.5",
            "AICc": "-1.25e-3",
            "Notes": " \t+FORMULA(1)",
        }
    ]

    write_summary_csv(rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        written = next(csv.DictReader(handle))

    assert written["Dataset"].startswith("'=HYPERLINK")
    assert written["Model"] == "'@malicious-name"
    assert written["BIC"] == "-12.5"
    assert written["AICc"] == "-1.25e-3"
    assert written["Notes"] == "' \t+FORMULA(1)"
