import pytest

from nmr_bind_fit.report import write_summary_csv


def test_write_summary_csv_rejects_empty_rows_explicitly(tmp_path):
    output_path = tmp_path / "summary.csv"

    with pytest.raises(ValueError, match="without at least one successful fit result"):
        write_summary_csv([], output_path)

    assert not output_path.exists()
