from types import SimpleNamespace

from nmr_bind_fit.cli import _build_dataset_labels, _index_results


def test_dataset_labels_keep_unique_names():
    ds1 = SimpleNamespace(name="first")
    ds2 = SimpleNamespace(name="second")
    labels = _build_dataset_labels([ds1, ds2])

    assert labels == {id(ds1): "first", id(ds2): "second"}


def test_dataset_labels_number_duplicate_names_without_exposing_paths(tmp_path):
    ds1 = SimpleNamespace(name="sample", path=tmp_path / "private-a" / "sample.csv")
    ds2 = SimpleNamespace(name="sample", path=tmp_path / "private-b" / "sample.csv")
    labels = _build_dataset_labels([ds1, ds2])

    assert labels == {id(ds1): "1. sample", id(ds2): "2. sample"}
    assert all(str(tmp_path) not in label for label in labels.values())


def test_index_results_uses_simultaneous_fitting_label():
    ds1 = SimpleNamespace(name="first")
    ds2 = SimpleNamespace(name="second")
    labels = _build_dataset_labels([ds1, ds2])
    result = SimpleNamespace(
        datasets=[ds1, ds2],
        success=True,
        model=SimpleNamespace(name="11"),
        message="",
    )

    results_by_key, failures_by_key = _index_results([result], labels)

    assert results_by_key == {"Simultaneous Fitting": {"11": result}}
    assert failures_by_key == {}
