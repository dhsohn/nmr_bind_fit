from pathlib import Path
from types import SimpleNamespace

from nmr_bind_fit.cli import _build_dataset_labels, _dataset_key


def test_dataset_key_disambiguates_same_stem_different_paths():
    ds1 = SimpleNamespace(name="sample", path=Path("a/sample.csv"))
    ds2 = SimpleNamespace(name="sample", path=Path("b/sample.csv"))
    labels = _build_dataset_labels([ds1, ds2])

    key1 = _dataset_key(SimpleNamespace(datasets=[ds1]), labels)
    key2 = _dataset_key(SimpleNamespace(datasets=[ds2]), labels)

    assert key1 != key2
    assert key1.startswith("sample")
    assert key2.startswith("sample")


def test_dataset_key_disambiguates_repeated_same_path():
    ds1 = SimpleNamespace(name="sample", path=Path("a/sample.csv"))
    ds2 = SimpleNamespace(name="sample", path=Path("a/sample.csv"))
    labels = _build_dataset_labels([ds1, ds2])

    key1 = _dataset_key(SimpleNamespace(datasets=[ds1]), labels)
    key2 = _dataset_key(SimpleNamespace(datasets=[ds2]), labels)

    assert key1 != key2
    assert "#1" in key1 or "#2" in key1
    assert "#1" in key2 or "#2" in key2


def test_dataset_key_keeps_replicate_grouping_label():
    ds1 = SimpleNamespace(name="sample", path=Path("a/sample.csv"))
    ds2 = SimpleNamespace(name="sample", path=Path("b/sample.csv"))
    labels = _build_dataset_labels([ds1, ds2])

    key = _dataset_key(SimpleNamespace(datasets=[ds1, ds2]), labels)
    assert key == "Simultaneous Fitting"
