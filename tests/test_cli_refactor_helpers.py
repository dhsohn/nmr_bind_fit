import argparse
from types import SimpleNamespace

import pytest

from nmrbindfit.cli import _index_results, _resolve_inputs, _resolve_logk_config


def test_resolve_logk_config_defaults_and_bounds():
    args = argparse.Namespace(
        k_starts=None,
        bootstrap_logk_jitter=0.1,
        k_min=None,
        k_max=None,
    )
    starts, bounds = _resolve_logk_config(args)
    assert len(starts) == 8
    assert bounds is None

    args.k_min = 1e2
    args.k_max = 1e4
    starts, bounds = _resolve_logk_config(args)
    assert bounds == (2.0, 4.0)


def test_resolve_logk_config_rejects_partial_bounds():
    args = argparse.Namespace(
        k_starts="10,100",
        bootstrap_logk_jitter=0.1,
        k_min=1e2,
        k_max=None,
    )
    with pytest.raises(ValueError, match="Both --k-min and --k-max must be provided."):
        _resolve_logk_config(args)


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
    csv_path.write_text("Host Conc.,Guest Conc.,ppm\n1e-3,0,7.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate input files detected:"):
        _resolve_inputs(
            [
                str(tmp_path / "*.csv"),
                str(csv_path),
                str(tmp_path / "sample.*"),
            ]
        )
