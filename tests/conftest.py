from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def examples_dir() -> Path:
    """Directory holding the deterministic synthetic example datasets."""
    return Path(__file__).parents[1] / "examples"
