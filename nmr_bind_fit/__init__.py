"""NMR binding fit package."""

from .fit import fit_models

__version__ = "0.2.0"

# Public API surface for importers.
__all__ = ["__version__", "fit_models"]
