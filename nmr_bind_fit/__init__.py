"""NMR binding fit package."""

from .fit import fit_models

__version__ = "0.1.0"

# Public API surface for importers.
__all__ = ["fit_models", "__version__"]
