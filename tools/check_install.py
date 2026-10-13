"""Smoke-test an installed distribution without importing checkout source.

Run with ``python -I /path/to/tools/check_install.py`` outside the checkout.
Only the two numerical runtime dependencies and the standard library are used.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import pkgutil
import re
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit

import curvefit_stats as cs

_Array = NDArray[np.float64]


def _decay(x: _Array, amplitude: float, rate: float) -> _Array:
    """Evaluate first-order decay for the installation smoke checks."""
    return amplitude * np.exp(-rate * x)


def main() -> None:
    """Validate installed metadata, package contents, and all ten features.

    Returns
    -------
    None
        Print installation location and success; failures raise exceptions.

    Raises
    ------
    AssertionError
        The import uses checkout source or an artifact/API check fails.

    Notes
    -----
    Expected ODR compatibility and conditional-covariance warnings are checked
    explicitly. Other warnings remain errors. This is an installation smoke
    test, not a substitute for the independent scientific reference suite.
    """
    checkout = Path(__file__).resolve().parents[1]
    location = Path(cs.__file__).resolve()
    assert not location.is_relative_to(checkout / "src"), location
    metadata = importlib.metadata.metadata("curvefit-stats")
    from pathlib import Path
    import tomllib

    project_root = Path(__file__).resolve().parents[1]

    with (project_root / "pyproject.toml").open("rb") as file:
        expected_version = tomllib.load(file)["project"]["version"]

    assert metadata["Version"] == expected_version
    assert metadata["License-Expression"] == "MIT"
    base_requirements = {
        re.split(r"[<>=!~\s\[]", requirement, maxsplit=1)[0].lower()
        for requirement in metadata.get_all("Requires-Dist", [])
        if ";" not in requirement
    }
    assert base_requirements == {"numpy", "scipy"}, base_requirements
    assert importlib.resources.files(cs).joinpath("py.typed").is_file()
    assert {module.name for module in pkgutil.iter_modules(cs.__path__)} == {
        "inference",
        "estimators",
    }
    assert len(cs.__all__) == 10
    warnings.simplefilter("error")
    x = np.linspace(0, 4, 40)
    y = _decay(x, 3, 0.7) + np.random.default_rng(932).normal(0, 0.04, x.size)
    beta, covariance = curve_fit(_decay, x, y, [2.5, 0.5])
    lower, upper = cs.generate_error_bands(_decay, x, beta, covariance, y)
    assert lower.shape == upper.shape == y.shape
    assert np.all(lower < upper)
    assert cs.audit_fit(_decay, x, y, beta)["dof"] == 38
    assert "Pr(>|t|)" in cs.fit_summary(_decay, x, y, beta, covariance)
    profile = cs.profile_likelihood_errors(_decay, x, y, beta, pcov=covariance)
    assert all(error["down"] < 0 < error["up"] for error in profile)
    _, empirical, samples = cs.bootstrap_curve_fit(
        _decay, x, y, beta, n_resamples=20, random_state=1
    )
    assert empirical.shape == (2, 2) and samples.shape == (20, 2)
    assert cs.collinearity_diagnostics(covariance)["rank"] == 2
    with warnings.catch_warnings(record=True) as emitted:
        warnings.filterwarnings(
            "always", message="Bound-active parameter rows", category=UserWarning
        )
        conditional = cs.robust_bounded_pcov(
            _decay, x, y, beta, ([0, beta[1]], [np.inf, beta[1]])
        )
    assert len(emitted) == 1
    assert np.isfinite(conditional[0, 0]) and np.isnan(conditional[1]).all()
    propagated = cs.propagate_uncertainty(
        lambda a, k: a * np.exp(-k), beta, covariance, n_samples=200, random_state=2
    )
    assert propagated["n_valid"] == 200
    group = cs.GlobalFitGroup(
        _decay, {"a": {"x": x, "y": y}, "b": {"x": x[::2], "y": y[::2] * 1.5}}, {"rate"}
    )
    global_fit = group.fit([0.6, 2.5, 4])
    assert global_fit.success and np.isfinite(global_fit.pcov).all()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.filterwarnings(
            "always", message="scipy.odr is deprecated in SciPy", category=FutureWarning
        )
        eiv, eiv_cov, odr = cs.eiv_curve_fit(
            _decay, x, y, beta, sx=0.02, sy=0.04, absolute_sigma=True, full_output=True
        )
    assert all(isinstance(item.message, FutureWarning) for item in emitted)
    assert odr.success and odr.identifiable
    assert eiv.shape == (2,) and np.isfinite(eiv_cov).all()
    print(f"Installed curvefit-stats {metadata['Version']} at {location}")
    print("Runtime-only metadata, py.typed, architecture, and ten smoke checks passed.")


if __name__ == "__main__":
    main()
