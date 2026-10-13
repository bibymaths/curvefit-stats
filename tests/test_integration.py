"""Cross-module workflows with independent statistical expectations."""

from __future__ import annotations

import inspect
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from scipy import optimize, stats

import curvefit_stats as cs
from curvefit_stats import estimators, inference

pytestmark = pytest.mark.filterwarnings(
    "ignore:scipy.odr is deprecated in SciPy:FutureWarning"
)
_Array = NDArray[np.float64]


def _line(x: _Array, slope: float, intercept: float) -> _Array:
    """Evaluate a line."""
    return slope * x + intercept


def _line_jac(x: _Array, slope: float, intercept: float) -> _Array:
    """Return independent analytical derivatives."""
    return np.column_stack((x, np.ones_like(x)))


def test_public_exports_are_identical_and_signatures_are_stable() -> None:
    """Re-export all ten objects without wrappers or additional functional modules."""
    expected = {
        "generate_error_bands": (
            "func",
            "x",
            "popt",
            "pcov",
            "y",
            "alpha",
            "interval_type",
        ),
        "audit_fit": ("func", "x", "y", "popt", "sigma", "absolute_sigma"),
        "fit_summary": ("func", "x", "y", "popt", "pcov", "param_names"),
        "profile_likelihood_errors": (
            "func",
            "x",
            "y",
            "popt",
            "sigma",
            "sigma_level",
            "bounds",
        ),
        "bootstrap_curve_fit": ("func", "x", "y", "p0", "n_resamples", "random_state"),
        "collinearity_diagnostics": ("pcov",),
        "robust_bounded_pcov": ("func", "x", "y", "popt", "bounds"),
        "propagate_uncertainty": ("derived_func", "popt", "pcov"),
        "GlobalFitGroup": ("model", "datasets", "shared_parameters"),
        "eiv_curve_fit": ("func", "xdata", "ydata", "p0"),
    }
    assert set(cs.__all__) == set(expected)
    for name, prefix in expected.items():
        module = estimators if name in estimators.__all__ else inference
        exported = getattr(cs, name)
        original = getattr(module, name)
        assert exported is original
        assert inspect.signature(exported) == inspect.signature(original)
        assert tuple(inspect.signature(exported).parameters)[: len(prefix)] == prefix
    assert {module.name for module in pkgutil.iter_modules(cs.__path__)} == {
        "inference",
        "estimators",
    }


@pytest.mark.parametrize(
    "entry", ["curvefit_stats", "curvefit_stats.inference", "curvefit_stats.estimators"]
)
def test_fresh_imports_do_not_trigger_odr_or_circular_imports(entry: str) -> None:
    """Top-level exports must not eagerly import a deprecated optional backend."""
    root = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        f"import {entry}\n"
        'assert "scipy.odr" not in sys.modules\n'
    )
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("known_scale", [False, True])
def test_scipy_curve_fit_to_inference(known_scale: bool) -> None:
    """Carry SciPy parameter covariance into statistically matched interval analysis."""
    generator = np.random.default_rng(517)
    x = np.linspace(-1, 3, 45)
    y = _line(x, 2.0, 0.7) + generator.normal(0, 0.12, x.size)
    sigma = np.full(x.size, 0.12) if known_scale else None
    beta, covariance = optimize.curve_fit(
        _line, x, y, sigma=sigma, absolute_sigma=known_scale
    )
    audit = cs.audit_fit(_line, x, y, beta, sigma=sigma, absolute_sigma=known_scale)
    assert audit["dof"] == x.size - 2
    grid = np.linspace(-2, 4, 79)
    design = _line_jac(grid, *beta)
    reference: Literal["normal", "student-t"] = "normal" if known_scale else "student-t"
    critical = stats.norm.isf(0.025) if known_scale else stats.t.isf(0.025, x.size - 2)
    lower, upper = cs.generate_error_bands(
        _line, grid, beta, covariance, y, x_fit=x, reference=reference
    )
    assert_allclose(
        (upper - lower) / 2,
        critical * np.sqrt(np.diag(design @ covariance @ design.T)),
        rtol=1e-8,
    )
    if not known_scale:
        summary = cs.fit_summary(_line, x, y, beta, covariance)
        assert "residual dof = 43" in summary
    dependence = cs.collinearity_diagnostics(covariance)
    assert dependence["rank"] == 2
    assert np.isfinite(dependence["vif"]).all()


def test_global_fit_to_joint_audit_and_local_bands() -> None:
    """Use global counts and scale while projecting covariance to one dataset."""
    x_a, x_b = np.linspace(-1, 2, 18), np.linspace(0.1, 4, 27)
    generator = np.random.default_rng(65)
    y_a = _line(x_a, 2.0, 0.5) + generator.normal(0, 0.1, x_a.size)
    y_b = _line(x_b, 2.0, -1.0) + generator.normal(0, 0.1, x_b.size)
    group = cs.GlobalFitGroup(
        _line, {"b": {"x": x_b, "y": y_b}, "a": {"x": x_a, "y": y_a}}, {"slope"}
    )
    result = group.fit([1.0, 0.0, 0.0], jac=_line_jac)
    assert result.success
    design = np.zeros((45, 3))
    design[:18, 0], design[18:, 0] = x_a, x_b
    design[:18, 1], design[18:, 2] = 1, 1

    def joint_model(coords: _Array, *theta: float) -> _Array:
        """Evaluate all responses using the actual global parameter vector."""
        return np.asarray(np.asarray(theta) @ coords, dtype=float)

    observations = np.r_[y_a, y_b]
    audit = cs.audit_fit(joint_model, design.T, observations, result.popt)
    assert audit["dof"] == result.diagnostics.dof == 42
    assert_allclose(audit["rss"], result.diagnostics.rss)
    assert_allclose(audit["residual_variance"], result.diagnostics.weighted_sse / 42)
    summary = cs.fit_summary(
        joint_model,
        design.T,
        observations,
        result.popt,
        result.pcov,
        list(result.parameter_names),
    )
    assert "residual dof = 42" in summary and "b.intercept" in summary
    indices = group.parameter_indices["a"]
    local_covariance = result.pcov[np.ix_(indices, indices)]
    grid = np.linspace(-2, 4, 31)
    lower, upper = cs.generate_error_bands(
        _line, grid, result.popt[indices], local_covariance, dof=42
    )
    parameter_variance = np.diag(
        _line_jac(grid, 0, 0) @ local_covariance @ _line_jac(grid, 0, 0).T
    )
    assert_allclose(
        (upper - lower) / 2,
        stats.t.isf(0.025, 42) * np.sqrt(parameter_variance),
        rtol=1e-8,
    )
    predicted_lower, predicted_upper = cs.generate_error_bands(
        _line,
        grid,
        result.popt[indices],
        local_covariance,
        dof=42,
        interval_type="prediction",
        prediction_variance=audit["residual_variance"],
    )
    assert_allclose(
        (predicted_upper - predicted_lower) / 2,
        stats.t.isf(0.025, 42)
        * np.sqrt(parameter_variance + audit["residual_variance"]),
        rtol=1e-8,
    )
    with pytest.raises(ValueError, match="dof must equal"):
        cs.generate_error_bands(
            _line, grid, result.popt[indices], local_covariance, y_a, x_fit=x_a, dof=42
        )


def test_eiv_parameter_uncertainty_and_objective_separation() -> None:
    """Use ODR covariance while preserving its errors-in-variables objective."""
    generator = np.random.default_rng(829)
    latent = np.linspace(-2, 3, 180)
    x = latent + generator.normal(0, 0.5, latent.size)
    y = _line(latent, 2.0, 0.7) + generator.normal(0, 0.1, latent.size)
    beta, covariance, odr = cs.eiv_curve_fit(
        _line, x, y, [2.0, 0.7], sx=0.5, sy=0.1, absolute_sigma=True, full_output=True
    )
    assert odr.success and odr.covariance_kind == "absolute"
    grid = np.linspace(-2, 3, 23)
    lower, upper = cs.generate_error_bands(
        _line, grid, beta, covariance, reference="normal"
    )
    design = _line_jac(grid, *beta)
    assert_allclose(
        (upper - lower) / 2,
        stats.norm.isf(0.025) * np.sqrt(np.diag(design @ covariance @ design.T)),
        rtol=1e-8,
    )
    propagation = cs.propagate_uncertainty(
        lambda slope, intercept: slope + intercept,
        beta,
        covariance,
        n_samples=20000,
        random_state=13,
    )
    expected_variance = np.array([1.0, 1.0]) @ covariance @ np.array([1.0, 1.0])
    assert_allclose(propagation["std"], np.sqrt(expected_variance), rtol=0.02)
    # A vertical residual audit is not the ODR likelihood/objective. The arrays
    # carry no estimator provenance, so compare meanings rather than relabeling.
    vertical = cs.audit_fit(
        _line, x, y, beta, sigma=np.full(x.size, 0.1), absolute_sigma=True
    )
    assert vertical["chi_square"] > 10 * odr.weighted_sse
    with pytest.raises(ValueError, match="not a local optimum"):
        cs.profile_likelihood_errors(
            _line, x, y, beta, sigma=np.full(x.size, 0.1), absolute_sigma=True
        )
    with pytest.raises(ValueError, match="explicit prediction_variance"):
        cs.generate_error_bands(
            _line,
            grid,
            beta,
            covariance,
            reference="normal",
            interval_type="prediction",
        )


def test_bootstrap_covariance_to_derived_uncertainty() -> None:
    """Compare Gaussian propagation to analytical and empirical bootstrap results."""
    x = np.linspace(-1, 4, 40)
    y = _line(x, 2.0, 1.0) + np.random.default_rng(118).normal(0, 0.1, x.size)
    mean, covariance, samples = cs.bootstrap_curve_fit(
        _line, x, y, [2.0, 1.0], n_resamples=150, random_state=7
    )
    assert samples.shape == (150, 2)
    derived_samples = 2 * samples[:, 0] + samples[:, 1]
    coefficients = np.array([2.0, 1.0])
    analytical_std = np.sqrt(coefficients @ covariance @ coefficients)
    assert_allclose(derived_samples.std(ddof=1), analytical_std, rtol=1e-12)
    result = cs.propagate_uncertainty(
        lambda slope, intercept: 2 * slope + intercept,
        mean,
        covariance,
        n_samples=25000,
        random_state=9,
    )
    assert_allclose(result["std"], analytical_std, rtol=0.02)
    assert abs(result["mean"] - derived_samples.mean()) < 0.02 * analytical_std
    # Percentiles of bootstrap samples and of a Gaussian covariance approximation
    # are different constructions; no exact equality of quantiles is asserted.
    percentile = np.quantile(derived_samples, [0.025, 0.975])
    assert percentile[0] < derived_samples.mean() < percentile[1]
    assert result == cs.propagate_uncertainty(
        lambda slope, intercept: 2 * slope + intercept,
        mean,
        covariance,
        n_samples=25000,
        random_state=9,
    )


@pytest.mark.parametrize("known_scale", [True, False])
def test_profile_and_covariance_intervals_share_only_matching_reference_laws(
    known_scale: bool,
) -> None:
    """Compare profile endpoints to their own likelihood reference laws."""
    x = np.arange(1.0, 11.0)
    y = 2 * x + np.array([0.1, -0.2, 0.15, -0.1, 0.2, -0.05, 0.1, -0.15, 0.2, -0.1])

    def model(x: _Array, slope: float) -> _Array:
        """Evaluate a one-parameter linear model."""
        return slope * x

    beta, covariance = optimize.curve_fit(
        model, x, y, p0=[2.0], sigma=0.2, absolute_sigma=known_scale
    )
    profile = cs.profile_likelihood_errors(
        model, x, y, beta, sigma=0.2, pcov=covariance, absolute_sigma=known_scale
    )
    rss = np.sum((y - model(x, *beta)) ** 2)
    reference_width = (
        0.2 / np.sqrt(x @ x)
        if known_scale
        else np.sqrt(rss * np.expm1(1 / x.size) / (x @ x))
    )
    assert_allclose(
        [profile[0]["down"], profile[0]["up"]],
        [-reference_width, reference_width],
        rtol=2e-6,
    )
    coverage_alpha = 2 * stats.norm.sf(1.0)
    if known_scale:
        assert_allclose(profile[0]["up"], np.sqrt(covariance[0, 0]), rtol=2e-6)
    else:
        t_width = stats.t.isf(coverage_alpha / 2, x.size - 1) * np.sqrt(
            covariance[0, 0]
        )
        assert abs(t_width - profile[0]["up"]) > 0.02 * profile[0]["up"]


@pytest.mark.parametrize("estimator", ["scipy", "global"])
def test_bound_active_covariance_is_conditional_not_marginal(estimator: str) -> None:
    """Reject full NaN covariance and analyze only explicitly fixed free directions."""
    x = np.linspace(0, 3, 25)
    y = _line(x, 2.0, -1.0) + np.random.default_rng(529).normal(0, 0.05, x.size)
    bounds = ([0.0, 0.0], [5.0, np.inf])
    if estimator == "scipy":
        beta, _ = optimize.curve_fit(_line, x, y, p0=[1.0, 0.1], bounds=bounds)
        with pytest.warns(UserWarning, match="conditional"):
            covariance = cs.robust_bounded_pcov(_line, x, y, beta, bounds)
    else:
        group = cs.GlobalFitGroup(
            _line, {"run": {"x": x, "y": y}}, {"slope", "intercept"}
        )
        with pytest.warns(UserWarning, match="conditional"):
            global_fit = group.fit([1.0, 0.1], bounds=bounds, jac=_line_jac)
        beta, covariance = global_fit.popt, global_fit.pcov
        assert global_fit.diagnostics.covariance_kind == "conditional"
        assert global_fit.diagnostics.covariance_dof == x.size - 1
    assert np.isnan(covariance[1]).all() and np.isnan(covariance[:, 1]).all()
    with pytest.warns(UserWarning, match="Non-finite"):
        full_diagnostics = cs.collinearity_diagnostics(covariance)
    assert (
        full_diagnostics["rank"] is None
        and full_diagnostics["potential_identifiability_issue"]
    )
    with pytest.raises(ValueError, match="finite"):
        cs.generate_error_bands(_line, x, beta, covariance, dof=x.size - 1)
    with pytest.raises(ValueError, match="finite"):
        cs.propagate_uncertainty(
            lambda slope, intercept: slope + intercept, beta, covariance
        )
    conditional = covariance[:1, :1]
    assert cs.collinearity_diagnostics(conditional)["rank"] == 1

    def fixed_model(x: _Array, slope: float) -> _Array:
        """Condition on the active intercept; do not claim its marginal uncertainty."""
        return np.asarray(slope * x + beta[1], dtype=float)

    lower, upper = cs.generate_error_bands(
        fixed_model, x, beta[:1], conditional, dof=x.size - 1
    )
    expected = stats.t.isf(0.025, x.size - 1) * np.abs(x) * np.sqrt(conditional[0, 0])
    assert_allclose((upper - lower) / 2, expected, rtol=1e-8, atol=1e-14)
    propagated = cs.propagate_uncertainty(
        lambda slope: slope * 2 + beta[1],
        beta[:1],
        conditional,
        n_samples=10000,
        random_state=5,
    )
    assert_allclose(propagated["std"], 2 * np.sqrt(conditional[0, 0]), rtol=0.03)


def test_rank_deficient_estimator_blocks_full_parameter_propagation() -> None:
    """Carry identifiability failure through the public inference boundary."""

    def model(x: _Array, a: float, b: float) -> _Array:
        """Evaluate a locally unidentified sum parameterization."""
        return (a + b) * x

    group = cs.GlobalFitGroup(
        model, {"a": {"x": [1.0, 2.0, 3.0], "y": [3.0, 6.0, 9.0]}}, {"a", "b"}
    )
    with pytest.warns(UserWarning, match="rank deficient"):
        result = group.fit(
            [1.0, 1.0], absolute_sigma=True, jac=lambda x, a, b: np.column_stack((x, x))
        )
    assert result.success and not result.diagnostics.identifiable
    with pytest.raises(ValueError, match="finite"):
        cs.propagate_uncertainty(lambda a, b: a + b, result.popt, result.pcov)
    with pytest.warns(UserWarning, match="Non-finite"):
        summary = cs.fit_summary(
            model, [1.0, 2.0, 3.0], [3.0, 6.0, 9.0], result.popt, result.pcov
        )
    assert "NaN" in summary
    assert_array_equal(result.pcov, np.full((2, 2), np.nan))


def test_tied_local_parameters_preserve_repeated_covariance_indices() -> None:
    """Project a single global degree of freedom to tied local model arguments."""

    def model(x: _Array, a: float, b: float) -> _Array:
        """Evaluate two coefficients tied to one shared global variable."""
        return (a + b) * x

    group = cs.GlobalFitGroup(
        model,
        {
            "a": {"x": [1.0, 2.0, 3.0], "y": [4.1, 7.9, 12.0]},
            "b": {"x": [2.0, 3.0, 4.0, 5.0], "y": [8.0, 12.1, 15.9, 20.0]},
        },
        {"a": "level", "b": "level"},
    )
    result = group.fit(
        [1.0], absolute_sigma=True, jac=lambda x, a, b: np.column_stack((x, x))
    )
    indices = group.parameter_indices["a"]
    assert_array_equal(indices, [0, 0])
    local_covariance = result.pcov[np.ix_(indices, indices)]
    assert local_covariance.shape == (2, 2)
    grid = np.array([1.0, 2.0, 3.0])
    lower, upper = cs.generate_error_bands(
        model, grid, result.popt[indices], local_covariance, reference="normal"
    )
    expected = stats.norm.isf(0.025) * 2 * grid * np.sqrt(result.pcov[0, 0])
    assert_allclose((upper - lower) / 2, expected, rtol=1e-8)
    assert result.diagnostics.dof == 7 - 1
