"""Seeded scientific stress tests with independent, scale-aware references."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy import optimize, special, stats

import curvefit_stats as cs

_Array = NDArray[np.float64]
pytestmark = pytest.mark.filterwarnings(
    "ignore:scipy.odr is deprecated in SciPy:FutureWarning"
)


def _line(x: _Array, a: float, b: float) -> _Array:
    """Evaluate a two-parameter line."""
    return a * x + b


def _line_jac(x: _Array, a: float, b: float) -> _Array:
    """Return symbolic derivatives."""
    return np.column_stack((x, np.ones_like(x)))


@pytest.mark.parametrize("scale", [1e-100, 1e-50, 1.0, 1e50, 1e100])
def test_response_unit_invariance(scale: float) -> None:
    """Keep weighted inference invariant across 200 orders of response units."""
    x = np.linspace(0.2, 4, 50)
    sigma = 0.04 * (1 + x)
    y = _line(x, 2, 1) + np.random.default_rng(547).normal(size=x.size) * sigma
    design = _line_jac(x, 0, 0)
    weighted = design / sigma[:, None]
    reference, _, _, _ = np.linalg.lstsq(weighted, y / sigma, rcond=None)
    covariance = np.linalg.inv(weighted.T @ weighted)

    def model(z: _Array, a: float, b: float) -> _Array:
        """Express the same model in changed response units."""
        return scale * _line(z, a, b)

    def jac(z: _Array, a: float, b: float) -> _Array:
        """Express symbolic derivatives in changed response units."""
        return scale * _line_jac(z, a, b)

    group = cs.GlobalFitGroup(
        model,
        {"sample": {"x": x, "y": y * scale, "sigma": sigma * scale}},
        {"a", "b"},
    )
    fitted = group.fit([1.5, 0.8], jac=jac, absolute_sigma=True)
    assert fitted.success
    assert_allclose(fitted.popt, reference, rtol=2e-10)
    assert_allclose(fitted.pcov, covariance, rtol=2e-10)
    bounded = cs.robust_bounded_pcov(
        model,
        x,
        y * scale,
        reference,
        (-np.inf, np.inf),
        sigma=sigma * scale,
        jac=jac,
        absolute_sigma=True,
    )
    assert_allclose(bounded, covariance, rtol=2e-10)
    lo, hi = cs.generate_error_bands(
        model,
        x,
        reference,
        covariance,
        jac=jac,
        reference="normal",
    )
    width = stats.norm.isf(0.025) * np.sqrt(
        np.sum((design @ covariance) * design, axis=1)
    )
    assert_allclose((hi - lo) / (2 * scale), width, rtol=2e-12)
    audit = cs.audit_fit(
        model, x, y * scale, reference, sigma=sigma * scale, absolute_sigma=True
    )
    wss = np.sum(((y - design @ reference) / sigma) ** 2)
    assert_allclose(audit["chi_square"], wss, rtol=2e-12)
    deviance = x.size * np.log(2 * np.pi) + 2 * np.sum(np.log(sigma * scale)) + wss
    assert_allclose(audit["aic"], deviance + 4, rtol=1e-12)
    assert_allclose(audit["bic"], deviance + 2 * np.log(x.size), rtol=1e-12)
    beta, cov, diagnostic = cs.eiv_curve_fit(
        model,
        x,
        y * scale,
        p0=reference,
        sx=0.05,
        sy=sigma * scale,
        absolute_sigma=True,
        full_output=True,
    )
    assert diagnostic.success and diagnostic.identifiable

    # Independent latent-coordinate least-squares solution to the ODR objective.
    def residual(theta: _Array) -> _Array:
        """Stack x and y measurement residuals in their original units."""
        return np.concatenate(
            ((theta[2:] - x) / 0.05, (_line(theta[2:], *theta[:2]) - y) / sigma)
        )

    independent = optimize.least_squares(
        residual, np.r_[reference, x], xtol=1e-12, ftol=1e-12, gtol=1e-12
    )
    assert independent.success
    assert_allclose(beta, independent.x[:2], rtol=3e-6)
    independent_jac = np.asarray(independent.jac, dtype=float)
    joint_cov = np.linalg.inv(independent_jac.T @ independent_jac)[:2, :2]
    assert_allclose(cov, joint_cov, rtol=2e-5)
    assert_allclose(
        diagnostic.weighted_sse, independent.fun @ independent.fun, rtol=2e-8
    )


@pytest.mark.parametrize("kind", ["exponential", "michaelis", "logistic"])
def test_nonlinear_multiple_datasets_and_profile(kind: str) -> None:
    """Recover shared nonlinear shape and independently check profile endpoints."""

    def model(x: _Array, amplitude: float, shape: float) -> _Array:
        """Evaluate each benchmark model without changing its parameter ordering."""
        if kind == "exponential":
            return amplitude * np.exp(-shape * x)
        if kind == "michaelis":
            return amplitude * x / (shape + x)
        return np.asarray(amplitude * special.expit(shape * x), dtype=float)

    rng = np.random.default_rng(950)
    datasets = {}
    for name, n, amplitude in [("first", 41, 2.0), ("second", 67, 3.5)]:
        x = np.linspace(0.05, 5, n)
        sigma = 0.02 * (1 + x / 5)
        datasets[name] = {
            "x": x,
            "y": model(x, amplitude, 0.8) + rng.normal(size=n) * sigma,
            "sigma": sigma,
        }
    group = cs.GlobalFitGroup(model, datasets, {"shape"})
    fit = group.fit(
        {"shape": 0.7, "first.amplitude": 1.8, "second.amplitude": 3.2},
        bounds=(0.01, 10),
        absolute_sigma=True,
    )
    assert fit.success and fit.diagnostics.identifiable
    truth = np.array([0.8, 2.0, 3.5])
    assert np.all(np.abs(fit.popt - truth) < 4 * np.sqrt(np.diag(fit.pcov)))

    # Eliminate dataset amplitudes analytically, then minimize the one shared shape.
    def projected(shape: float) -> float:
        """Solve amplitude normal equations separately for a fixed common shape."""
        total = 0.0
        for dataset in datasets.values():
            column = model(dataset["x"], 1, shape) / dataset["sigma"]
            response = dataset["y"] / dataset["sigma"]
            amplitude = column @ response / (column @ column)
            total += float(np.sum((response - amplitude * column) ** 2))
        return total

    ref = optimize.minimize_scalar(
        projected, bounds=(0.1, 2), method="bounded", options={"xatol": 1e-12}
    )
    assert_allclose(fit.popt[0], ref.x, rtol=3e-6)
    # Profile a single dataset against independently eliminated nuisance amplitude.
    dataset = datasets["second"]
    x, y, sigma = dataset["x"], dataset["y"], dataset["sigma"]
    beta, covariance = optimize.curve_fit(
        model, x, y, [3.5, 0.8], sigma=sigma, absolute_sigma=True, bounds=(0.01, 10)
    )
    errors = cs.profile_likelihood_errors(
        model,
        x,
        y,
        beta,
        sigma=sigma,
        pcov=covariance,
        absolute_sigma=True,
        bounds=(0.01, 10),
    )
    base = float(np.sum(((y - model(x, *beta)) / sigma) ** 2))
    for direction in ["up", "down"]:
        endpoint = beta[1] + errors[1][direction]
        column = model(x, 1, endpoint) / sigma
        amplitude = float(column @ (y / sigma) / (column @ column))
        increase = np.sum(((y - model(x, amplitude, endpoint)) / sigma) ** 2) - base
        assert_allclose(increase, 1, rtol=3e-5, atol=3e-6)
    summary = cs.fit_summary(model, x, y, beta, covariance)
    assert "nan" not in summary.lower()


@pytest.mark.parametrize("bound,truth", [(0.0, -1.0), (2.0, 3.0)])
def test_active_lower_upper_conditional_reference(bound: float, truth: float) -> None:
    """Compare conditional covariance to the remaining intercept normal equation."""
    x = np.linspace(-1, 1, 61)
    y = _line(x, truth, 0.7) + np.random.default_rng(53).normal(0, 0.03, x.size)
    limits = (
        ([bound, -np.inf], [np.inf, np.inf])
        if bound == 0
        else ([-np.inf, -np.inf], [bound, np.inf])
    )
    group = cs.GlobalFitGroup(_line, {"a": {"x": x, "y": y, "sigma": 0.03}}, {"a", "b"})
    with pytest.warns(UserWarning, match="[Bb]ound-active"):
        fit = group.fit([1, 0], bounds=limits, absolute_sigma=True, jac=_line_jac)
    assert fit.success
    assert np.isnan(fit.pcov[0]).all()
    assert_allclose(fit.pcov[1, 1], 0.03**2 / x.size, rtol=2e-12)
    assert_allclose(fit.popt[0], bound, atol=1e-10)
    with pytest.warns(UserWarning, match="[Bb]ound-active"):
        conditional = cs.robust_bounded_pcov(
            _line,
            x,
            y,
            fit.popt,
            limits,
            sigma=0.03,
            absolute_sigma=True,
            jac=_line_jac,
        )
    assert_allclose(conditional, fit.pcov, equal_nan=True)


@pytest.mark.parametrize("separation", [1e-3, 1e-9, 0.0])
def test_correlated_and_nearly_singular_jacobians(separation: float) -> None:
    """Distinguish severe correlation from numerical non-identifiability."""
    x = np.linspace(0, 1, 60)

    def model(z: _Array, a: float, b: float) -> _Array:
        """Make two columns increasingly indistinguishable."""
        return a * z + b * (z + separation)

    group = cs.GlobalFitGroup(
        model, {"d": {"x": x, "y": model(x, 2, 3), "sigma": 0.1}}, {"a", "b"}
    )
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = group.fit([1, 1], absolute_sigma=True)
    if separation <= 1e-9:
        assert not result.diagnostics.identifiable
        assert np.isnan(result.pcov).all()
    else:
        with pytest.warns(UserWarning, match="correlation"):
            diagnostic = cs.collinearity_diagnostics(result.pcov)
        assert diagnostic["condition_number"] > 1e6
        correlation = diagnostic["correlation_matrix"]
        assert_allclose(
            diagnostic["vif"], np.diag(np.linalg.inv(correlation)), rtol=1e-8
        )


@pytest.mark.parametrize("offset", [1e12, 1e15, 1e16])
def test_monte_carlo_offset_variance_regression(offset: float) -> None:
    """Compare represented output variance using translation before accumulation."""
    n = 20000
    result = cs.propagate_uncertainty(
        lambda a: offset + a, [0], [[1]], n_samples=n, random_state=165
    )
    represented = offset + np.random.default_rng(165).normal(size=n)
    reference = np.std(represented - offset, ddof=1)
    assert_allclose(result["std"], reference, rtol=2e-14)


def test_bootstrap_large_offset_covariance_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate aggregation from optimization using finite known bootstrap outputs."""
    samples = 1e16 + np.array([-4, -2, 0, 2, 4, 6], dtype=float)
    counter = -1

    def fitted(*args: object, **kwargs: object) -> tuple[_Array, _Array]:
        """Return a valid base fit followed by prescribed successful draws."""
        nonlocal counter
        counter += 1
        value = 1e16 if counter == 0 else samples[counter - 1]
        return np.array([value]), np.array([[1.0]])

    monkeypatch.setattr(optimize, "curve_fit", fitted)

    def model(x: _Array, a: float) -> _Array:
        """Evaluate a constant fitted response."""
        return np.full_like(x, a)

    _, covariance, draws = cs.bootstrap_curve_fit(
        model, np.arange(4.0), np.full(4, 1e16), [1e16], n_resamples=6, random_state=2
    )
    assert_allclose(covariance[0, 0], np.var(draws[:, 0] - 1e16, ddof=1), rtol=2e-14)


def test_bootstrap_ratio_and_gaussian_monte_carlo_sampling_error() -> None:
    """Use sampling-error bounds for bootstrap covariance and a safe derived ratio."""
    x = np.linspace(-2, 2, 70)
    y = _line(x, 2, 4) + np.random.default_rng(829).normal(0, 0.2, x.size)
    design = _line_jac(x, 0, 0)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    # Empirical residual bootstrap uses variance with divisor N, not N-p.
    reference = np.mean(residual**2) * np.linalg.inv(design.T @ design)
    mean, covariance, _ = cs.bootstrap_curve_fit(
        _line, x, y, beta, n_resamples=1500, random_state=51
    )
    relative_se = np.sqrt(2 / 1499)
    assert_allclose(np.diag(covariance), np.diag(reference), rtol=5 * relative_se)
    assert np.all(np.abs(mean - beta) < 5 * np.sqrt(np.diag(reference) / 1500))
    derived = cs.propagate_uncertainty(
        lambda a, b: a / b, mean, covariance, n_samples=20000, random_state=11
    )
    gradient = np.array([1 / mean[1], -mean[0] / mean[1] ** 2])
    delta_std = np.sqrt(gradient @ covariance @ gradient)
    assert_allclose(derived["std"], delta_std, rtol=5 / np.sqrt(2 * 19999) + 0.002)


@pytest.mark.parametrize("bad_sigma", [0.0, -1.0, np.inf, np.nan])
def test_pathological_uncertainties_are_rejected(bad_sigma: float) -> None:
    """Keep invalid inputs separate from positive, representable scale extremes."""
    x = np.linspace(0, 1, 10)
    with pytest.raises(ValueError):
        cs.eiv_curve_fit(_line, x, _line(x, 2, 1), [1, 1], sx=bad_sigma, sy=0.1)
    with pytest.raises(ValueError):
        cs.audit_fit(_line, x, _line(x, 2, 1), [2, 1], sigma=bad_sigma)


def test_failed_global_optimization_has_no_uncertainty() -> None:
    """A forced iteration limit cannot manufacture a converged covariance."""
    x = np.linspace(0, 4, 50)

    def decay(z: _Array, a: float, b: float) -> _Array:
        """Evaluate a nonlinear decay benchmark."""
        return a * np.exp(-b * z)

    group = cs.GlobalFitGroup(decay, {"d": {"x": x, "y": decay(x, 3, 0.7)}}, {"a", "b"})
    with pytest.warns(UserWarning, match="converge"):
        fit = group.fit([0.2, 4], max_nfev=1)
    assert not fit.success
    assert np.isnan(fit.pcov).all()


@pytest.mark.parametrize("parameter_scale", [1e-12, 1e12])
def test_parameter_units_preserve_information(parameter_scale: float) -> None:
    """Keep a large parameter-unit difference from creating false rank failure."""
    x = np.linspace(-2, 2, 60)
    y = _line(x, 2, 1) + np.random.default_rng(82).normal(0, 0.1, x.size)

    def model(z: _Array, a: float, b: float) -> _Array:
        """Scale only the slope parameter's physical units."""
        return (a / parameter_scale) * z + b

    def jac(z: _Array, a: float, b: float) -> _Array:
        """Supply scale-aware analytical derivatives."""
        return np.column_stack((z / parameter_scale, np.ones_like(z)))

    fit = cs.GlobalFitGroup(
        model, {"d": {"x": x, "y": y, "sigma": 0.1}}, {"a", "b"}
    ).fit([1.5 * parameter_scale, 0.8], jac=jac, absolute_sigma=True)
    design = _line_jac(x, 0, 0)
    reference = np.linalg.lstsq(design, y, rcond=None)[0]
    transform = np.diag([parameter_scale, 1])
    cov = transform @ (0.1**2 * np.linalg.inv(design.T @ design)) @ transform
    assert fit.success and fit.diagnostics.identifiable
    assert_allclose(fit.popt / np.array([parameter_scale, 1]), reference, rtol=2e-10)
    units = np.sqrt(np.diag(cov))
    assert_allclose(
        fit.pcov / units[:, None] / units[None, :],
        cov / units[:, None] / units[None, :],
        rtol=2e-10,
        atol=2e-14,
    )
    diagnostic = cs.collinearity_diagnostics(fit.pcov)
    assert diagnostic["rank"] == 2
    assert_allclose(diagnostic["vif"], [1, 1], rtol=1e-12)
