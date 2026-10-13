"""Independent statistical references for the completed inference module.

Reference calculations use closed-form regression, variable projection,
incomplete beta identities, and conditional Gaussian integration. No package
private helpers are used to construct expectations.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from scipy import integrate, optimize, special

from curvefit_stats import inference

_Array = NDArray[np.float64]
_Model = Callable[[_Array, float, float], _Array]


def _exponential(x: _Array, amplitude: float, rate: float) -> _Array:
    """Evaluate exponential decay."""
    return amplitude * np.exp(-rate * x)


def _michaelis(x: _Array, amplitude: float, scale: float) -> _Array:
    """Evaluate Michaelis-Menten saturation."""
    return amplitude * x / (scale + x)


def _logistic(x: _Array, amplitude: float, rate: float) -> _Array:
    """Evaluate logistic growth with a fixed midpoint at zero."""
    return np.asarray(amplitude * special.expit(rate * x), dtype=float)


def _jacobian(kind: str, x: _Array, beta: _Array) -> _Array:
    """Evaluate independent symbolic derivatives for each scientific model."""
    a, b = beta
    if kind == "exponential":
        z = np.exp(-b * x)
        return np.column_stack((z, -a * x * z))
    if kind == "michaelis":
        return np.column_stack((x / (b + x), -a * x / (b + x) ** 2))
    z = special.expit(b * x)
    return np.column_stack((z, a * x * z * (1 - z)))


def _variable_projection(
    model: _Model,
    x: _Array,
    y: _Array,
    sigma: _Array,
    shape_bounds: tuple[float, float],
) -> _Array:
    """Eliminate amplitude analytically and minimize over the shape parameter."""

    def amplitude_at(shape: float) -> float:
        """Solve the weighted one-column normal equation."""
        column = model(x, 1.0, shape) / sigma
        return float(column @ (y / sigma) / (column @ column))

    def objective(shape: float) -> float:
        """Calculate the weighted residual sum of squares."""
        residual = (y - model(x, amplitude_at(shape), shape)) / sigma
        return float(residual @ residual)

    result = optimize.minimize_scalar(
        objective,
        bounds=shape_bounds,
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    assert result.success
    return np.array([amplitude_at(float(result.x)), result.x])


@pytest.fixture(params=["exponential", "michaelis", "logistic"])
def scientific_case(
    request: pytest.FixtureRequest,
) -> tuple[str, _Model, _Array, _Array, _Array, _Array, tuple[float, float]]:
    """Provide independently fitted nonlinear data with known relative weights."""
    kind: str = request.param
    models: dict[str, _Model] = {
        "exponential": _exponential,
        "michaelis": _michaelis,
        "logistic": _logistic,
    }
    model = models[kind]
    x = np.linspace(0.15, 4.5, 45)
    truth = np.array([3.0, 1.2 if kind == "michaelis" else 0.8])
    sigma = 0.04 * (1 + 0.2 * x)
    y = model(x, *truth) + np.random.default_rng(725).normal(size=x.size) * sigma
    bounds = (0.15, 4.0) if kind == "michaelis" else (0.1, 2.0)
    beta = _variable_projection(model, x, y, sigma, bounds)
    return kind, model, x, y, sigma, beta, bounds


def test_nonlinear_delta_and_wald_references(
    scientific_case: tuple[
        str, _Model, _Array, _Array, _Array, _Array, tuple[float, float]
    ],
) -> None:
    """Compare Delta and Wald quantities to symbolic derivatives and beta identities."""
    kind, model, x, y, sigma, beta, _ = scientific_case
    derivative = _jacobian(kind, x, beta)
    information = (derivative / sigma[:, None]).T @ (derivative / sigma[:, None])
    covariance = np.linalg.solve(information, np.eye(2))
    grid = np.linspace(0.1, 5.0, 61)
    grid_jacobian = _jacobian(kind, grid, beta)
    variance = np.diag(grid_jacobian @ covariance @ grid_jacobian.T)
    dof = x.size - beta.size
    alpha = 0.05
    incomplete_beta_quantile = special.betaincinv(dof / 2, 0.5, alpha)
    critical = np.sqrt(dof * (1 / incomplete_beta_quantile - 1))
    lo, hi = inference.generate_error_bands(model, grid, beta, covariance, y, x_fit=x)
    assert_allclose((hi - lo) / 2, critical * np.sqrt(variance), rtol=2e-7)
    summary = inference.fit_summary(model, x, y, beta, covariance, precision=13)
    standard = np.sqrt(np.diag(covariance))
    wald = beta / standard
    # P(|T_nu| >= |t|) = I_{nu/(nu+t^2)}(nu/2, 1/2).
    probabilities = np.asarray(
        special.betainc(dof / 2, 0.5, dof / (dof + wald**2)), dtype=float
    )
    for index, row in enumerate(summary.splitlines()[4:]):
        numbers = np.array([float(field.strip()) for field in row.split("|")[1:]])
        expected = [
            beta[index],
            standard[index],
            wald[index],
            probabilities[index],
            beta[index] - critical * standard[index],
            beta[index] + critical * standard[index],
        ]
        assert_allclose(numbers[:4], expected[:4], rtol=2e-11)
        # SciPy 1.12's Student-t inverse differs from the independent beta
        # identity at about 1e-9 relative precision. Retain a tighter tolerance
        # for estimates, standard errors, and independently calculated tails.
        assert_allclose(numbers[4:], expected[4:], rtol=2e-9)


@pytest.mark.parametrize("absolute", [True, False])
def test_nonlinear_profile_against_variable_projection(
    scientific_case: tuple[
        str, _Model, _Array, _Array, _Array, _Array, tuple[float, float]
    ],
    absolute: bool,
) -> None:
    """Locate endpoints with independent one-dimensional nuisance elimination."""
    _, model, x, y, sigma, beta, shape_bounds = scientific_case
    base = np.sum(((y - model(x, *beta)) / sigma) ** 2)
    sigma_level = 1.5
    intervals = inference.profile_likelihood_errors(
        model,
        x,
        y,
        beta,
        sigma=sigma,
        absolute_sigma=absolute,
        bounds=([0.05, shape_bounds[0]], [10.0, shape_bounds[1]]),
        sigma_level=sigma_level,
    )

    def sse_at(index: int, value: float) -> float:
        """Profile amplitude in closed form, or shape with a scalar optimizer."""
        if index == 1:
            column = model(x, 1.0, value) / sigma
            amplitude = float(column @ (y / sigma) / (column @ column))
            return float(np.sum(((y - model(x, amplitude, value)) / sigma) ** 2))
        result = optimize.minimize_scalar(
            lambda shape: np.sum(((y - model(x, value, shape)) / sigma) ** 2),
            bounds=shape_bounds,
            method="bounded",
            options={"xatol": 1e-12},
        )
        assert result.success
        return float(result.fun)

    def threshold(index: int, value: float) -> float:
        """Compute a Gaussian likelihood ratio using an independent profile."""
        profiled = sse_at(index, value)
        likelihood_ratio = (
            profiled - base if absolute else x.size * np.log(profiled / base)
        )
        return float(likelihood_ratio - sigma_level**2)

    for index, (lower, upper) in enumerate(((0.05, 10.0), shape_bounds)):
        for direction, bound in [("down", lower), ("up", upper)]:
            bracket = sorted((float(beta[index]), bound))
            expected = optimize.brentq(
                lambda value, j=index: threshold(j, value),
                bracket[0],
                bracket[1],
                xtol=1e-11,
            )
            endpoint = beta[index] + intervals[index][direction]
            assert_allclose(endpoint, expected, rtol=3e-6, atol=2e-7)
            assert abs(threshold(index, endpoint)) < 3e-5
            frozen = beta.copy()
            frozen[index] = endpoint
            frozen_sse = np.sum(((y - model(x, *frozen)) / sigma) ** 2)
            assert frozen_sse > sse_at(index, endpoint) + 0.005


def test_pointwise_confidence_and_prediction_coverage() -> None:
    """Calibrate pointwise confidence and future-observation prediction coverage."""
    generator = np.random.default_rng(821)
    x = np.linspace(-1, 1, 16)
    design = np.column_stack((x, np.ones_like(x)))
    inverse_information = np.linalg.solve(design.T @ design, np.eye(2))
    projection = np.linalg.solve(design.T @ design, design.T)
    truth = np.array([1.2, 2.0])
    evaluation = np.array([0.0])
    n_trials = 2000
    confidence, prediction, future_in_confidence = 0, 0, 0

    def line(x: _Array, slope: float, intercept: float) -> _Array:
        """Evaluate a line."""
        return slope * x + intercept

    def jac(x: _Array, slope: float, intercept: float) -> _Array:
        """Return the analytical design matrix."""
        return np.column_stack((x, np.ones_like(x)))

    for _ in range(n_trials):
        y = design @ truth + generator.normal(0, 0.4, x.size)
        beta = projection @ y
        residual = y - design @ beta
        covariance = inverse_information * (residual @ residual) / (x.size - 2)
        lo, hi = inference.generate_error_bands(
            line, evaluation, beta, covariance, y, x_fit=x, jac=jac
        )
        plo, phi = inference.generate_error_bands(
            line,
            evaluation,
            beta,
            covariance,
            y,
            x_fit=x,
            jac=jac,
            interval_type="prediction",
        )
        future = truth[1] + generator.normal(0, 0.4)
        confidence += int(lo[0] <= truth[1] <= hi[0])
        prediction += int(plo[0] <= future <= phi[0])
        future_in_confidence += int(lo[0] <= future <= hi[0])
    # Fixed seeds plus broad binomial tolerances keep this deterministic while
    # testing scientific coverage, rather than asserting every realization.
    assert abs(confidence / n_trials - 0.95) < 0.025
    assert abs(prediction / n_trials - 0.95) < 0.025
    assert future_in_confidence / n_trials < 0.7


@pytest.mark.parametrize("absolute", [True, False])
def test_gaussian_likelihood_constants_and_weight_scaling(absolute: bool) -> None:
    """Verify joint Gaussian normalization and residual-scale parameter counting."""
    x = np.arange(6.0)
    residual = np.array([0.1, -0.3, 0.2, 0.15, -0.1, 0.05])
    covariance = 0.02 * np.eye(6) + 0.01 * np.ones((6, 6))
    y = _exponential(x, 2.0, 0.3) + residual
    quadratic = residual @ np.linalg.solve(covariance, residual)
    sign, determinant = np.linalg.slogdet(covariance)
    assert sign == 1
    k = 2 if absolute else 3
    deviance = (
        quadratic + determinant + 6 * np.log(2 * np.pi)
        if absolute
        else 6 * (np.log(2 * np.pi) + 1 + np.log(quadratic / 6)) + determinant
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        audit = inference.audit_fit(
            _exponential, x, y, [2.0, 0.3], sigma=covariance, absolute_sigma=absolute
        )
        scaled = inference.audit_fit(
            _exponential,
            x,
            y,
            [2.0, 0.3],
            sigma=49 * covariance,
            absolute_sigma=absolute,
        )
    assert_allclose(
        [audit["aic"], audit["bic"]], [deviance + 2 * k, deviance + k * np.log(6)]
    )
    assert audit["dof"] == 4
    if absolute:
        assert_allclose(
            scaled["aic"] - audit["aic"], 6 * np.log(49) + (1 / 49 - 1) * quadratic
        )
    else:
        assert_allclose([scaled["aic"], scaled["bic"]], [audit["aic"], audit["bic"]])
    unweighted = inference.audit_fit(_exponential, x, y, [2.0, 0.3])
    expected = 6 * (np.log(2 * np.pi) + 1 + np.log(residual @ residual / 6))
    assert_allclose(
        [unweighted["aic"], unweighted["bic"]], [expected + 6, expected + 3 * np.log(6)]
    )


def test_nonlinear_bootstrap_independent_refits() -> None:
    """Compare every nonlinear bootstrap fit and covariance to variable projection."""
    x = np.linspace(0.2, 5, 30)
    y = _michaelis(x, 3.0, 1.2) + np.random.default_rng(241).normal(0, 0.03, x.size)
    beta = _variable_projection(_michaelis, x, y, np.ones(x.size), (0.1, 4.0))
    predictions = _michaelis(x, *beta)
    residual = y - predictions
    residual -= residual.mean()
    generator = np.random.default_rng(751)
    samples = np.array(
        [
            _variable_projection(
                _michaelis,
                x,
                predictions + generator.choice(residual, size=x.size),
                np.ones(x.size),
                (0.1, 4.0),
            )
            for _ in range(50)
        ]
    )
    mean, covariance, actual = inference.bootstrap_curve_fit(
        _michaelis,
        x,
        y,
        beta,
        n_resamples=50,
        random_state=751,
        bounds=([0.1, 0.1], [10.0, 4.0]),
    )
    assert_allclose(actual, samples, rtol=3e-6, atol=2e-7)
    reference_mean = samples.sum(axis=0) / 50
    reference_covariance = (
        (samples - reference_mean).T @ (samples - reference_mean) / 49
    )
    assert_allclose(mean, reference_mean, rtol=1e-6)
    assert_allclose(covariance, reference_covariance, rtol=1e-4)
    repeated = inference.bootstrap_curve_fit(
        _michaelis,
        x,
        y,
        beta,
        n_resamples=50,
        random_state=751,
        bounds=([0.1, 0.1], [10.0, 4.0]),
    )
    assert_array_equal(actual, repeated[2])


def test_strong_nonlinear_parameter_correlation() -> None:
    """Diagnose local dependence from an independently derived Fisher matrix."""
    x = np.linspace(9.9, 10.1, 40)
    beta = np.array([3.0, 0.1])
    jacobian = _jacobian("exponential", x, beta)
    information = jacobian.T @ jacobian / 1e-8
    covariance = np.linalg.solve(information, np.eye(2))
    standard = np.sqrt(np.diag(covariance))
    correlation = covariance / standard[:, None] / standard[None, :]
    rho = correlation[0, 1]
    assert abs(rho) > 0.999
    with pytest.warns(UserWarning, match="Strong local"):
        result = inference.collinearity_diagnostics(covariance)
    assert_allclose(result["correlation_matrix"], correlation, rtol=1e-12)
    assert_allclose(result["vif"], [1 / (1 - rho**2)] * 2, rtol=1e-8)
    assert_allclose(
        result["condition_number"], (1 + abs(rho)) / (1 - abs(rho)), rtol=1e-8
    )
    assert result["rank"] == 2
    actual = inference.robust_bounded_pcov(
        _exponential,
        x,
        _exponential(x, *beta),
        beta,
        (-np.inf, np.inf),
        sigma=np.full(x.size, 1e-4),
        absolute_sigma=True,
        jac=lambda x, a, b: _jacobian("exponential", x, np.array([a, b])),
    )
    assert_allclose(actual, covariance, rtol=1e-8)


@pytest.mark.parametrize("absolute", [True, False])
def test_nonlinear_bound_active_conditional_reference(
    scientific_case: tuple[
        str, _Model, _Array, _Array, _Array, _Array, tuple[float, float]
    ],
    absolute: bool,
) -> None:
    """Use only the free shape column and N-1 in conditional covariance."""
    kind, model, x, y, sigma, beta, _ = scientific_case
    free_column = _jacobian(kind, x, beta)[:, 1] / sigma
    expected = 1 / (free_column @ free_column)
    if not absolute:
        expected *= np.sum(((y - model(x, *beta)) / sigma) ** 2) / (x.size - 1)
    with pytest.warns(UserWarning, match="conditional"):
        covariance = inference.robust_bounded_pcov(
            model,
            x,
            y,
            beta,
            ([beta[0], 0.0], [beta[0], 10.0]),
            sigma=sigma,
            absolute_sigma=absolute,
        )
    assert_allclose(covariance[1, 1], expected, rtol=2e-7)
    assert np.isnan(covariance[0]).all() and np.isnan(covariance[:, 0]).all()


def test_correlated_ratio_quantiles_by_gaussian_integration() -> None:
    """Check ratio quantiles, including the negative-denominator event exactly."""
    mean = np.array([2.0, 1.0])
    covariance = np.array([[0.04, 0.045], [0.045, 0.09]])
    sigma_b = np.sqrt(covariance[1, 1])
    conditional_sigma = np.sqrt(
        covariance[0, 0] - covariance[0, 1] ** 2 / covariance[1, 1]
    )
    zero_b = -mean[1] / sigma_b

    def cdf(ratio: float) -> float:
        """Integrate A|B, reversing the inequality when B is negative."""

        def integrand(z: float, negative: bool) -> float:
            """Evaluate the conditional event probability times B's density."""
            b = mean[1] + sigma_b * z
            conditional_mean = mean[0] + covariance[0, 1] / sigma_b * z
            standardized = (ratio * b - conditional_mean) / conditional_sigma
            probability = special.ndtr(-standardized if negative else standardized)
            return float(probability * np.exp(-z * z / 2) / np.sqrt(2 * np.pi))

        return float(
            integrate.quad(lambda z: integrand(z, True), -np.inf, zero_b, epsabs=1e-10)[
                0
            ]
            + integrate.quad(
                lambda z: integrand(z, False), zero_b, np.inf, epsabs=1e-10
            )[0]
        )

    result = inference.propagate_uncertainty(
        lambda a, b: a / b, mean, covariance, n_samples=40000, random_state=124
    )
    assert abs(cdf(result["lower"]) - 0.025) < 0.004
    assert abs(cdf(result["upper"]) - 0.975) < 0.004
    assert result["n_valid"] == 40000 and result["n_invalid"] == 0
    # A normal denominator permits values arbitrarily close to zero. Finite
    # sample moments are not claimed to estimate existing population moments.
    with pytest.warns(UserWarning, match="Singular"):
        deterministic = inference.propagate_uncertainty(
            lambda a, b: a / b,
            [2.0, 1.0],
            [[0.04, 0.02], [0.02, 0.01]],
            n_samples=1000,
            random_state=22,
        )
    assert_allclose(
        [deterministic["lower"], deterministic["upper"]], [2.0, 2.0], atol=1e-12
    )


def test_strongly_correlated_nonlinear_profiles() -> None:
    """Reoptimize nuisance parameters in a nearly collinear nonlinear model."""
    x = np.linspace(9.9, 10.1, 40)
    beta = np.array([3.0, 0.1])
    sigma = np.full(x.size, 1e-4)
    y = _exponential(x, *beta)
    design = _jacobian("exponential", x, beta) / sigma[:, None]
    covariance = np.linalg.solve(design.T @ design, np.eye(2))
    intervals = inference.profile_likelihood_errors(
        _exponential,
        x,
        y,
        beta,
        sigma=sigma,
        pcov=covariance,
        absolute_sigma=True,
        bounds=([1.0, 0.01], [5.0, 0.2]),
    )

    def reference_profile(index: int, value: float) -> float:
        """Eliminate amplitude exactly or optimize rate with a scalar solver."""
        if index == 1:
            basis = np.exp(-value * x)
            amplitude = float(basis @ y / (basis @ basis))
            return float(
                np.sum(((_exponential(x, amplitude, value) - y) / sigma) ** 2) - 1
            )
        result = optimize.minimize_scalar(
            lambda rate: np.sum(((_exponential(x, value, rate) - y) / sigma) ** 2),
            method="bounded",
            bounds=(0.01, 0.2),
            options={"xatol": 1e-12},
        )
        assert result.success
        return float(result.fun - 1)

    for index, limits in enumerate(((1.0, 5.0), (0.01, 0.2))):
        for direction, bound in [("down", limits[0]), ("up", limits[1])]:
            bracket = sorted((float(beta[index]), bound))
            root = optimize.brentq(
                lambda value, j=index: reference_profile(j, value),
                bracket[0],
                bracket[1],
                xtol=1e-12,
            )
            endpoint = beta[index] + intervals[index][direction]
            assert_allclose(endpoint, root, rtol=2e-6)
            assert abs(reference_profile(index, endpoint)) < 3e-5
            frozen = beta.copy()
            frozen[index] = endpoint
            assert np.sum(((_exponential(x, *frozen) - y) / sigma) ** 2) > 1000


def test_bootstrap_actual_solver_convergence_failures() -> None:
    """Use a real optimizer evaluation limit to exercise replicate failure counts."""
    x = np.linspace(0.2, 5.0, 30)
    y = _michaelis(x, 3.0, 1.2) + np.random.default_rng(81).normal(0, 0.03, x.size)
    beta = _variable_projection(_michaelis, x, y, np.ones(x.size), (0.1, 4.0))
    # The base fit is already optimal; noisy refits cannot converge in one
    # evaluation. This exercises actual SciPy failures without mocking a solver.
    with (
        pytest.warns(UserWarning, match="10 of 10"),
        pytest.raises(RuntimeError, match="two successful"),
    ):
        inference.bootstrap_curve_fit(
            _michaelis,
            x,
            y,
            beta,
            n_resamples=10,
            random_state=452,
            method="trf",
            max_nfev=1,
            gtol=1e-5,
        )


def test_zero_residual_variance_semantics() -> None:
    """Separate degenerate empirical covariance from undefined variance profiling."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2 * x

    def model(x: _Array, slope: float) -> _Array:
        """Evaluate the single-column exact linear model."""
        return slope * x

    mean, covariance, samples = inference.bootstrap_curve_fit(
        model, x, y, [2.0], n_resamples=10, random_state=482
    )
    assert_array_equal(samples, np.full((10, 1), 2.0))
    assert_array_equal(mean, [2.0])
    assert_array_equal(covariance, [[0.0]])
    with pytest.raises(ValueError, match="zero residual variance"):
        inference.profile_likelihood_errors(model, x, y, [2.0])
    intervals = inference.profile_likelihood_errors(
        model, x, y, [2.0], absolute_sigma=True
    )
    assert_allclose(
        [intervals[0]["down"], intervals[0]["up"]],
        [-1 / np.sqrt(30), 1 / np.sqrt(30)],
        rtol=2e-6,
    )
