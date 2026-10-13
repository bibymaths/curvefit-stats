"""Independent analytical and pathological tests for Phase 1 inference."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from scipy import optimize, stats

from curvefit_stats import inference as inf

Array = NDArray[np.float64]


def linear(x: Array, slope: float, intercept: float) -> Array:
    """Evaluate a line."""
    return slope * x + intercept


def linear_jac(x: Array, slope: float, intercept: float) -> Array:
    """Return the analytical line Jacobian."""
    return np.column_stack((x, np.ones_like(x)))


def through_origin(x: Array, slope: float) -> Array:
    """Evaluate a line through zero."""
    return slope * x


@pytest.fixture
def line_data() -> tuple[Array, Array, Array, Array]:
    """Provide reproducible fitting data."""
    x = np.linspace(-1, 3, 30)
    y = 2.5 * x + 0.8 + np.random.default_rng(123).normal(0, 0.1, x.size)
    beta, covariance = optimize.curve_fit(linear, x, y)
    return x, y, beta, covariance


def test_public_surface() -> None:
    """Public surface."""
    assert set(inf.__all__) == {
        "generate_error_bands",
        "audit_fit",
        "fit_summary",
        "profile_likelihood_errors",
        "bootstrap_curve_fit",
        "collinearity_diagnostics",
        "robust_bounded_pcov",
        "propagate_uncertainty",
    }


def test_linear_bands_and_prediction_grid(
    line_data: tuple[Array, Array, Array, Array],
) -> None:
    """Linear bands and prediction grid."""
    x, y, beta, covariance = line_data
    grid = np.linspace(-2, 5, 71)
    design = np.column_stack((grid, np.ones_like(grid)))
    width = stats.t.ppf(0.975, x.size - 2) * np.sqrt(
        np.diag(design @ covariance @ design.T)
    )
    lo, hi = inf.generate_error_bands(
        linear, grid, beta, covariance, y, x_fit=x, jac=linear_jac
    )
    assert_allclose(lo, linear(grid, *beta) - width)
    assert_allclose(hi, linear(grid, *beta) + width)
    residual_variance = np.sum((y - linear(x, *beta)) ** 2) / (x.size - 2)
    expected = stats.t.ppf(0.975, x.size - 2) * np.sqrt(
        (width / stats.t.ppf(0.975, x.size - 2)) ** 2 + residual_variance
    )
    plo, phi = inf.generate_error_bands(
        linear, grid, beta, covariance, y, interval_type="prediction", x_fit=x
    )
    assert_allclose((phi - plo) / 2, expected, rtol=1e-8)
    with pytest.raises(ValueError, match="coordinate"):
        inf.generate_error_bands(linear, grid, beta, covariance, y)
    with pytest.raises(ValueError, match="dof"):
        inf.generate_error_bands(linear, grid, beta, covariance, y, x_fit=x, dof=69)


def test_nonlinear_jacobian_and_known_noise() -> None:
    """Nonlinear jacobian and known noise."""
    x = np.linspace(0, 4, 13)
    beta = np.array([2.0, 0.7])
    covariance = np.array([[0.04, -0.003], [-0.003, 0.002]])

    def exponential(x: Array, amplitude: float, rate: float) -> Array:
        """Exponential."""
        return amplitude * np.exp(-rate * x)

    derivative = np.column_stack(
        (np.exp(-beta[1] * x), -beta[0] * x * np.exp(-beta[1] * x))
    )
    variance = np.diag(derivative @ covariance @ derivative.T)
    lower, upper = inf.generate_error_bands(
        exponential, x, beta, covariance, reference="normal"
    )
    assert_allclose(
        (upper - lower) / 2, stats.norm.ppf(0.975) * np.sqrt(variance), rtol=1e-8
    )
    pv = np.linspace(0.01, 0.04, x.size)
    lower, upper = inf.generate_error_bands(
        exponential,
        x,
        beta,
        covariance,
        reference="normal",
        interval_type="prediction",
        prediction_variance=pv,
    )
    assert_allclose(
        (upper - lower) / 2, stats.norm.ppf(0.975) * np.sqrt(variance + pv), rtol=1e-8
    )


def test_scalar_and_multicoordinate_bands() -> None:
    """Scalar and multicoordinate bands."""
    lo, hi = inf.generate_error_bands(
        through_origin,
        2.0,
        [3.0],
        [[0.1]],
        dof=3,
        jac=lambda x, a: np.array([float(x)]),
    )
    assert lo.shape == hi.shape == (1,)
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lo, hi = inf.generate_error_bands(
        lambda x, a: a * x[0] + x[1], x, [2.0], [[0.1]], dof=8
    )
    assert lo.shape == hi.shape == (3,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": np.nan},
        {"interval_type": "other"},
        {"reference": "other"},
        {"dof": 0},
        {"dof": True},
        {"dof": 2.5},
        {"prediction_variance": 1},
        {"x_fit": [1.0]},
        {"jac": lambda x, a: np.zeros((2, 2))},
    ],
)
def test_band_validation(kwargs: dict[str, Any]) -> None:
    """Band validation."""
    options = {"dof": 5} | kwargs
    with pytest.raises(ValueError):
        inf.generate_error_bands(through_origin, [1.0, 2.0], [2.0], [[0.1]], **options)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"reference": "normal"},
        {"prediction_variance": -1},
        {"prediction_variance": [0.1]},
        {"prediction_variance": [[0.1, 0.2]]},
    ],
)
def test_prediction_validation(kwargs: dict[str, Any]) -> None:
    """Prediction validation."""
    with pytest.raises(ValueError):
        inf.generate_error_bands(
            through_origin,
            [1.0, 2.0],
            [2.0],
            [[0.1]],
            interval_type="prediction",
            dof=4,
            **kwargs,
        )


@pytest.mark.parametrize(
    "covariance",
    [
        [[-0.1]],
        [[np.inf]],
        [[0.1, 0.0]],
        [[1.0, 0.2], [0.1, 1.0]],
        [[1.0, 2.0], [2.0, 1.0]],
    ],
)
def test_covariance_validation(covariance: list[list[float]]) -> None:
    """Covariance validation."""
    beta = [2.0, 1.0] if len(covariance) == 2 else [2.0]
    model = linear if len(beta) == 2 else through_origin
    with pytest.raises(ValueError):
        inf.generate_error_bands(model, [1.0, 2.0], beta, covariance, dof=5)


def test_covariance_roundoff_and_nonmutation() -> None:
    """Covariance roundoff and nonmutation."""
    covariance = np.array([[1.0, 1.0], [1.0, 1.0 - 1e-15]])
    original = covariance.copy()
    with pytest.warns(UserWarning, match="roundoff"):
        lo, hi = inf.generate_error_bands(
            linear, [1.0, 2.0], [2.0, 1.0], covariance, dof=3
        )
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert_array_equal(covariance, original)


@pytest.mark.parametrize("sigma_kind", ["none", "scalar", "vector", "matrix"])
@pytest.mark.parametrize("absolute", [False, True])
def test_audit_gaussian_likelihood(sigma_kind: str, absolute: bool) -> None:
    """Audit gaussian likelihood."""
    x = np.arange(5.0)
    beta = np.array([1.8, 0.4])
    residual = np.array([0.2, -0.1, 0.3, -0.2, 0.1])
    y = linear(x, *beta) + residual
    sigma: Any = None
    covariance = np.eye(5)
    if sigma_kind == "scalar":
        sigma = 0.2
        covariance *= sigma**2
    elif sigma_kind == "vector":
        sigma = np.linspace(0.1, 0.3, 5)
        covariance = np.diag(sigma**2)
    elif sigma_kind == "matrix":
        covariance = 0.02 * np.eye(5) + 0.01 * np.ones((5, 5))
        sigma = covariance
    # Independent reference via multivariate_normal, including normalization.
    wss = residual @ np.linalg.solve(covariance, residual)
    likelihood_covariance = covariance if absolute else covariance * wss / 5
    log_likelihood = stats.multivariate_normal.logpdf(
        residual, mean=np.zeros(5), cov=likelihood_covariance
    )
    k = 2 if absolute else 3
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = inf.audit_fit(linear, x, y, beta, sigma=sigma, absolute_sigma=absolute)
    assert len(captured) == int(sigma is not None and not absolute)
    if captured:
        assert "valid relative-weighting" in str(captured[0].message)
    assert_allclose(result["aic"], -2 * log_likelihood + 2 * k)
    assert_allclose(result["bic"], -2 * log_likelihood + k * np.log(5))
    assert_allclose(result["residual_variance"], residual @ residual / 3)
    if sigma is not None or absolute:
        assert_allclose(result["chi_square"], wss)
        assert_allclose(result["reduced_chi_square"], wss / 3)
    else:
        assert np.isnan(result["chi_square"])


def test_audit_undefined() -> None:
    """Audit undefined."""
    with pytest.warns(UserWarning, match="N <= p"):
        result = inf.audit_fit(
            linear, [1.0, 2.0], [2.1, 4.0], [2.0, 0.0], absolute_sigma=True
        )
    assert np.isnan(result["residual_variance"])
    assert np.isnan(result["reduced_chi_square"])
    with pytest.warns(UserWarning, match="unbounded"):
        result = inf.audit_fit(through_origin, [1.0, 2.0], [2.0, 4.0], [2.0])
    assert np.isnan(result["aic"])
    result = inf.audit_fit(
        through_origin, [1.0, 2.0], [2.0, 4.0], [2.0], absolute_sigma=True
    )
    assert np.isfinite(result["aic"])


@pytest.mark.parametrize(
    "sigma",
    [
        0.0,
        -1.0,
        [1.0],
        [1.0, 0.0],
        [[1.0, 0.0]],
        [[1.0, 2.0], [2.0, 1.0]],
        [[1.0, 0.1], [0.2, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        [np.nan, 1.0],
    ],
)
def test_invalid_noise(sigma: Any) -> None:
    """Invalid noise."""
    with pytest.raises(ValueError):
        inf.audit_fit(through_origin, [1.0, 2.0], [2.1, 4.0], [2.0], sigma=sigma)


def test_summary_statistics_and_names(
    line_data: tuple[Array, Array, Array, Array],
) -> None:
    """Summary statistics and names."""
    x, y, beta, covariance = line_data
    text = inf.fit_summary(linear, x, y, beta, covariance, precision=12)
    assert "N = 30; residual dof = 28; confidence = 95%" in text
    rows = text.splitlines()[4:]
    se = np.sqrt(np.diag(covariance))
    for index, row in enumerate(rows):
        fields = [value.strip() for value in row.split("|")]
        expected = [
            beta[index],
            se[index],
            beta[index] / se[index],
            2 * stats.t.sf(abs(beta[index] / se[index]), 28),
            beta[index] - stats.t.ppf(0.975, 28) * se[index],
            beta[index] + stats.t.ppf(0.975, 28) * se[index],
        ]
        assert_allclose([float(value) for value in fields[1:]], expected, rtol=1e-10)

    def option_model(x: Array, slope: float, *, scale: float = 1.0) -> Array:
        """Option model."""
        return x * slope * scale

    assert "slope" in inf.fit_summary(
        option_model, [1.0, 2.0], [2.0, 4.0], [2.0], [[0.1]]
    )
    assert "p0" in inf.fit_summary(
        lambda x, *args: x * args[0], [1.0, 2.0], [2.0, 4.0], [2.0], [[0.1]]
    )
    assert "chosen" in inf.fit_summary(
        through_origin, [1.0, 2.0], [2.0, 4.0], [2.0], [[0.1]], ["chosen"]
    )
    with pytest.raises(ValueError):
        inf.fit_summary(linear, x, y, beta, covariance, ["only_one"])


@pytest.mark.parametrize(
    "covariance, data, message",
    [
        ([[np.inf]], ([1.0, 2.0], [2.0, 4.0]), "Non-finite"),
        ([[0.0]], ([1.0, 2.0], [2.0, 4.0]), "Zero standard"),
        ([[0.1]], ([1.0], [2.0]), "degrees of freedom"),
    ],
)
def test_summary_undefined(covariance: Any, data: Any, message: str) -> None:
    """Summary undefined."""
    with pytest.warns(UserWarning, match=message):
        text = inf.fit_summary(through_origin, data[0], data[1], [2.0], covariance)
    assert "NaN" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"precision": 0},
        {"precision": True},
        {"alpha": 1},
        {"param_names": [""]},
        {"pcov": [[1.0, 0.0]]},
        {"pcov": [[-1.0]]},
    ],
)
def test_summary_invalid(kwargs: dict[str, Any]) -> None:
    """Summary invalid."""
    parameters = {"pcov": [[0.1]]} | kwargs
    with pytest.raises(ValueError):
        inf.fit_summary(through_origin, [1.0, 2.0], [2.0, 4.0], [2.0], **parameters)


@pytest.mark.parametrize("absolute", [True, False])
def test_profile_linear_exact(
    absolute: bool, line_data: tuple[Array, Array, Array, Array]
) -> None:
    """Profile linear exact."""
    x, y, beta, covariance = line_data
    sigma = np.full(x.size, 0.1)
    design = np.column_stack((x, np.ones_like(x)))
    inv_information = np.linalg.solve(design.T @ design / 0.01, np.eye(2))
    wss = np.sum(((y - linear(x, *beta)) / 0.1) ** 2)
    increase = 1 if absolute else wss * np.expm1(1 / x.size)
    expected = np.sqrt(np.diag(inv_information) * increase)
    intervals = inf.profile_likelihood_errors(
        linear, x, y, beta, sigma=sigma, absolute_sigma=absolute, pcov=covariance
    )
    assert_allclose([value["up"] for value in intervals], expected, rtol=2e-5)
    assert_allclose([value["down"] for value in intervals], -expected, rtol=2e-5)


def test_profile_single_parameter_and_correlated_noise() -> None:
    """Profile single parameter and correlated noise."""
    x = np.arange(1.0, 6.0)
    covariance = 0.04 * np.eye(5) + 0.01 * np.ones((5, 5))
    y = 2 * x + np.array([0.1, -0.2, 0.2, -0.1, 0.05])
    optimum = np.array(
        [(x @ np.linalg.solve(covariance, y)) / (x @ np.linalg.solve(covariance, x))]
    )
    expected = 1 / np.sqrt(x @ np.linalg.solve(covariance, x))
    result = inf.profile_likelihood_errors(
        through_origin, x, y, optimum, sigma=covariance, absolute_sigma=True
    )
    assert_allclose(result[0]["up"], expected, rtol=1e-5)
    assert_allclose(result[0]["down"], -expected, rtol=1e-5)


def test_profile_bounds_flat_and_fixed() -> None:
    """Profile bounds flat and fixed."""
    with pytest.warns(UserWarning, match="not resolved"):
        result = inf.profile_likelihood_errors(
            through_origin,
            [1.0, 2.0],
            [2.0, 4.0],
            [2.0],
            bounds=([1.99], [2.01]),
            absolute_sigma=True,
        )
    assert np.isnan(result[0]["up"]) and np.isnan(result[0]["down"])
    with pytest.warns(UserWarning, match="boundary-limited"):
        result = inf.profile_likelihood_errors(
            through_origin,
            [1.0, 2.0],
            [2.0, 4.0],
            [2.0],
            bounds=([2.0], [2.0]),
            absolute_sigma=True,
        )
    assert np.isnan(result[0]["up"])
    with pytest.warns(UserWarning, match="not resolved"):
        result = inf.profile_likelihood_errors(
            lambda x, a: np.ones_like(x),
            [1.0, 2.0],
            [1.1, 0.9],
            [1.0],
            max_bracket_steps=3,
        )
    assert np.isnan(result[0]["up"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_level": 0},
        {"sigma_level": np.inf},
        {"sigma_level": 1e300},
        {"tolerance": 0},
        {"tolerance": 2},
        {"max_nfev": 0},
        {"max_bracket_steps": 0},
        {"bounds": ([3.0], [4.0])},
        {"bounds": ([3.0], [1.0])},
        {"bounds": ([np.nan], [4.0])},
        {"bounds": ([1.0, 1.0], [4.0, 4.0])},
    ],
)
def test_profile_invalid(kwargs: dict[str, Any]) -> None:
    """Profile invalid."""
    with pytest.raises(ValueError):
        inf.profile_likelihood_errors(
            through_origin, [1.0, 2.0], [2.1, 3.9], [1.98], **kwargs
        )


def test_profile_zero_sse_and_suboptimal() -> None:
    """Profile zero sse and suboptimal."""
    with pytest.raises(ValueError, match="zero residual"):
        inf.profile_likelihood_errors(through_origin, [1.0, 2.0], [2.0, 4.0], [2.0])
    with pytest.raises(ValueError, match="not a local optimum"):
        inf.profile_likelihood_errors(
            through_origin, [1.0, 2.0], [2.0, 4.0], [1.0], absolute_sigma=True
        )


def test_profile_rejects_near_collinear_false_convergence() -> None:
    """Reject a suboptimal baseline with nearly collinear parameters."""

    def model(
        x: np.ndarray,
        a: float,
        b: float,
    ) -> np.ndarray:
        return a + b * (1.0 + 1e-10 * x)

    x = np.array([1.0, -1.0])
    y = np.array([10.0, -10.0])
    popt = np.array([0.0, 0.0])

    baseline_sse = float(np.sum((y - model(x, *popt)) ** 2))

    assert baseline_sse == pytest.approx(200.0)

    # A substantially better feasible parameter vector exists.
    alternative = np.array([-1e11, 1e11])
    alternative_sse = float(np.sum((y - model(x, *alternative)) ** 2))

    assert alternative_sse < 1e-6

    with pytest.raises(
        ValueError,
        match="not a local optimum",
    ):
        inf.profile_likelihood_errors(
            model,
            x,
            y,
            popt,
            sigma=np.ones_like(y),
            absolute_sigma=True,
        )


def test_bootstrap_reproducibility_and_manual_reference() -> None:
    """Bootstrap reproducibility and manual reference."""
    x = np.arange(1.0, 8.0)
    y = 2 * x + np.array([0.1, -0.3, 0.15, -0.1, 0.2, -0.05, 0.1])
    initial = np.array([1.9])
    original = initial.copy()
    np.random.seed(7)
    state = np.random.get_state()
    mean, covariance, samples = inf.bootstrap_curve_fit(
        through_origin, x, y, initial, n_resamples=40, random_state=12
    )
    assert_array_equal(np.random.get_state()[1], state[1])
    assert_array_equal(initial, original)
    base = x @ y / (x @ x)
    residual = y - base * x
    residual -= residual.mean()
    generator = np.random.default_rng(12)
    reference = np.array(
        [
            x @ (base * x + generator.choice(residual, size=7)) / (x @ x)
            for _ in range(40)
        ]
    )
    assert_allclose(samples[:, 0], reference, rtol=1e-8)
    assert_allclose(mean, [reference.mean()])
    assert_allclose(covariance, [[reference.var(ddof=1)]])
    other = inf.bootstrap_curve_fit(
        through_origin, x, y, n_resamples=40, random_state=12
    )
    assert_allclose(other[2], samples)


def test_bootstrap_nonlinear_bounded() -> None:
    """Bootstrap nonlinear bounded."""
    x = np.linspace(0, 5, 40)

    def decay(x: Array, amplitude: float, rate: float) -> Array:
        """Decay."""
        return amplitude * np.exp(-rate * x)

    y = decay(x, 3, 0.6) + np.random.default_rng(12).normal(0, 0.03, x.size)
    mean, covariance, samples = inf.bootstrap_curve_fit(
        decay,
        x,
        y,
        [2.0, 0.5],
        n_resamples=30,
        bounds=([0.0, 0.0], [5.0, 2.0]),
        random_state=1,
    )
    assert_allclose(mean, [3, 0.6], atol=0.03)
    assert samples.shape == (30, 2) and covariance.shape == (2, 2)
    assert np.all((samples >= 0) & (samples <= [5.0, 2.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_resamples": 1},
        {"n_resamples": True},
        {"sigma": [1.0, 1.0]},
        {"full_output": True},
        {"nan_policy": "omit"},
        {"check_finite": False},
    ],
)
def test_bootstrap_invalid(kwargs: dict[str, Any]) -> None:
    """Bootstrap invalid."""
    with pytest.raises(ValueError):
        inf.bootstrap_curve_fit(through_origin, [1.0, 2.0], [2.0, 4.0], [2.0], **kwargs)


def test_bootstrap_failure_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bootstrap failure counts."""
    original = optimize.curve_fit
    calls = 0

    def fail_some(*args: Any, **kwargs: Any) -> Any:
        """Fail some."""
        nonlocal calls
        calls += 1
        if calls in (2, 3):
            raise RuntimeError("controlled nonconvergence")
        return original(*args, **kwargs)

    monkeypatch.setattr(optimize, "curve_fit", fail_some)
    with pytest.warns(UserWarning) as captured:
        _, covariance, samples = inf.bootstrap_curve_fit(
            through_origin, [1.0, 2.0, 3.0], [2.1, 3.9, 6.1], [2.0], n_resamples=5
        )
    assert samples.shape == (3, 1) and covariance.shape == (1, 1)
    assert len(captured) == 2 and "2 of 5" in str(captured[0].message)
    calls = 0
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="two successful"):
        inf.bootstrap_curve_fit(
            through_origin, [1.0, 2.0, 3.0], [2.1, 3.9, 6.1], [2.0], n_resamples=2
        )


def test_collinearity_analytical_and_singular() -> None:
    """Collinearity analytical and singular."""
    result = inf.collinearity_diagnostics([[4.0, 1.0], [1.0, 1.0]])
    assert_allclose(result["correlation_matrix"], [[1.0, 0.5], [0.5, 1.0]])
    assert_allclose(result["vif"], [4 / 3, 4 / 3])
    assert_allclose(result["condition_number"], 3)
    assert result["rank"] == 2
    assert not result["potential_identifiability_issue"]
    with pytest.warns(UserWarning, match="Strong local"):
        strong = inf.collinearity_diagnostics([[1.0, -0.999], [-0.999, 1.0]])
    assert strong["high_correlation_pairs"] == [(0, 1)]
    with pytest.warns(UserWarning):
        singular = inf.collinearity_diagnostics([[1.0, 1.0], [1.0, 1.0]])
    assert np.isinf(singular["vif"]).all() and singular["rank"] == 1
    with pytest.warns(UserWarning, match="Zero parameter"):
        partial = inf.collinearity_diagnostics([[1.0, 0.0], [0.0, 0.0]])
    assert np.isnan(partial["vif"][1]) and partial["rank"] == 1
    with pytest.warns(UserWarning, match="Non-finite"):
        invalid = inf.collinearity_diagnostics([[np.inf]])
    assert invalid["potential_identifiability_issue"]
    single = inf.collinearity_diagnostics([[1.0]])
    assert single["max_abs_correlation"] == 0


@pytest.mark.parametrize(
    "pcov, threshold",
    [
        ([], 0.99),
        ([[1.0, 2.0]], 0.99),
        ([[1.0]], 1.0),
        ([[1.0]], 0.0),
        ([[1.0]], np.nan),
        ([[-1.0]], 0.99),
    ],
)
def test_collinearity_invalid(pcov: Any, threshold: float) -> None:
    """Collinearity invalid."""
    with pytest.raises(ValueError):
        inf.collinearity_diagnostics(pcov, correlation_threshold=threshold)


@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("matrix_noise", [True, False])
def test_bounded_covariance_matches_weighted_linear(
    absolute: bool, matrix_noise: bool
) -> None:
    """Bounded covariance matches weighted linear."""
    x = np.linspace(-1, 3, 15)
    y = 2 * x + 1 + np.random.default_rng(3).normal(0, 0.1, x.size)
    covariance = (
        0.02 * np.eye(x.size) + 0.005 * np.ones((x.size, x.size))
        if matrix_noise
        else np.diag(np.linspace(0.1, 0.2, x.size) ** 2)
    )
    sigma = covariance if matrix_noise else np.sqrt(np.diag(covariance))
    design = np.column_stack((x, np.ones_like(x)))
    information = design.T @ np.linalg.solve(covariance, design)
    beta = np.linalg.solve(information, design.T @ np.linalg.solve(covariance, y))
    reference = np.linalg.solve(information, np.eye(2))
    if not absolute:
        residual = y - design @ beta
        reference *= residual @ np.linalg.solve(covariance, residual) / (x.size - 2)
    result = inf.robust_bounded_pcov(
        linear,
        x,
        y,
        beta,
        (-np.inf, np.inf),
        sigma=sigma,
        absolute_sigma=absolute,
        jac=linear_jac,
    )
    numeric = inf.robust_bounded_pcov(
        linear, x, y, beta, (-np.inf, np.inf), sigma=sigma, absolute_sigma=absolute
    )
    assert_allclose(result, reference, rtol=1e-12)
    assert_allclose(numeric, reference, rtol=1e-8)


def test_bounded_active_and_effective_dof() -> None:
    """Bounded active and effective dof."""
    x = np.arange(4.0)
    y = 2 * x + 1 + np.array([0.1, -0.2, 0.1, 0.3])
    with pytest.warns(UserWarning, match="conditional"):
        result = inf.robust_bounded_pcov(
            linear, x, y, [2.0, 1.0], ([-np.inf, 1.0], [np.inf, 1.0])
        )
    assert np.isnan(result[1]).all() and np.isnan(result[:, 1]).all()
    expected = np.sum((y - linear(x, 2, 1)) ** 2) / 3 / np.sum(x * x)
    assert_allclose(result[0, 0], expected, rtol=1e-8)
    with pytest.warns(UserWarning):
        all_active = inf.robust_bounded_pcov(
            linear, x, y, [2.0, 1.0], ([2.0, 1.0], [2.0, 1.0])
        )
    assert np.isnan(all_active).all()
    with pytest.warns(UserWarning):
        near = inf.robust_bounded_pcov(
            linear, x, y, [2.0, 1.0 + 1e-10], ([-np.inf, 1.0], [np.inf, 2.0])
        )
    assert np.isnan(near[1]).all()
    with pytest.raises(ValueError):
        inf.robust_bounded_pcov(
            linear, x, y, [2.0, 1.0], (-np.inf, np.inf), bound_tolerance=-1
        )


def test_rank_deficiency_and_nonpositive_dof() -> None:
    """Rank deficiency and nonpositive dof."""
    with pytest.warns(UserWarning, match="rank deficient"):
        result = inf.robust_bounded_pcov(
            lambda x, a, b: (a + b) * x,
            [1.0, 2.0, 3.0],
            [2.1, 3.9, 6.0],
            [1.0, 1.0],
            (-np.inf, np.inf),
            jac=lambda x, a, b: np.column_stack((x, x)),
        )
    assert np.isnan(result).all()
    with pytest.warns(UserWarning, match="degrees of freedom"):
        result = inf.robust_bounded_pcov(
            through_origin, [1.0], [2.0], [2.0], (-np.inf, np.inf)
        )
    assert np.isnan(result).all()


def test_propagation_linear_correlated_and_seed() -> None:
    """Propagation linear correlated and seed."""
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    result = inf.propagate_uncertainty(
        lambda a, b: 2 * a - b, [1.0, 3.0], covariance, n_samples=30000, random_state=8
    )
    assert result["n_valid"] == result["n_samples"] == 30000
    assert result["n_invalid"] == 0
    assert abs(result["mean"] + 1) < 0.01
    standard = np.sqrt(4 * 0.04 + 0.09 - 4 * 0.01)
    assert_allclose(result["std"], standard, rtol=0.02)
    assert_allclose(
        [result["lower"], result["upper"]],
        [-1 - stats.norm.ppf(0.975) * standard, -1 + stats.norm.ppf(0.975) * standard],
        atol=0.025,
    )
    other = inf.propagate_uncertainty(
        lambda a, b: 2 * a - b, [1.0, 3.0], covariance, n_samples=30000, random_state=8
    )
    assert result == other


def test_propagation_vector_ratio_and_singular() -> None:
    """Propagation vector ratio and singular."""
    result = inf.propagate_uncertainty(
        lambda a, b: np.array([a + b, a - b, a / b]),
        [1.0, 3.0],
        [[0.001, 0.0], [0.0, 0.001]],
        n_samples=3000,
        random_state=5,
    )
    assert result["mean"].shape == (3,)
    assert_allclose(result["mean"], [4.0, -2.0, 1 / 3], atol=0.005)
    with pytest.warns(UserWarning, match="Singular"):
        singular = inf.propagate_uncertainty(
            lambda a, b: a - b, [1.0, 1.0], [[1.0, 1.0], [1.0, 1.0]], n_samples=10
        )
    assert singular["std"] == 0
    with pytest.warns(UserWarning, match="Singular"):
        zero = inf.propagate_uncertainty(lambda a: a + 1, [1.0], [[0.0]], n_samples=2)
    assert zero["lower"] == zero["upper"] == 2


def test_propagation_rejections_and_shapes() -> None:
    """Propagation rejections and shapes."""
    with (
        pytest.warns(UserWarning, match="non-finite"),
        pytest.raises(ValueError, match="Too few"),
    ):
        inf.propagate_uncertainty(
            lambda a: np.sqrt(a), [0.0], [[1.0]], n_samples=100, random_state=4
        )
    with pytest.warns(UserWarning, match="non-finite"):
        result = inf.propagate_uncertainty(
            lambda a: np.sqrt(a),
            [0.0],
            [[1.0]],
            n_samples=100,
            random_state=4,
            min_valid_fraction=0.3,
        )
    assert result["n_valid"] + result["n_invalid"] == 100
    assert result["n_invalid"] > 0
    with pytest.raises(ValueError, match="scalar or"):
        inf.propagate_uncertainty(
            lambda a: np.ones((2, 2)), [1.0], [[1.0]], n_samples=2
        )
    with pytest.raises(ValueError, match="shape must be constant"):
        inf.propagate_uncertainty(
            lambda a: np.ones(1 if a < 0 else 2),
            [0.0],
            [[1.0]],
            n_samples=30,
            random_state=7,
        )
    with pytest.raises(ZeroDivisionError):
        inf.propagate_uncertainty(lambda a: 1 / 0, [1.0], [[1.0]], n_samples=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 1},
        {"n_samples": 1},
        {"n_samples": True},
        {"min_valid_fraction": 0},
        {"min_valid_fraction": 1.1},
        {"min_valid_fraction": np.nan},
    ],
)
def test_propagation_invalid(kwargs: dict[str, Any]) -> None:
    """Propagation invalid."""
    with pytest.raises(ValueError):
        inf.propagate_uncertainty(lambda a: a, [1.0], [[1.0]], **kwargs)


@pytest.mark.parametrize(
    "x, y, beta, model",
    [
        ([], [], [2.0], through_origin),
        ([1.0, 2.0], [2.0], [2.0], through_origin),
        ([1.0, np.nan], [2.0, 4.0], [2.0], through_origin),
        ([1.0, 2.0], [2.0, np.inf], [2.0], through_origin),
        ([1.0, 2.0], [2.0, 4.0], [np.nan], through_origin),
        ([1.0, 2.0], [2.0, 4.0], [], through_origin),
        ([1.0, 2.0], [[2.0, 4.0]], [2.0], through_origin),
        ([1.0, 2.0], [2.0, 4.0], [2.0], lambda x, a: np.array([1.0])),
        ([1.0, 2.0], [2.0, 4.0], [2.0], lambda x, a: np.full(2, np.nan)),
    ],
)
def test_invalid_data(x: Any, y: Any, beta: Any, model: Any) -> None:
    """Invalid data."""
    with pytest.raises(ValueError):
        inf.audit_fit(model, x, y, beta)


def test_disparate_parameter_scales_and_numerical_domain() -> None:
    """Refine differences for slow rates and small positive domains."""
    x = np.linspace(0, 2e6, 20)
    beta = np.array([1e6, 1e-6])
    covariance = np.diag([1e4, 1e-16])

    def slow_decay(x: Array, amplitude: float, rate: float) -> Array:
        """Evaluate a scaled exponential."""
        return amplitude * np.exp(-rate * x)

    derivative = np.column_stack(
        (np.exp(-beta[1] * x), -beta[0] * x * np.exp(-beta[1] * x))
    )
    reference = np.sqrt(np.diag(derivative @ covariance @ derivative.T))
    lo, hi = inf.generate_error_bands(
        slow_decay, x, beta, covariance, reference="normal"
    )
    assert_allclose((hi - lo) / (2 * stats.norm.ppf(0.975)), reference, rtol=1e-6)
    # Initial central perturbations cross the logarithm's domain; refinement
    # must find a valid stencil, not fabricate a derivative or covariance.
    lo, hi = inf.generate_error_bands(
        lambda x, a: np.log(a) * x, [1.0, 2.0], [1e-10], [[1e-24]], reference="normal"
    )
    assert_allclose((hi - lo) / (2 * stats.norm.ppf(0.975)), [0.01, 0.02], rtol=1e-6)
    with pytest.raises(ValueError, match="did not stabilize"):
        inf.generate_error_bands(
            lambda x, a: np.where(a == 0, x, np.nan), [1.0, 2.0], [0.0], [[1.0]], dof=3
        )


@pytest.mark.parametrize("kind", ["michaelis", "logistic", "gaussian"])
def test_scientific_models_analytic_bands(kind: str) -> None:
    """Compare independent analytic Jacobians for scientific models."""
    x = np.linspace(0.2, 4, 31)
    if kind == "michaelis":
        beta = np.array([3.0, 1.2])

        def model(x: Array, a: float, b: float) -> Array:
            """Evaluate saturation kinetics."""
            return a * x / (b + x)

        derivative = np.column_stack(
            (x / (beta[1] + x), -beta[0] * x / (beta[1] + x) ** 2)
        )
    elif kind == "logistic":
        beta = np.array([4.0, 0.8])

        def model(x: Array, a: float, b: float) -> Array:
            """Evaluate logistic growth with fixed midpoint."""
            return a / (1 + np.exp(-b * x))

        z = 1 / (1 + np.exp(-beta[1] * x))
        derivative = np.column_stack((z, beta[0] * x * z * (1 - z)))
    else:
        beta = np.array([2.0, 1.1])

        def model(x: Array, a: float, b: float) -> Array:
            """Evaluate a Gaussian peak with fixed center."""
            return a * np.exp(-0.5 * (x / b) ** 2)

        z = np.exp(-0.5 * (x / beta[1]) ** 2)
        derivative = np.column_stack((z, beta[0] * z * x**2 / beta[1] ** 3))
    covariance = np.array([[0.01, 0.003], [0.003, 0.004]])
    expected = stats.t.ppf(0.975, 25) * np.sqrt(
        np.diag(derivative @ covariance @ derivative.T)
    )
    lower, upper = inf.generate_error_bands(model, x, beta, covariance, dof=25)
    assert_allclose((upper - lower) / 2, expected, rtol=1e-7)


def test_nonlinear_profile_thresholds() -> None:
    """Validate nonlinear profile endpoints via a different nuisance solver."""
    x = np.linspace(0, 5, 35)

    def decay(x: Array, a: float, rate: float) -> Array:
        """Evaluate exponential decay."""
        return a * np.exp(-rate * x)

    y = decay(x, 3.0, 0.6) + np.random.default_rng(29).normal(0, 0.05, x.size)
    beta, covariance = optimize.curve_fit(decay, x, y, p0=[2.8, 0.5])
    base = np.sum(((y - decay(x, *beta)) / 0.05) ** 2)
    intervals = inf.profile_likelihood_errors(
        decay,
        x,
        y,
        beta,
        sigma=0.05,
        pcov=covariance,
        bounds=(0, np.inf),
        absolute_sigma=True,
    )
    for direction in ("up", "down"):
        amplitude = beta[0] + intervals[0][direction]
        minimized = optimize.minimize_scalar(
            lambda rate, a=amplitude: np.sum(((y - decay(x, a, rate)) / 0.05) ** 2),
            bounds=(0.1, 1.5),
            method="bounded",
            options={"xatol": 1e-12},
        )
        assert_allclose(minimized.fun - base, 1, atol=3e-5)
        rate = beta[1] + intervals[1][direction]
        z = np.exp(-rate * x)
        amplitude = (z @ y) / (z @ z)
        assert_allclose(np.sum(((y - amplitude * z) / 0.05) ** 2) - base, 1, atol=3e-5)


def test_profile_optimizer_failure_reporting(
    monkeypatch: pytest.MonkeyPatch, line_data: tuple[Array, Array, Array, Array]
) -> None:
    """Keep nuisance optimization failure distinct from valid crossings."""
    x, y, beta, covariance = line_data
    real = optimize.least_squares
    calls = 0

    def fail_nuisance(*args: Any, **kwargs: Any) -> Any:
        """Fail only after baseline optimality validation."""
        nonlocal calls
        calls += 1
        result = real(*args, **kwargs)
        if calls > 1:
            result.success = False
            result.message = "controlled nuisance failure"
        return result

    monkeypatch.setattr(optimize, "least_squares", fail_nuisance)
    with pytest.warns(UserWarning) as captured:
        result = inf.profile_likelihood_errors(linear, x, y, beta, pcov=covariance)
    assert all(np.isnan(item["up"]) and np.isnan(item["down"]) for item in result)
    assert any("controlled nuisance failure" in str(item.message) for item in captured)


def test_bootstrap_optimization_warning_and_small_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclude warned replicates and reject insufficient residual information."""
    with (
        pytest.warns(optimize.OptimizeWarning),
        pytest.raises(ValueError, match="N > p"),
    ):
        inf.bootstrap_curve_fit(through_origin, [1.0], [2.0], [2.0], n_resamples=3)
    real = optimize.curve_fit
    calls = 0

    def unreliable(*args: Any, **kwargs: Any) -> Any:
        """Mark one replicate covariance unreliable."""
        nonlocal calls
        calls += 1
        result = real(*args, **kwargs)
        if calls == 2:
            warnings.warn(
                "controlled singular covariance", optimize.OptimizeWarning, stacklevel=2
            )
        elif calls == 3:
            warnings.warn("other model warning", UserWarning, stacklevel=2)
        return result

    monkeypatch.setattr(optimize, "curve_fit", unreliable)
    with pytest.warns(UserWarning) as captured:
        _, covariance, samples = inf.bootstrap_curve_fit(
            through_origin, [1.0, 2.0, 3.0], [2.1, 3.9, 6.1], [2.0], n_resamples=4
        )
    assert samples.shape == (3, 1) and covariance.shape == (1, 1)
    assert any("other model warning" in str(item.message) for item in captured)


def test_all_inputs_unchanged(line_data: tuple[Array, Array, Array, Array]) -> None:
    """Preserve observation, parameter, and covariance arrays across APIs."""
    x, y, beta, covariance = line_data
    sigma = np.ones(x.size) * 0.1
    arrays = [x, y, beta, covariance, sigma]
    originals = [item.copy() for item in arrays]
    inf.generate_error_bands(linear, x, beta, covariance, y, interval_type="prediction")
    inf.audit_fit(linear, x, y, beta, sigma=sigma, absolute_sigma=True)
    inf.fit_summary(linear, x, y, beta, covariance)
    inf.profile_likelihood_errors(
        linear, x, y, beta, sigma=sigma, absolute_sigma=True, pcov=covariance
    )
    inf.bootstrap_curve_fit(linear, x, y, beta, n_resamples=5, random_state=1)
    inf.collinearity_diagnostics(covariance)
    inf.robust_bounded_pcov(linear, x, y, beta, (-np.inf, np.inf), sigma=sigma)
    inf.propagate_uncertainty(
        lambda a, b: a + b, beta, covariance, n_samples=20, random_state=1
    )
    for original, item in zip(originals, arrays, strict=True):
        assert_array_equal(original, item)


@pytest.mark.parametrize("entry", ["parameters", "covariance", "data", "derived"])
def test_complex_inputs_rejected(entry: str) -> None:
    """Reject imaginary components instead of silently casting to real."""
    with pytest.raises(ValueError, match="real-valued"):
        if entry == "parameters":
            inf.propagate_uncertainty(lambda a: a, [1 + 1j], [[1.0]])
        elif entry == "covariance":
            inf.fit_summary(through_origin, [1.0, 2.0], [2.0, 4.0], [2.0], [[1 + 1j]])
        elif entry == "data":
            inf.audit_fit(through_origin, [1.0, 2.0], [2 + 1j, 4.0], [2.0])
        else:
            inf.propagate_uncertainty(
                lambda a: complex(a, 1), [1.0], [[1.0]], n_samples=2
            )


def test_large_derived_values_without_intermediate_overflow() -> None:
    """Compute finite moments when naive sums or squared deviations overflow."""
    ordinary = inf.propagate_uncertainty(
        lambda a: a, [2.0], [[1.0]], n_samples=100, random_state=17
    )
    large = inf.propagate_uncertainty(
        lambda a: 1e200 * a, [2.0], [[1.0]], n_samples=100, random_state=17
    )
    for key in ["mean", "median", "std", "lower", "upper"]:
        assert np.isfinite(large[key])
        assert_allclose(large[key] / 1e200, ordinary[key], rtol=1e-14)


@pytest.mark.parametrize("entry", ["bands", "summary", "diagnostics", "propagation"])
def test_covariance_validity_is_independent_of_parameter_units(entry: str) -> None:
    """Do not hide a negative small-unit variance behind a large-unit variance."""
    covariance = [[1e20, 0.0], [0.0, -1.0]]
    with pytest.raises(ValueError, match="positive semidefinite|nonnegative"):
        if entry == "bands":
            inf.generate_error_bands(linear, [1.0, 2.0], [1.0, 1.0], covariance, dof=5)
        elif entry == "summary":
            inf.fit_summary(
                linear, [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 1.0], covariance
            )
        elif entry == "diagnostics":
            inf.collinearity_diagnostics(covariance)
        else:
            inf.propagate_uncertainty(
                lambda a, b: a + b, [1.0, 1.0], covariance, n_samples=10
            )


def test_indefinite_covariance_with_positive_disparate_diagonals() -> None:
    """Reject indefiniteness even when the small eigenvalue is tiny in raw units."""
    with pytest.raises(ValueError, match="positive semidefinite"):
        inf.propagate_uncertainty(
            lambda a, b: b, [0.0, 0.0], [[1e20, 2e10], [2e10, 1.0]], n_samples=10
        )
    with pytest.raises(ValueError):
        inf.collinearity_diagnostics([[0.0, 1e-20], [1e-20, 1.0]])


def test_extreme_confidence_quantiles() -> None:
    """Compute two-sided tails without rounding 1-alpha/2 to one."""
    alpha = 1e-20
    references: list[tuple[Literal["normal", "student-t"], float]] = [
        ("normal", float(stats.norm.isf(alpha / 2))),
        ("student-t", float(stats.t.isf(alpha / 2, 20))),
    ]
    for reference, critical in references:
        lower, upper = inf.generate_error_bands(
            through_origin,
            [1.0, 2.0],
            [0.0],
            [[1.0]],
            dof=20,
            reference=reference,
            alpha=alpha,
        )
        assert_allclose(upper, critical * np.array([1.0, 2.0]))
        assert_allclose(lower, -upper)
    text = inf.fit_summary(
        through_origin, [1.0, 2.0, 3.0], [0.1, -0.1, 0.1], [0.0], [[1.0]], alpha=alpha
    )
    assert "inf" not in text.lower()


def test_large_offset_profile_root_accuracy() -> None:
    """Solve displacements accurately even when parameter estimates are large."""
    x = np.arange(4.0)
    y = 1e9 + np.array([-0.3, 0.2, 0.3, -0.2])

    def constant(x: Array, level: float) -> Array:
        """Evaluate a constant model."""
        return np.full_like(x, level)

    intervals = inf.profile_likelihood_errors(
        constant, x, y, [1e9], pcov=[[0.04]], absolute_sigma=True
    )
    assert_allclose([intervals[0]["down"], intervals[0]["up"]], [-0.5, 0.5], atol=2e-6)


def test_profile_does_not_differentiate_exactly_fixed_parameters() -> None:
    """Do not perturb coordinates fixed at a model's domain boundary."""

    def fixed_model(x: Array, a: float) -> Array:
        """Evaluate only at the exactly fixed parameter value."""
        return x if a == 0 else np.full_like(x, np.nan)

    with pytest.warns(UserWarning, match="boundary-limited"):
        intervals = inf.profile_likelihood_errors(
            fixed_model, [1.0, 2.0], [1.1, 1.9], [0.0], bounds=([0.0], [0.0])
        )
    assert np.isnan(intervals[0]["down"]) and np.isnan(intervals[0]["up"])


@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
def test_unrepresentable_residual_squares_fail_explicitly(magnitude: float) -> None:
    """Neither report infinite diagnostics nor mistake underflow for an exact fit."""
    with pytest.raises(FloatingPointError, match="float64|represent|range"):
        inf.audit_fit(through_origin, [1.0, 2.0], [magnitude, -magnitude], [0.0])


def test_bands_avoid_unnecessary_variance_overflow_and_underflow() -> None:
    """Propagate standard errors when their squares exceed the float64 range."""
    for coordinate, covariance in [(1e100, 1e300), (1e-100, 1e-300)]:
        lower, upper = inf.generate_error_bands(
            through_origin,
            [coordinate],
            [0.0],
            [[covariance]],
            reference="normal",
            jac=lambda x, a: x[:, None],
        )
        standard = coordinate * np.sqrt(covariance)
        assert_allclose(upper / standard, [stats.norm.isf(0.025)], rtol=1e-14)
        assert_allclose(lower / standard, [-stats.norm.isf(0.025)], rtol=1e-14)


def test_conditional_covariance_rank_is_independent_of_parameter_units() -> None:
    """Column scaling must preserve a full-rank free Jacobian."""
    x = np.linspace(-1, 1, 7)
    design = np.column_stack((1e20 * x, np.ones_like(x)))
    covariance = inf.robust_bounded_pcov(
        lambda x, a, b: 1e20 * a * x + b,
        x,
        np.ones(7),
        [0.0, 1.0],
        (-np.inf, np.inf),
        absolute_sigma=True,
        jac=lambda x, a, b: design,
    )
    assert_allclose(
        covariance, np.diag([1e-40 / (x @ x), 1 / 7]), atol=1e-55, rtol=1e-12
    )


def test_propagation_disparate_units_not_falsely_singular() -> None:
    """Preserve small-unit uncertainty and avoid a false rank warning."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = inf.propagate_uncertainty(
            lambda a, b: b,
            [0.0, 0.0],
            [[1e20, 0.0], [0.0, 1.0]],
            n_samples=20000,
            random_state=81,
        )
    assert not captured
    assert_allclose(result["std"], 1.0, rtol=0.02)


def test_propagation_preserves_global_rng_state() -> None:
    """Sampling must not mutate NumPy's legacy random state."""
    before = np.random.get_state()
    result = inf.propagate_uncertainty(
        lambda a: a, [1.0], [[1.0]], n_samples=1050, random_state=9
    )
    after = np.random.get_state()
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    repeated = inf.propagate_uncertainty(
        lambda a: a, [1.0], [[1.0]], n_samples=1050, random_state=9
    )
    assert result == repeated


@pytest.mark.parametrize("target", ["audit", "profile", "bounded"])
def test_absolute_sigma_requires_boolean(target: str) -> None:
    """Do not interpret the truthiness of strings as a noise model."""
    kwargs: dict[str, Any] = {"absolute_sigma": "false"}
    with pytest.raises(ValueError, match="boolean"):
        if target == "audit":
            inf.audit_fit(through_origin, [1.0, 2.0], [2.1, 3.9], [1.98], **kwargs)
        elif target == "profile":
            inf.profile_likelihood_errors(
                through_origin, [1.0, 2.0], [2.1, 3.9], [1.98], **kwargs
            )
        else:
            inf.robust_bounded_pcov(
                through_origin,
                [1.0, 2.0],
                [2.1, 3.9],
                [1.98],
                (-np.inf, np.inf),
                **kwargs,
            )


@pytest.mark.parametrize("seed", [-1, True, 1.5, np.random.default_rng(2)])
@pytest.mark.parametrize("target", ["bootstrap", "propagation"])
def test_monte_carlo_seed_validation(seed: Any, target: str) -> None:
    """Reject invalid seeds and shared generator objects."""
    with pytest.raises(ValueError, match="random_state"):
        if target == "bootstrap":
            inf.bootstrap_curve_fit(
                through_origin,
                [1.0, 2.0, 3.0],
                [2.1, 3.9, 6.1],
                [2.0],
                n_resamples=3,
                random_state=seed,
            )
        else:
            inf.propagate_uncertainty(
                lambda a: a, [1.0], [[1.0]], n_samples=3, random_state=seed
            )


def test_summary_nonfinite_arithmetic_is_not_significance() -> None:
    """Display undefined inference rather than an overflowing t-statistic."""
    x = np.array([1e-308, 2e-308, 3e-308])
    with pytest.warns(UserWarning) as caught:
        summary = inf.fit_summary(
            through_origin, x, [1.0, 2.0, 3.0], [1e308], [[1e-300]]
        )
    assert "inf" not in summary.lower() and "NaN" in summary
    assert any("infinite significance" in str(item.message) for item in caught)
    with pytest.warns(UserWarning, match="Singular parameter"):
        inf.fit_summary(
            linear,
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [1.0, 1.0],
            [[1.0, 1.0], [1.0, 1.0]],
        )


def test_unrepresentable_uncertainty_does_not_collapse_to_zero() -> None:
    """Require recentering when uncertainty is smaller than mean resolution."""
    with pytest.raises(FloatingPointError, match="resolution"):
        inf.propagate_uncertainty(lambda a: a - 1e30, [1e30], [[1.0]], n_samples=10)
    with pytest.raises(FloatingPointError, match="represented"):
        inf.generate_error_bands(
            through_origin, [1.0], [1e30], [[1.0]], reference="normal"
        )


def test_conditional_covariance_unrepresentable_variance() -> None:
    """Warn about numerical range rather than declaring an identified variance zero."""
    with pytest.warns(UserWarning, match="representable"):
        covariance = inf.robust_bounded_pcov(
            through_origin,
            [1e200, 2e200],
            [1.0, 2.0],
            [1e-200],
            (-np.inf, np.inf),
            absolute_sigma=True,
            jac=lambda x, a: x[:, None],
        )
    assert np.isnan(covariance).all()


def test_nonfinite_profile_trials_are_unresolved() -> None:
    """A model-domain failure must not masquerade as a confidence endpoint."""

    def domain_model(x: Array, a: float) -> Array:
        """Restrict a constant model to a narrow parameter domain."""
        return np.full_like(x, a) if abs(a - 1) <= 0.01 else np.full_like(x, np.nan)

    with pytest.warns(UserWarning) as captured:
        intervals = inf.profile_likelihood_errors(
            domain_model,
            [1.0, 2.0, 3.0],
            [0.99, 1.0, 1.01],
            [1.0],
            pcov=[[0.001]],
            absolute_sigma=True,
        )
    assert np.isnan(intervals[0]["up"]) and np.isnan(intervals[0]["down"])
    assert any("profile failed" in str(item.message) for item in captured)


def test_bootstrap_rejects_unreliable_base_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not bootstrap a base fit whose covariance is already undefined."""

    def invalid_base(*args: Any, **kwargs: Any) -> tuple[Array, Array]:
        """Return an explicitly undefined covariance."""
        return np.array([2.0]), np.array([[np.inf]])

    monkeypatch.setattr(optimize, "curve_fit", invalid_base)
    with pytest.raises(RuntimeError, match="Base fit covariance"):
        inf.bootstrap_curve_fit(
            through_origin, [1.0, 2.0, 3.0], [2.1, 3.9, 6.1], [2.0], n_resamples=3
        )


def test_smallest_alpha_fails_explicitly() -> None:
    """Reject a two-sided probability that cannot be represented."""
    with pytest.raises(ValueError, match="alpha/2"):
        inf.generate_error_bands(
            through_origin,
            [1.0],
            [1.0],
            [[1.0]],
            reference="normal",
            alpha=np.nextafter(0.0, 1.0),
        )


@pytest.mark.parametrize("target", ["summary", "diagnostics"])
def test_nonfinite_covariance_does_not_hide_negative_variances(target: str) -> None:
    """Reject negative variances even in partially non-finite matrices."""
    covariance = [[np.inf, 0.0], [0.0, -1.0]]
    with pytest.raises(ValueError, match="nonnegative"):
        if target == "summary":
            inf.fit_summary(
                linear, [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 1.0], covariance
            )
        else:
            inf.collinearity_diagnostics(covariance)


def test_unknown_rank_is_not_reported_as_zero() -> None:
    """Distinguish unavailable numerical rank from an actual zero covariance."""
    with pytest.warns(UserWarning, match="Non-finite"):
        result = inf.collinearity_diagnostics([[np.inf]])
    assert result["rank"] is None


def test_underflowing_profile_threshold_is_not_a_zero_interval() -> None:
    """Fail explicitly when a requested positive likelihood threshold rounds to zero."""
    with pytest.raises(ValueError, match="representable"):
        inf.profile_likelihood_errors(
            through_origin, [1.0, 2.0], [2.1, 3.9], [1.98], sigma_level=1e-200
        )
