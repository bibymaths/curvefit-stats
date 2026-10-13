"""Independent Deming and latent-coordinate references for ODR fitting."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from scipy import optimize

from curvefit_stats.estimators import eiv_curve_fit

# The compatibility warning is verified explicitly below. Other warnings remain
# errors under the suite's -W error policy, including backend numerical warnings.
pytestmark = pytest.mark.filterwarnings(
    "ignore:scipy.odr is deprecated in SciPy:FutureWarning"
)

_Array = NDArray[np.float64]


def _line(x: _Array, slope: float, intercept: float) -> _Array:
    """Evaluate a line in ordinary curve_fit argument order."""
    return slope * x + intercept


def _decay(x: _Array, amplitude: float, rate: float) -> _Array:
    """Evaluate exponential decay."""
    return amplitude * np.exp(-rate * x)


def _linear_observations() -> tuple[_Array, _Array, float, float]:
    """Generate substantial predictor error and modest response error."""
    generator = np.random.default_rng(811)
    truth_x = generator.normal(0, 1, 800)
    sx, sy = 0.8, 0.25
    x = truth_x + generator.normal(0, sx, truth_x.size)
    y = 2.5 * truth_x + 1.5 + generator.normal(0, sy, truth_x.size)
    return x, y, sx, sy


def test_linear_deming_reference_and_ols_attenuation() -> None:
    """Recover Deming's closed-form line rather than the attenuated OLS slope."""
    x, y, sx, sy = _linear_observations()
    xc, yc = x - x.mean(), y - y.mean()
    sxx, sxy, syy = xc @ xc, xc @ yc, yc @ yc
    ratio = sy**2 / sx**2
    reference_slope = (
        syy - ratio * sxx + np.sqrt((syy - ratio * sxx) ** 2 + 4 * ratio * sxy**2)
    ) / (2 * sxy)
    reference_intercept = y.mean() - reference_slope * x.mean()
    popt, covariance, diagnostics = eiv_curve_fit(
        _line,
        x,
        y,
        [2.0, 1.0],
        sx=sx,
        sy=sy,
        absolute_sigma=True,
        full_output=True,
        sstol=1e-12,
    )
    ols = np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]
    assert_allclose(popt, [reference_slope, reference_intercept], rtol=2e-6, atol=2e-6)
    assert abs(popt[0] - 2.5) < 0.15
    assert abs(popt[0] - 2.5) < 0.25 * abs(ols[0] - 2.5)
    assert diagnostics.success and diagnostics.identifiable
    assert diagnostics.method == "odr" and diagnostics.rank == 2
    assert diagnostics.dof == x.size - 2
    assert covariance.shape == (2, 2)
    assert np.isfinite(covariance).all()
    assert_allclose(diagnostics.adjusted_x, x + diagnostics.delta)
    assert_allclose(diagnostics.fitted_y, _line(diagnostics.adjusted_x, *popt))
    assert_allclose(diagnostics.eps, diagnostics.fitted_y - y, atol=1e-12)
    objective = np.sum((diagnostics.delta / sx) ** 2 + (diagnostics.eps / sy) ** 2)
    assert_allclose(diagnostics.weighted_sse, objective, rtol=1e-10)
    assert_allclose(diagnostics.residual_variance, objective / (x.size - 2), rtol=1e-10)
    # Schur complement of the full latent-x least-squares information matrix.
    design = np.column_stack((diagnostics.adjusted_x, np.ones_like(x)))
    effective_variance = sy**2 + popt[0] ** 2 * sx**2
    reference_covariance = np.linalg.solve(
        design.T @ design / effective_variance, np.eye(2)
    )
    assert_allclose(covariance, reference_covariance, rtol=3e-5)
    assert_allclose(diagnostics.standard_errors**2, np.diag(covariance))
    assert_allclose(
        diagnostics.backend_standard_errors**2,
        np.diag(covariance) * diagnostics.residual_variance,
        rtol=1e-10,
    )


def test_absolute_sigma_convention_matches_backend_residual_scale() -> None:
    """Distinguish known scales from a fitted common scale in both axes."""
    x, y, sx, sy = _linear_observations()
    absolute, known_cov, known = eiv_curve_fit(
        _line, x, y, [2.0, 1.0], sx=sx, sy=sy, absolute_sigma=True, full_output=True
    )
    relative, fitted_cov, fitted = eiv_curve_fit(
        _line, x, y, [2.0, 1.0], sx=sx, sy=sy, full_output=True
    )
    assert_array_equal(absolute, relative)
    assert_allclose(fitted_cov, known_cov * known.residual_variance, rtol=1e-12)
    assert_allclose(fitted.standard_errors, fitted.backend_standard_errors, rtol=1e-12)
    assert known.covariance_kind == "absolute" and fitted.covariance_kind == "scaled"
    # Scaling both supplied standard deviations changes the absolute covariance
    # but cancels against res_var when one common scale is being estimated.
    scaled, scaled_cov = eiv_curve_fit(_line, x, y, [2.0, 1.0], sx=3 * sx, sy=3 * sy)
    assert_allclose(scaled, relative, rtol=2e-6)
    assert_allclose(scaled_cov, fitted_cov, rtol=5e-5)


@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("weighted", [True, False])
def test_exact_x_matches_weighted_least_squares(absolute: bool, weighted: bool) -> None:
    """Omitting sx must fix predictor coordinates, not assign unit x error."""
    x = np.linspace(-2, 4, 60)
    sy = np.linspace(0.1, 0.3, x.size) if weighted else np.ones(x.size)
    y = _line(x, 1.7, -0.3) + np.random.default_rng(53).normal(size=x.size) * sy
    design = np.column_stack((x, np.ones_like(x)))
    information = (design / sy[:, None]).T @ (design / sy[:, None])
    expected = np.linalg.solve(information, (design / sy[:, None]).T @ (y / sy))
    reference_covariance = np.linalg.solve(information, np.eye(2))
    if not absolute:
        reference_covariance *= np.sum(((y - design @ expected) / sy) ** 2) / (
            x.size - 2
        )
    popt, covariance, diagnostics = eiv_curve_fit(
        _line,
        x,
        y,
        sy=sy if weighted else None,
        absolute_sigma=absolute,
        full_output=True,
    )
    assert_allclose(popt, expected, atol=2e-6, rtol=2e-6)
    assert_allclose(covariance, reference_covariance, rtol=2e-5)
    assert diagnostics.method == "ols"
    assert_array_equal(diagnostics.delta, np.zeros_like(x))
    assert diagnostics.sum_square_x == 0


def test_negligible_x_uncertainty_limit() -> None:
    """Approach ordinary nonlinear regression as predictor uncertainty vanishes."""
    x = np.linspace(0.1, 5, 70)
    sy = np.linspace(0.03, 0.06, x.size)
    y = _decay(x, 3.0, 0.7) + np.random.default_rng(213).normal(size=x.size) * sy
    reference, reference_covariance = optimize.curve_fit(
        _decay, x, y, p0=[2.8, 0.6], sigma=sy, absolute_sigma=True
    )
    popt, covariance = eiv_curve_fit(
        _decay, x, y, [2.8, 0.6], sx=1e-7, sy=sy, absolute_sigma=True, sstol=1e-12
    )
    assert_allclose(popt, reference, rtol=3e-6)
    assert_allclose(covariance, reference_covariance, rtol=1e-4)


def test_nonlinear_eiv_against_explicit_latent_coordinate_optimization() -> None:
    """Compare nonlinear ODR to the full independent augmented residual problem."""
    generator = np.random.default_rng(452)
    true_x = np.linspace(0.1, 5, 45)
    sx = np.linspace(0.06, 0.1, true_x.size)
    sy = np.linspace(0.03, 0.05, true_x.size)
    x = true_x + generator.normal(size=true_x.size) * sx
    y = _decay(true_x, 3.0, 0.65) + generator.normal(size=true_x.size) * sy

    def residuals(parameters: _Array) -> _Array:
        """Stack independent predictor and response residuals."""
        a, rate = parameters[:2]
        latent = parameters[2:]
        return np.asarray(
            np.r_[(latent - x) / sx, (_decay(latent, a, rate) - y) / sy], dtype=float
        )

    def jacobian(parameters: _Array) -> _Array:
        """Derive the full joint Jacobian analytically."""
        a, rate = parameters[:2]
        latent = parameters[2:]
        z = np.exp(-rate * latent)
        result = np.zeros((2 * x.size, x.size + 2))
        result[: x.size, 2:] = np.diag(1 / sx)
        result[x.size :, :2] = np.column_stack((z / sy, -a * latent * z / sy))
        result[x.size :, 2:] = np.diag(-a * rate * z / sy)
        return result

    reference = optimize.least_squares(
        residuals,
        np.r_[2.8, 0.6, x],
        jac=jacobian,
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    assert reference.success
    reference_jacobian = jacobian(np.asarray(reference.x, dtype=float))
    normal = reference_jacobian.T @ reference_jacobian
    reference_covariance = np.linalg.solve(normal, np.eye(normal.shape[0]))[:2, :2]
    popt, covariance, diagnostics = eiv_curve_fit(
        _decay,
        x,
        y,
        [2.8, 0.6],
        sx=sx,
        sy=sy,
        absolute_sigma=True,
        full_output=True,
        sstol=1e-12,
        partol=1e-12,
    )
    assert_allclose(popt, reference.x[:2], rtol=3e-6)
    assert_allclose(diagnostics.adjusted_x, reference.x[2:], atol=3e-6)
    assert_allclose(covariance, reference_covariance, rtol=1e-4)
    assert_allclose(popt, [3.0, 0.65], atol=0.08)
    assert_allclose(diagnostics.weighted_sse, 2 * reference.cost, rtol=1e-8)


def test_nonconvergence_raises_or_returns_explicit_diagnostics() -> None:
    """Exercise an actual ODR iteration limit without replacing the solver."""
    x = np.linspace(0.1, 5, 40)
    y = _decay(x, 3.0, 0.7) + np.random.default_rng(9).normal(0, 0.04, x.size)
    with pytest.raises(RuntimeError, match="did not converge"):
        eiv_curve_fit(_decay, x, y, [0.5, 2.0], sx=0.1, sy=0.04, maxit=1)
    with pytest.warns(UserWarning, match="did not converge"):
        popt, covariance, diagnostics = eiv_curve_fit(
            _decay, x, y, [0.5, 2.0], sx=0.1, sy=0.04, maxit=1, full_output=True
        )
    assert not diagnostics.success
    assert diagnostics.info % 10 == 4
    assert np.isfinite(popt).all() and np.isnan(covariance).all()
    assert diagnostics.covariance_kind == "undefined"


def test_unidentified_parameters_never_receive_finite_covariance() -> None:
    """Detect numerical derivative noise that can hide exact parameter dependence."""

    def model(x: _Array, a: float, b: float) -> _Array:
        """Evaluate non-identifiable sum parameters."""
        return (a + b) * x

    x = np.arange(1.0, 11.0)
    y = 3 * x + np.random.default_rng(14).normal(0, 0.1, x.size)
    with pytest.warns(UserWarning):
        _, covariance, diagnostics = eiv_curve_fit(
            model, x, y, [1.0, 1.0], sx=0.1, sy=0.1, full_output=True
        )
    assert np.isnan(covariance).all()
    assert not diagnostics.identifiable
    assert diagnostics.rank in (1, None)
    assert diagnostics.dof is None


def test_parameter_guess_shapes_and_input_preservation() -> None:
    """Infer ordinary signature counts but require explicit variadic guesses."""
    x = np.linspace(0.1, 4, 20)
    y = _line(x, 2.0, 1.0) + np.random.default_rng(12).normal(0, 0.1, x.size)
    sx, sy, p0 = np.full(x.size, 0.05), np.full(x.size, 0.1), np.array([1.0, 0.0])
    originals = [value.copy() for value in (x, y, sx, sy, p0)]
    first = eiv_curve_fit(_line, x, y, p0, sx=sx, sy=sy)

    def variadic(x: _Array, *beta: float) -> _Array:
        """Evaluate a model with an ambiguous parameter signature."""
        return beta[0] * x + beta[1]

    with pytest.raises(ValueError, match="p0"):
        eiv_curve_fit(variadic, x, y, sx=sx, sy=sy)
    second = eiv_curve_fit(variadic, x, y, p0, sx=sx, sy=sy)
    assert_allclose(first[0], second[0])
    for before, after in zip(originals, (x, y, sx, sy, p0), strict=True):
        assert_array_equal(before, after)
    assert first[0].shape == (2,) and first[1].shape == (2, 2)
    single, covariance = eiv_curve_fit(
        lambda x, a: a * x,
        x,
        2 * x + np.random.default_rng(2).normal(0, 0.1, x.size),
        [2.0],
        sx=sx,
        sy=sy,
    )
    assert single.shape == (1,) and covariance.shape == (1, 1)


@pytest.mark.parametrize("name", ["sx", "sy"])
@pytest.mark.parametrize(
    "uncertainty",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        [1.0],
        [1.0, 0.0, 1.0],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        1 + 1j,
    ],
)
def test_invalid_uncertainties(name: str, uncertainty: Any) -> None:
    """Reject nonpositive values, non-finite values, and covariance-shaped inputs."""
    kwargs: dict[str, Any] = {name: uncertainty}
    with pytest.raises(ValueError):
        eiv_curve_fit(_line, [1.0, 2.0, 3.0], [3.0, 5.0, 7.0], [1.0, 1.0], **kwargs)


@pytest.mark.parametrize(
    "options",
    [
        {"bounds": (0.0, 5.0)},
        {"method": "trf"},
        {"sigma": 0.1},
        {"jac": "2-point"},
        {"maxfev": 100},
        {"loss": "soft_l1"},
        {"covx": np.eye(3)},
        {"job": 0},
        {"maxit": -1},
        {"maxit": True},
        {"maxit": 1.5},
        {"sstol": 0},
        {"partol": np.inf},
        {"taufac": 2},
        {"ndigit": -1},
        {"absolute_sigma": "true"},
        {"full_output": "yes"},
    ],
)
def test_invalid_or_unsupported_options(options: dict[str, Any]) -> None:
    """Unsupported curve_fit controls must not be silently ignored."""
    with pytest.raises(ValueError):
        eiv_curve_fit(_line, [1.0, 2.0, 3.0], [3.1, 4.9, 7.0], [1.0, 1.0], **options)


@pytest.mark.parametrize(
    "x, y, p0",
    [
        ([], [], [1.0]),
        ([1.0, 2.0], [2.0], [1.0]),
        ([[1.0, 2.0, 3.0]], [2.0, 4.0, 6.0], [1.0]),
        ([1.0, 2.0, 3.0], [[2.0, 4.0, 6.0]], [1.0]),
        ([1.0, np.nan, 3.0], [2.0, 4.0, 6.0], [1.0]),
        ([1.0, 2.0, 3.0], [2.0, np.inf, 6.0], [1.0]),
        ([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [np.nan]),
        ([1.0, 2.0], [2.0, 4.0], [1.0, 1.0]),
    ],
)
def test_invalid_shapes_and_insufficient_degrees_of_freedom(
    x: Any, y: Any, p0: Any
) -> None:
    """Require aligned scalar observations and positive inference degrees of freedom."""
    with pytest.raises(ValueError):
        eiv_curve_fit(lambda x, *beta: beta[0] * x, x, y, p0)


@pytest.mark.parametrize("uncertainty", [1e200, 1e-200])
def test_unrepresentable_odr_weights(uncertainty: float) -> None:
    """Fail explicitly when ODR standard-deviation conversion loses numeric range."""
    with pytest.raises(FloatingPointError, match="weights"):
        eiv_curve_fit(
            _line, [1.0, 2.0, 3.0], [3.0, 5.0, 7.0], [1.0, 1.0], sx=uncertainty
        )


def test_backend_version_policy_and_deprecation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warn transparently on deprecated versions and reject removed backends."""
    x = np.arange(1.0, 6.0)
    y = 2 * x + np.array([0.1, -0.1, 0.05, -0.1, 0.1])
    version = tuple(int(part) for part in scipy.__version__.split(".")[:2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eiv_curve_fit(lambda x, a: a * x, x, y, [2.0], sy=0.1)
    compatibility = [
        item for item in caught if issubclass(item.category, FutureWarning)
    ]
    assert len(compatibility) == int(version >= (1, 17))
    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)
    monkeypatch.setattr(scipy, "__version__", "1.19.0")
    with pytest.raises(RuntimeError, match=">=1.12,<1.19"):
        eiv_curve_fit(lambda x, a: a * x, x, y, [2.0])


def test_model_errors_are_explicit() -> None:
    """Reject wrong parameter counts, non-finite models, and matrix responses."""
    with pytest.raises(ValueError, match="incompatible"):
        eiv_curve_fit(_line, [1.0, 2.0, 3.0], [3.0, 5.0, 7.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        eiv_curve_fit(
            lambda x, a: np.full_like(x, np.nan),
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [1.0],
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        eiv_curve_fit(
            lambda x, a: x[:, None] * a, [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0]
        )


def test_zero_iterations_and_zero_residual_variance() -> None:
    """Make diagnostic-only evaluation and degenerate scaled uncertainty explicit."""
    x = np.arange(1.0, 6.0)
    with pytest.warns(UserWarning, match="did not converge"):
        popt, covariance, diagnostics = eiv_curve_fit(
            lambda x, a: a * x, x, 2 * x, [1.0], maxit=0, full_output=True
        )
    assert_array_equal(popt, [1.0])
    assert not diagnostics.success and np.isnan(covariance).all()
    with pytest.warns(UserWarning, match="Zero residual variance"):
        _, covariance, diagnostics = eiv_curve_fit(
            lambda x, a: a * x, x, 2 * x, [2.0], full_output=True
        )
    assert_array_equal(covariance, [[0.0]])
    assert diagnostics.residual_variance == 0 and diagnostics.success
    _, known = eiv_curve_fit(lambda x, a: a * x, x, 2 * x, [2.0], absolute_sigma=True)
    assert_allclose(known, [[1 / (x @ x)]], rtol=1e-5)
