"""Regression cases from the independent pre-release statistical review."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from scipy import optimize

import curvefit_stats as cs

_Array = NDArray[np.float64]


def _line(x: _Array, a: float, b: float) -> _Array:
    """Evaluate a line with independent slope and intercept."""
    return a * x + b


def _line_jac(x: _Array, a: float, b: float) -> _Array:
    """Supply exact derivatives independently of numerical optimization."""
    return np.column_stack((x, np.ones_like(x)))


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e5, 1e50, 1e100])
@pytest.mark.parametrize("analytic", [False, True])
@pytest.mark.parametrize("known", [False, True])
def test_global_common_relative_weights_cannot_fake_convergence(
    scale: float,
    analytic: bool,
    known: bool,
) -> None:
    """Compare the actual optimum, covariance, Jacobian, and objective to WLS."""
    x = np.linspace(0, 2, 30)
    sigma = 1 + x / 2
    y = 2 * x + 1 + np.random.default_rng(91).normal(0, 0.1, x.size)
    design = _line_jac(x, 0, 0)
    weighted = design / sigma[:, None]
    beta = np.linalg.lstsq(weighted, y / sigma, rcond=None)[0]
    residual = (y - design @ beta) / sigma
    reference = (
        (residual @ residual) / (x.size - 2) * np.linalg.inv(weighted.T @ weighted)
    )
    group = cs.GlobalFitGroup(
        _line, {"d": {"x": x, "y": y, "sigma": scale * sigma}}, {"a", "b"}
    )
    fit = group.fit(
        [0.5, 0.0], jac=_line_jac if analytic else "3-point", absolute_sigma=known
    )
    assert fit.success and fit.diagnostics.identifiable
    assert_allclose(fit.popt, beta, rtol=2e-8)
    if known:
        assert_allclose(
            fit.pcov / scale / scale, np.linalg.inv(weighted.T @ weighted), rtol=2e-7
        )
    else:
        assert_allclose(fit.pcov, reference, rtol=2e-7)
    assert_allclose(fit.jacobian * scale, -weighted, rtol=3e-8)
    assert_allclose(
        fit.diagnostics.weighted_sse * scale * scale, residual @ residual, rtol=3e-12
    )
    assert_allclose(
        fit.solver.cost * scale * scale, (residual @ residual) / 2, rtol=3e-12
    )


@pytest.mark.parametrize("analytic", [False, True])
def test_bootstrap_rejects_trust_region_pseudoinverse(analytic: bool) -> None:
    """Finite SciPy pseudoinverse covariance cannot certify an identified model."""
    x = np.linspace(0, 1, 20)
    y = 3 * x + np.random.default_rng(1).normal(0, 0.1, x.size)

    def model(z: _Array, a: float, b: float) -> _Array:
        """Only the sum of the parameters is identifiable."""
        return (a + b) * z

    options: dict[str, Any] = {"method": "trf"}
    if analytic:
        options["jac"] = lambda z, a, b: np.column_stack((z, z))
    with pytest.raises(RuntimeError, match="identifiable"):
        cs.bootstrap_curve_fit(
            model, x, y, [1, 1], n_resamples=10, random_state=2, **options
        )


@pytest.mark.parametrize("kind", ["nonfinite", "singular"])
def test_bootstrap_checks_each_replicate_covariance(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Exclude an unreliable replicate even if SciPy emits no OptimizeWarning."""
    calls = 0
    real = optimize.curve_fit

    def fitted(*args: Any, **kwargs: Any) -> tuple[_Array, _Array]:
        """Corrupt one otherwise successful refit's covariance."""
        nonlocal calls
        calls += 1
        beta, covariance = real(*args, **kwargs)
        if calls == 2:
            covariance = (
                np.full((2, 2), np.nan) if kind == "nonfinite" else np.ones((2, 2))
            )
        return beta, covariance

    monkeypatch.setattr(optimize, "curve_fit", fitted)
    x = np.linspace(-2, 2, 30)
    y = _line(x, 2, 1) + np.random.default_rng(19).normal(0, 0.1, x.size)
    with pytest.warns(UserWarning, match="1 of 10"):
        _, covariance, samples = cs.bootstrap_curve_fit(
            _line, x, y, [2, 1], n_resamples=10, random_state=22
        )
    assert samples.shape == (9, 2)
    assert np.isfinite(covariance).all()


@pytest.mark.parametrize("seed", [1, 3])
def test_monte_carlo_large_offset_mean_and_quantiles(seed: int) -> None:
    """Use an independently translated reference for represented sample summaries."""
    offset = 1e16
    draws = offset + np.random.default_rng(seed).normal(1.5, 1, 2000)
    centered = draws - offset
    result = cs.propagate_uncertainty(
        lambda a: offset + a, [1.5], [[1]], n_samples=2000, random_state=seed
    )
    assert result["mean"] == offset + centered.mean()
    assert result["median"] == offset + np.median(centered)
    assert result["lower"] == offset + np.quantile(centered, 0.025)
    assert result["upper"] == offset + np.quantile(centered, 0.975)


def test_bootstrap_large_offset_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve the correctly rounded empirical mean of large fitted parameters."""
    values = 1e16 + np.random.default_rng(1).normal(1.5, 1, 100)
    calls = -1

    def fitted(*args: Any, **kwargs: Any) -> tuple[_Array, _Array]:
        """Return a regular model and prescribed parameter samples."""
        nonlocal calls
        calls += 1
        return np.array([1e16 if calls == 0 else values[calls - 1]]), np.ones((1, 1))

    monkeypatch.setattr(optimize, "curve_fit", fitted)

    def constant(x: _Array, a: float) -> _Array:
        """Evaluate a parameter-dependent constant response."""
        return np.full_like(x, a)

    mean, _, _ = cs.bootstrap_curve_fit(
        constant,
        np.arange(4.0),
        np.full(4, 1e16),
        [1e16],
        n_resamples=100,
        random_state=1,
    )
    assert mean[0] == 1e16 + (values - 1e16).mean()


@pytest.mark.parametrize("known", [False, True])
def test_profile_endpoint_accuracy_is_independent_of_covariance_seed(
    known: bool,
) -> None:
    """A large bracket must not allow a zero displacement to count as a root."""
    x = np.arange(1.0, 6.0)
    y = 2 * x + np.array([0.02, -0.02, 0.01, -0.01, 0.02])
    beta = np.array([x @ y / (x @ x)])

    def model(z: _Array, a: float) -> _Array:
        """Evaluate an identifiable one-column linear model."""
        return a * z

    error = cs.profile_likelihood_errors(
        model, x, y, beta, sigma=0.1, absolute_sigma=known, pcov=[[1e40]]
    )[0]
    rss = np.sum((y - model(x, *beta)) ** 2)
    width = (
        0.1 / np.sqrt(x @ x) if known else np.sqrt(rss * np.expm1(1 / x.size) / (x @ x))
    )
    assert_allclose([error["down"], error["up"]], [-width, width], rtol=3e-6)


@pytest.mark.parametrize("sigma_level", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("sigma", [None, [2.0, 1.0], [[4.0, 0.0], [0.0, 1.0]]])
def test_profile_optimum_check_ignores_known_scale_irreducible_sse(
    sigma_level: float, sigma: Any
) -> None:
    """Reject a poor baseline despite a huge parameter-independent residual."""
    # The independent WLS optimum is a=10; its known-scale deviance gain is
    # 100 (unit errors) or 25 (sigma[0]=2), far above tolerance*sigma_level**2.
    with pytest.raises(ValueError, match="not a local optimum"):
        cs.profile_likelihood_errors(
            lambda x, a: a * x,
            [1.0, 0.0],
            [10.0, 1e6],
            [0.0],
            sigma=sigma,
            sigma_level=sigma_level,
            absolute_sigma=True,
        )


def test_profile_optimum_check_uses_unknown_scale_likelihood_threshold() -> None:
    """Scale unknown-variance improvements by the chosen deviance threshold."""
    sigma_level = 1e-3
    threshold = sigma_level**2
    gain = 2 * np.log1p(100 / 1e12)
    assert gain > 1e-6 * threshold
    with pytest.raises(ValueError, match="not a local optimum"):
        cs.profile_likelihood_errors(
            lambda x, a: a * x,
            [1.0, 0.0],
            [10.0, 1e6],
            [0.0],
            sigma_level=sigma_level,
        )


@pytest.mark.parametrize("sigma_level", [0.5, 1.0, 2.0])
def test_profile_large_sse_true_optimum_has_analytical_endpoints(
    sigma_level: float,
) -> None:
    """Retain valid known-scale endpoints despite a large constant SSE term."""
    result = cs.profile_likelihood_errors(
        lambda x, a: a * x,
        [1.0, 0.0],
        [10.0, 1e4],
        [10.0],
        sigma=[2.0, 1.0],
        absolute_sigma=True,
        sigma_level=sigma_level,
    )[0]
    # Independently, delta deviance=(a-10)**2/4, so widths are 2*sigma_level.
    assert_allclose(
        [result["down"], result["up"]],
        [-2 * sigma_level, 2 * sigma_level],
        rtol=2e-6,
        atol=0,
    )


def test_profile_negligible_likelihood_gain_remains_acceptable() -> None:
    """Allow input error smaller than the documented likelihood-scale tolerance."""
    tolerance, sigma_level, standard_deviation = 1e-4, 1.0, 2.0
    offset = 0.25 * np.sqrt(tolerance) * sigma_level * standard_deviation
    result = cs.profile_likelihood_errors(
        lambda x, a: a * x,
        [1.0, 0.0],
        [10.0, 1.0],
        [10.0 - offset],
        sigma=[standard_deviation, 1.0],
        absolute_sigma=True,
        sigma_level=sigma_level,
        tolerance=tolerance,
    )[0]
    width = np.sqrt((sigma_level * standard_deviation) ** 2 + offset**2)
    assert_allclose(
        [result["down"], result["up"]],
        [offset - width, offset + width],
        rtol=2 * tolerance,
        atol=0,
    )


def test_profile_baseline_optimum_check_uses_likelihood_scale() -> None:
    """Reject a suboptimal fit despite large parameter-independent SSE."""

    def model(x: np.ndarray, a: float) -> np.ndarray:
        return a * x

    x = np.array([1.0, 0.0])
    y = np.array([10.0, 1e6])
    popt = np.array([0.0])

    # The baseline has a large parameter-independent residual.
    baseline_sse = np.sum((y - model(x, *popt)) ** 2)

    # The optimal parameter is a=10, improving SSE by 100.
    optimal_sse = np.sum((y - model(x, 10.0)) ** 2)

    assert baseline_sse - optimal_sse == pytest.approx(100.0)

    # The improvement exceeds the likelihood-ratio threshold,
    # despite being tiny relative to the total SSE.
    with pytest.raises(
        ValueError,
        match="popt is not a local optimum",
    ):
        cs.profile_likelihood_errors(
            model,
            x,
            y,
            popt,
            absolute_sigma=True,
        )


def test_positive_band_uncertainty_underflow_is_not_zero_width() -> None:
    """Reject a positive analytical width too small to represent in float64."""
    with pytest.raises(FloatingPointError, match="underflows"):
        cs.generate_error_bands(
            lambda x, a: a * x,
            [1e-200],
            [0.0],
            [[1e-300]],
            reference="normal",
            jac=lambda x, a: x[:, None],
        )


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("prediction", [False, True])
def test_singular_covariance_positive_support_underflow(
    correlated: bool, prediction: bool
) -> None:
    """Reject underflow at an uncertain point even with another zero-SE point."""
    covariance = np.array(
        [[1e-300, 1e-300], [1e-300, 1e-300]]
        if correlated
        else [[1e-300, 0.0], [0.0, 0.0]]
    )
    x = np.array([0.0, 1e-200])
    # Independent exact-decimal quadratic form: both gradients are x.
    variance = Decimal.from_float(x[1]) ** 2 * sum(
        Decimal.from_float(float(value)) for value in covariance.flat
    )
    assert variance > 0
    assert float(variance.sqrt()) == 0
    with pytest.raises(FloatingPointError, match="Positive mean uncertainty"):
        cs.generate_error_bands(
            lambda x, a, b: (a + b) * x,
            x,
            [0.0, 0.0],
            covariance,
            reference="normal",
            jac=lambda x, a, b: np.column_stack((x, x)),
            interval_type="prediction" if prediction else "confidence",
            prediction_variance=0.0 if prediction else None,
        )


def test_singular_underflow_with_large_fixed_parameter_gradient() -> None:
    """A large gradient in a fixed direction must not hide uncertain support."""
    with pytest.raises(FloatingPointError, match="Positive mean uncertainty"):
        cs.generate_error_bands(
            lambda x, a, b: a * x + b / x,
            [1e-200],
            [0.0, 0.0],
            [[1e-300, 0.0], [0.0, 0.0]],
            reference="normal",
            jac=lambda x, a, b: np.column_stack((x, 1 / x)),
        )


@pytest.mark.parametrize(
    "covariance, gradients",
    [
        ([[1e-300, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]),
        ([[4.0, 2.0], [2.0, 1.0]], [[1.0, -2.0], [0.0, 0.0]]),
        ([[0.0, 0.0], [0.0, 0.0]], [[1.0, 2.0], [0.0, 0.0]]),
    ],
)
def test_singular_nullspace_bands_remain_zero(
    covariance: list[list[float]], gradients: list[list[float]]
) -> None:
    """Preserve exact zero variance for fixed, correlated-null and zero inputs."""
    design = np.array(gradients)
    matrix = np.array(covariance)
    assert_allclose(np.einsum("ij,jk,ik->i", design, matrix, design), 0, atol=0)
    parameters = np.array([2.0, 3.0])
    lower, upper = cs.generate_error_bands(
        lambda x, a, b: a * x[0] + b * x[1],
        design.T,
        parameters,
        covariance,
        reference="normal",
        jac=lambda x, a, b: x.T,
    )
    assert_allclose(lower, design @ parameters, rtol=0, atol=0)
    assert_allclose(upper, lower, rtol=0, atol=0)


@pytest.mark.parametrize("correlated", [False, True])
def test_singular_representable_subnormal_bands(correlated: bool) -> None:
    """Keep positive subnormal widths, using an independent decimal reference."""
    covariance = np.array(
        [[1e-300, 1e-300], [1e-300, 1e-300]]
        if correlated
        else [[1e-300, 0.0], [0.0, 0.0]]
    )
    x = np.array([1e-160])
    variance = Decimal.from_float(x[0]) ** 2 * sum(
        Decimal.from_float(float(value)) for value in covariance.flat
    )
    from scipy.special import ndtri

    expected = float(ndtri(0.975)) * float(variance.sqrt())
    assert expected > 0
    lower, upper = cs.generate_error_bands(
        lambda x, a, b: (a + b) * x,
        x,
        [0.0, 0.0],
        covariance,
        reference="normal",
        jac=lambda x, a, b: np.column_stack((x, x)),
    )
    assert_allclose([lower[0], upper[0]], [-expected, expected], rtol=5e-13, atol=0)


def test_tiny_confidence_quantile_cannot_collapse_positive_width() -> None:
    """Reject underflow when multiplying a positive SE by a tiny critical value."""
    # sqrt(1e-300)*1e-173 is a positive subnormal SE; q near zero then underflows.
    with pytest.raises(FloatingPointError, match="Interval width"):
        cs.generate_error_bands(
            lambda x, a: a * x,
            [1e-173],
            [0.0],
            [[1e-300]],
            alpha=np.nextafter(1.0, 0.0),
            reference="normal",
            jac=lambda x, a: x[:, None],
        )


@pytest.mark.parametrize("fake_full_rank_covariance", [False, True])
def test_nonidentifiable_parameters_have_no_wald_significance(
    fake_full_rank_covariance: bool,
) -> None:
    """Report undefined inference for a rank-deficient model, not tiny p-values."""
    x = np.linspace(0, 1, 20)
    y = 3 * x + np.random.default_rng(1).normal(0, 0.1, x.size)

    def model(z: _Array, a: float, b: float) -> _Array:
        """Only a+b has an estimable data likelihood."""
        return (a + b) * z

    beta, covariance = optimize.curve_fit(
        model, x, y, [1, 1], method="trf", jac=lambda z, a, b: np.column_stack((z, z))
    )
    if fake_full_rank_covariance:
        covariance = np.eye(2)
    with pytest.warns(UserWarning, match="[Ww]ald"):
        table = cs.fit_summary(model, x, y, beta, covariance)
    for row in table.splitlines()[4:]:
        cells = row.split("|")[1:]
        assert all(cell.strip().lower() == "nan" for cell in cells[1:]), row


@pytest.mark.parametrize("scale", [1e-100, 1e-8, 1.0, 1e100])
@pytest.mark.parametrize("known", [False, True])
def test_bootstrap_trust_region_response_units(scale: float, known: bool) -> None:
    """Check every refit against independent linear residual resampling."""
    x = np.linspace(0, 2, 30)
    y = _line(x, 2, 1) + np.random.default_rng(91).normal(0, 0.1, x.size)
    design = _line_jac(x, 0, 0)
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    residual -= residual.mean()
    generator = np.random.default_rng(2)
    reference = np.stack(
        [
            np.linalg.lstsq(
                design, design @ beta + generator.choice(residual, x.size), rcond=None
            )[0]
            for _ in range(30)
        ]
    )

    def model(z: _Array, a: float, b: float) -> _Array:
        """Change only response units, not the physical parameter vector."""
        return scale * _line(z, a, b)

    def jac(z: _Array, a: float, b: float) -> _Array:
        """Supply symbolic derivatives in the changed units."""
        return scale * _line_jac(z, a, b)

    mean, covariance, samples = cs.bootstrap_curve_fit(
        model,
        x,
        scale * y,
        [0.5, 0],
        method="trf",
        jac=jac,
        absolute_sigma=known,
        n_resamples=30,
        random_state=2,
    )
    assert_allclose(samples, reference, rtol=3e-9, atol=3e-10)
    assert_allclose(mean, reference.mean(axis=0), rtol=3e-9)
    assert_allclose(covariance, np.cov(reference.T, ddof=1), rtol=3e-7)


def test_bootstrap_zero_observations_with_inferred_parameter() -> None:
    """Preserve signature-based guesses while fitting a tiny-response model."""
    x = np.arange(1.0, 5.0)

    def model(z: _Array, a: float) -> _Array:
        """Evaluate a regular one-parameter model in tiny response units."""
        return 1e-100 * a * z

    mean, covariance, _ = cs.bootstrap_curve_fit(
        model, x, np.zeros(4), method="trf", n_resamples=5, random_state=2
    )
    assert_allclose(mean, [0], atol=1e-12)
    assert covariance[0, 0] < 1e-25


@pytest.mark.parametrize("known", [False, True])
def test_audit_cannot_penalize_unidentified_redundant_parameters(known: bool) -> None:
    """Refuse regular criteria whose parameter count exceeds the model's dimension."""
    x = np.linspace(0, 1, 20)
    y = 3 * x + np.random.default_rng(1).normal(0, 0.1, x.size)

    def redundant(z: _Array, a: float, b: float) -> _Array:
        """Expose a one-dimensional family using two non-identifiable arguments."""
        return (a + b) * z

    with pytest.raises(ValueError, match="AIC/BIC"):
        cs.audit_fit(redundant, x, y, [1, 2], absolute_sigma=known)
    # The correctly reduced model retains the normalized Gaussian likelihood.
    estimate = float(x @ y / (x @ x))
    reduced = cs.audit_fit(
        lambda z, a: a * z,
        x,
        y,
        [estimate],
        sigma=0.1 if known else None,
        absolute_sigma=known,
    )
    rss = np.sum((y - estimate * x) ** 2)
    deviance = (
        x.size * np.log(2 * np.pi) + 2 * x.size * np.log(0.1) + rss / 0.01
        if known
        else x.size * (np.log(2 * np.pi) + 1 + np.log(rss / x.size))
    )
    k = 1 if known else 2
    assert_allclose(reduced["aic"], deviance + 2 * k, rtol=2e-12)
    assert_allclose(reduced["bic"], deviance + k * np.log(x.size), rtol=2e-12)


def test_bootstrap_preserves_robust_loss_in_physical_response_units() -> None:
    """Compare robust refits with an independent monotone score-equation solver."""
    x = np.linspace(0, 4, 35)
    column = np.exp(-0.7 * x)
    y = 2 * column + np.random.default_rng(77).normal(0, 0.02, x.size)
    y[3] += 0.8
    y[20] -= 0.4
    threshold = 0.05

    def fitted_amplitude(response: _Array) -> float:
        """Solve the soft-L1 estimating equation in the original measurement units."""

        def score(amplitude: float) -> float:
            """Differentiate the original robust objective independently."""
            residual = amplitude * column - response
            return float(column @ (residual / np.sqrt(1 + (residual / threshold) ** 2)))

        return float(optimize.brentq(score, 0.1, 4, xtol=1e-13))

    beta = fitted_amplitude(y)
    residual = y - beta * column
    residual -= residual.mean()
    generator = np.random.default_rng(52)
    reference = np.array(
        [
            fitted_amplitude(beta * column + generator.choice(residual, x.size))
            for _ in range(20)
        ]
    )
    _, _, samples = cs.bootstrap_curve_fit(
        lambda z, a: a * np.exp(-0.7 * z),
        x,
        y,
        [1.5],
        method="trf",
        loss="soft_l1",
        f_scale=threshold,
        n_resamples=20,
        random_state=52,
    )
    assert_allclose(samples[:, 0], reference, rtol=2e-6)


def test_covariance_informs_finite_difference_scale_at_zero_mean() -> None:
    """Resolve a representable band even when default absolute probe changes vanish."""

    def model(x: _Array, a: float) -> _Array:
        """Evaluate a linear function on a tiny independent-variable grid."""
        return a * x

    lower, upper = cs.generate_error_bands(
        model, [1e-320], [0], [[1e300]], reference="normal"
    )
    # Independent analytic derivative is x, so SE is x*sqrt(parameter variance).
    from scipy.special import ndtri

    expected = float(ndtri(0.975)) * 1e-320 * 1e150
    assert_allclose([lower[0], upper[0]], [-expected, expected], rtol=2e-5, atol=0)
