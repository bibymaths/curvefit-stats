"""Local and resampling inference for nonlinear least-squares models.

Only NumPy and SciPy are required. Model calls use ``func(x, *parameters)``.
Observations and model outputs are scalar or one-dimensional; coordinates may
be multidimensional with observations on the last axis. All computations use
float64. None of these methods establishes structural identifiability or exact
coverage for arbitrary nonlinear models.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from fractions import Fraction
from functools import wraps
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize, stats

__all__ = [
    "generate_error_bands",
    "audit_fit",
    "fit_summary",
    "profile_likelihood_errors",
    "bootstrap_curve_fit",
    "collinearity_diagnostics",
    "robust_bounded_pcov",
    "propagate_uncertainty",
]

_Array = NDArray[np.float64]
_Model = Callable[..., ArrayLike]
_Jacobian = Callable[..., ArrayLike]


def _warn(message: str) -> None:
    """Warn at the calling public function's caller."""
    warnings.warn(message, UserWarning, stacklevel=3)


def _real(value: ArrayLike, name: str) -> _Array:
    """Convert real numerical values without silently discarding imaginary parts."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real-valued.")
    return np.asarray(value, dtype=np.float64)


def _array(value: ArrayLike, name: str) -> _Array:
    """Convert without mutating the input and require finite values."""
    result = _real(value, name)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _computed(value: ArrayLike, name: str) -> _Array:
    """Reject non-finite numerical results rather than inventing inference."""
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError(
            f"{name} is outside the representable float64 range; rescale inputs."
        )
    return result


def _norms(values: _Array, axis: int) -> _Array:
    """Calculate stable Euclidean norms without squaring physical magnitudes."""
    scale = np.max(np.abs(values), axis=axis, keepdims=True)
    safe = np.where(scale == 0, 1.0, scale)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = np.sqrt(np.sum((values / safe) ** 2, axis=axis)) * np.squeeze(
            scale, axis=axis
        )
    return _computed(result, "Euclidean norm")


def _squared_norm(values: _Array) -> float:
    """Reject overflow and underflow of a nonzero residual sum of squares."""
    norm = float(_norms(values, axis=0))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        squared = float(np.multiply(norm, norm))
    if not np.isfinite(squared) or (norm > 0 and squared == 0):
        raise FloatingPointError(
            "Residual sum of squares is outside the representable float64 range; "
            "rescale the observations or noise."
        )
    return squared


def _boolean(value: bool, name: str) -> bool:
    """Require an actual boolean for a statistical interpretation flag."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean.")
    return bool(value)


def _generator(random_state: int | None) -> np.random.Generator:
    """Require a nonnegative integer seed without accepting shared RNG objects."""
    if random_state is not None:
        random_state = _count(random_state, "random_state", 0)
    return np.random.default_rng(random_state)


def _vector(value: ArrayLike, name: str) -> _Array:
    """Normalize a scalar or nonempty vector."""
    result = _array(value, name)
    if result.ndim == 0:
        result = result.reshape(1)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    return result


def _parameters(popt: ArrayLike) -> _Array:
    """Validate the parameter vector."""
    return _vector(popt, "popt")


def _alpha(alpha: float) -> None:
    """Validate the two-sided significance level."""
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must satisfy 0 < alpha < 1.")
    if alpha / 2 == 0:
        raise ValueError("alpha/2 must be representable in float64.")


def _count(value: int, name: str, minimum: int = 1) -> int:
    """Reject booleans and non-integer counts."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _coordinates(x: ArrayLike, n: int | None = None) -> _Array:
    """Validate coordinates, with observations on their last axis."""
    result = _array(x, "x")
    size = 1 if result.ndim == 0 else result.shape[-1]
    if size == 0 or (n is not None and size != n):
        raise ValueError("x must have one coordinate per observation on its last axis.")
    return result


def _evaluate(func: _Model, x: _Array, beta: _Array, n: int | None = None) -> _Array:
    """Evaluate a finite scalar/vector model with a stable shape."""
    result = _vector(func(x, *beta), "model output")
    expected = (1 if x.ndim == 0 else x.shape[-1]) if n is None else n
    if result.size != expected:
        raise ValueError("Model output must have one value per coordinate.")
    return result


def _data(
    func: _Model, x: ArrayLike, y: ArrayLike, beta: _Array
) -> tuple[_Array, _Array, _Array]:
    """Validate fitting inputs and evaluate residuals."""
    observations = _vector(y, "y")
    coordinates = _coordinates(x, observations.size)
    with np.errstate(over="ignore", invalid="ignore"):
        residual = _computed(
            observations - _evaluate(func, coordinates, beta, observations.size),
            "Residuals",
        )
    return coordinates, observations, residual


def _covariance(pcov: ArrayLike, p: int) -> tuple[_Array, _Array, _Array]:
    """Validate in correlation units and return a physically scaled eigenbasis.

    The second result contains correlation eigenvalues; the third is the
    correlation eigenbasis multiplied by parameter standard deviations.
    Multiplying its columns by sqrt(eigenvalues) gives a covariance factor.
    This avoids rank and PSD decisions that depend on arbitrary parameter units.
    """
    covariance = _array(pcov, "pcov")
    if covariance.shape != (p, p):
        raise ValueError(f"pcov must have shape ({p}, {p}).")
    diagonal = np.diag(covariance)
    if np.any(diagonal < 0):
        raise ValueError("pcov diagonal variances must be nonnegative.")
    zero = diagonal == 0
    if np.any(covariance[zero] != 0) or np.any(covariance[:, zero] != 0):
        raise ValueError("Zero covariance variances require zero rows and columns.")
    standard = np.sqrt(diagonal)
    safe = np.where(zero, 1.0, standard)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = covariance / safe[:, None] / safe[None, :]
    if not np.isfinite(normalized).all():
        raise ValueError("pcov must be positive semidefinite in correlation units.")
    tolerance = 100 * np.finfo(float).eps * max(p, 1)
    if np.max(np.abs(normalized - normalized.T)) > tolerance:
        raise ValueError("pcov must be symmetric.")
    symmetric = normalized * 0.5 + normalized.T * 0.5
    eigenvalues, eigenvectors = linalg.eigh(symmetric)
    if eigenvalues[0] < -tolerance:
        raise ValueError("pcov must be positive semidefinite.")
    physical_vectors = standard[:, None] * eigenvectors
    if eigenvalues[0] < 0:
        _warn(
            "Tiny negative correlation eigenvalues were set to zero "
            "within roundoff tolerance."
        )
        eigenvalues = np.maximum(eigenvalues, 0)
        factor = physical_vectors * np.sqrt(eigenvalues)
        with np.errstate(over="ignore", invalid="ignore"):
            covariance = _computed(factor @ factor.T, "Repaired covariance")
    else:
        covariance = covariance * 0.5 + covariance.T * 0.5
    return covariance, eigenvalues, physical_vectors


def _jacobian(
    func: _Model, x: _Array, beta: _Array, jac: _Jacobian | None = None
) -> _Array:
    """Use an analytic Jacobian or scale-aware central differences."""
    n = _evaluate(func, x, beta).size
    if jac is not None:
        result = _array(jac(x, *beta), "jac")
        if n == 1 and result.shape == (beta.size,):
            result = result.reshape(1, beta.size)
        if result.shape != (n, beta.size):
            raise ValueError(f"jac must have shape ({n}, {beta.size}).")
        return result
    return np.column_stack(
        [_central_column(func, x, beta, index) for index in range(beta.size)]
    )


def _central_column(func: _Model, x: _Array, beta: _Array, index: int) -> _Array:
    """Refine central differences for disparate scales and restricted domains."""
    step = float(np.cbrt(np.finfo(float).eps) * max(1.0, abs(beta[index])))
    previous: _Array | None = None
    for _ in range(40):
        plus, minus = beta.copy(), beta.copy()
        plus[index] += step
        minus[index] -= step
        if plus[index] == minus[index]:
            break
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                derivative = (_evaluate(func, x, plus) - _evaluate(func, x, minus)) / (
                    plus[index] - minus[index]
                )
        except ValueError:
            # The base model was validated. Perturbations may leave its domain.
            previous = None
            step *= 0.5
            continue
        if not np.isfinite(derivative).all():
            previous = None
        elif previous is not None:
            scale = max(
                float(np.max(np.abs(derivative))),
                float(np.max(np.abs(previous))),
                np.finfo(float).tiny,
            )
            difference = np.linalg.norm(derivative / scale - previous / scale)
            norm = float(np.linalg.norm(derivative / scale))
            if difference <= 1e-6 * max(norm, np.finfo(float).tiny):
                return np.asarray(derivative, dtype=np.float64)
        if np.isfinite(derivative).all():
            previous = derivative
        step *= 0.5
    raise ValueError(
        "Numerical Jacobian did not stabilize; supply an analytical jac "
        "or rescale the parameters."
    )


def _noise(sigma: ArrayLike | None, n: int) -> tuple[_Array | None, float]:
    """Return standard deviations or a Cholesky factor, and log determinant."""
    if sigma is None:
        return None, 0.0
    supplied = _array(sigma, "sigma")
    if supplied.ndim == 0:
        supplied = np.full(n, float(supplied))
    if supplied.ndim == 1:
        if supplied.shape != (n,) or np.any(supplied <= 0):
            raise ValueError(
                "sigma must contain one positive standard deviation per observation."
            )
        return supplied, float(2 * np.log(supplied).sum())
    if supplied.shape != (n, n):
        raise ValueError(f"sigma covariance must have shape ({n}, {n}).")
    if not np.allclose(supplied, supplied.T, rtol=1e-12, atol=0):
        raise ValueError("sigma covariance must be symmetric.")
    try:
        factor = linalg.cholesky(supplied, lower=True)
    except linalg.LinAlgError as error:
        raise ValueError("sigma covariance must be positive definite.") from error
    return factor, float(2 * np.log(np.diag(factor)).sum())


def _whiten(value: _Array, noise: _Array | None) -> _Array:
    """Whiten finite residuals or Jacobians, with explicit range failures."""
    if noise is None:
        return _computed(value, "Residuals or Jacobian")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if noise.ndim == 1:
            result = value / (noise if value.ndim == 1 else noise[:, None])
        else:
            result = linalg.solve_triangular(noise, value, lower=True)
    return _computed(result, "Whitened residuals or Jacobian")


def _bounds(bounds: tuple[Any, Any], beta: _Array) -> tuple[_Array, _Array]:
    """Normalize parameter bounds, allowing exactly fixed parameters."""
    if len(bounds) != 2:
        raise ValueError("bounds must be a (lower, upper) pair.")
    try:
        lower = np.broadcast_to(_real(bounds[0], "lower bounds"), beta.shape).copy()
        upper = np.broadcast_to(_real(bounds[1], "upper bounds"), beta.shape).copy()
    except ValueError as error:
        raise ValueError(
            "Bounds must be scalar or match the parameter vector."
        ) from error
    if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
        raise ValueError("Bounds must be ordered and must not contain NaN.")
    if np.any(beta < lower) or np.any(beta > upper):
        raise ValueError("popt must lie within bounds.")
    return lower, upper


def _local_covariance(
    jacobian: _Array, residual: _Array, absolute_sigma: bool
) -> _Array:
    """Estimate covariance by column-scaled SVD without false rank conclusions."""
    n, p = jacobian.shape
    column_scale = _norms(jacobian, axis=0)
    safe = np.where(column_scale == 0, 1.0, column_scale)
    _, singular, right = linalg.svd(jacobian / safe, full_matrices=False)
    tolerance = np.finfo(float).eps * max(n, p) * (singular[0] if singular.size else 0)
    if singular.size < p or np.count_nonzero(singular > tolerance) < p:
        _warn("Free-parameter Jacobian is rank deficient; covariance is undefined.")
        return np.full((p, p), np.nan)
    if not absolute_sigma and n <= p:
        _warn(
            "Nonpositive residual degrees of freedom; covariance scaling is undefined."
        )
        return np.full((p, p), np.nan)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        factor = (right.T / singular) / safe[:, None]
        if not absolute_sigma:
            factor *= float(_norms(residual, axis=0)) / np.sqrt(n - p)
        covariance = np.asarray(factor @ factor.T, dtype=float)
    lost_variance = (np.diag(covariance) == 0) & np.any(factor != 0, axis=1)
    if not np.isfinite(covariance).all() or np.any(lost_variance):
        _warn(
            "Conditional covariance is outside the representable float64 range; "
            "returning NaN. Rescale parameters."
        )
        return np.full((p, p), np.nan)
    return covariance


def generate_error_bands(
    func: _Model,
    x: ArrayLike,
    popt: ArrayLike,
    pcov: ArrayLike,
    y: ArrayLike | None = None,
    alpha: float = 0.05,
    interval_type: Literal["confidence", "prediction"] = "confidence",
    *,
    x_fit: ArrayLike | None = None,
    dof: int | None = None,
    jac: _Jacobian | None = None,
    prediction_variance: float | ArrayLike | None = None,
    reference: Literal["student-t", "normal"] = "student-t",
) -> tuple[_Array, _Array]:
    """Return pointwise Delta Method confidence or prediction bands.

    Parameters
    ----------
    func : callable
        Model evaluated as ``func(x, *popt)``.
    x : array_like
        Evaluation coordinates; observations occupy the last axis.
    popt : array_like
        Fitted parameter vector of length p.
    pcov : array_like
        Finite positive-semidefinite parameter covariance, shape (p, p).
    y : array_like, optional
        Original fitting observations. Used to infer degrees of freedom and,
        for prediction intervals, unknown common independent noise variance.
    alpha : float, default 0.05
        Two-sided significance level.
    interval_type : {'confidence', 'prediction'}
        Confidence for fitted means or prediction for future scalar observations.
    x_fit : array_like, optional
        Fitting coordinates aligned with y; defaults to x when y is supplied.
    dof : int, optional
        Positive fitting residual degrees of freedom for Student-t intervals.
        Required if y is absent. Never inferred from an evaluation-only grid.
    jac : callable, optional
        Model parameter Jacobian of shape (n_evaluation, p). Without jac,
        numerical differentiation scales parameters by max(abs(popt), parameter
        standard deviation), using one for zero scales. Central differences
        are refined until successive columns agree within
        1e-6 relative norm, with at most 40 refinements. Supply jac for
        nonsmooth models or derivatives that do not stabilize.
    prediction_variance : float or array_like, optional
        Nonnegative future-observation variances in evaluation order. Required
        for known, heteroscedastic, or correlated noise; includes only marginal
        noise and assumes future noise is independent of parameter estimation.
    reference : {'student-t', 'normal'}
        Use 'normal' explicitly for known absolute Gaussian error scales. That
        option does not require dof and never estimates noise from residuals.

    Returns
    -------
    lower_band, upper_band : ndarray
        One-dimensional arrays, including for a single evaluation point.

    Raises
    ------
    ValueError
        Invalid data/covariance, interval options, prediction variance, or dof.
    FloatingPointError
        Non-finite numerical calculations or unrepresentable band endpoints.

    Notes
    -----
    At each point, mean variance is j @ pcov @ j.T. Endpoints are
    f(x, popt) +/- q * sqrt(mean_variance + future_variance), where the
    future term is zero for confidence intervals. q is the specified normal
    or Student-t upper-tail quantile at alpha/2.

    These are pointwise, not simultaneous, intervals. Residual-based prediction
    assumes independent homoscedastic Gaussian errors with unknown variance.
    The API cannot infer a noise model from pcov; choosing it is the caller's
    responsibility. Linearization may be unreliable near bounds or in strongly
    nonlinear or non-identifiable models. Unrepresentable band endpoints raise
    FloatingPointError instead of returning infinite or collapsed bands.
    Singular covariance permits zero mean uncertainty in its nullspace. A
    nonzero projection onto its positive-variance support that underflows to
    zero instead raises FloatingPointError.

    Examples
    --------
    >>> import numpy as np
    >>> lo, hi = generate_error_bands(lambda x, a: a*x, [1., 2.],
    ...     [2.], [[0.01]], dof=8)
    >>> bool(np.all(lo < hi))
    True
    """
    _alpha(alpha)
    if interval_type not in ("confidence", "prediction"):
        raise ValueError("interval_type must be 'confidence' or 'prediction'.")
    if reference not in ("student-t", "normal"):
        raise ValueError("reference must be 'student-t' or 'normal'.")
    beta = _parameters(popt)
    _, eigenvalues, eigenvectors = _covariance(pcov, beta.size)
    coordinates = _coordinates(x)
    fitted = _evaluate(func, coordinates, beta)
    residual: _Array | None = None
    fitting_dof: int | None = None
    if y is not None:
        _, observations, residual = _data(func, x if x_fit is None else x_fit, y, beta)
        fitting_dof = observations.size - beta.size
    elif x_fit is not None:
        raise ValueError("x_fit requires aligned fitting observations y.")
    if dof is not None:
        dof = _count(dof, "dof")
        if fitting_dof is not None and dof != fitting_dof:
            raise ValueError(
                "dof must equal the number of fitting observations minus parameters."
            )
    else:
        dof = fitting_dof
    if reference == "student-t":
        if dof is None or dof <= 0:
            raise ValueError(
                "Student-t intervals require positive fitting degrees of freedom."
            )
        critical = float(stats.t.isf(alpha / 2, dof))
    else:
        critical = float(stats.norm.isf(alpha / 2))
    if jac is None:
        # The covariance supplies meaningful parameter units at a zero mean.
        # Fixed absolute steps can otherwise see only underflowed model changes.
        parameter_scale = np.maximum(
            np.abs(beta), np.sqrt(np.diag(_real(pcov, "pcov")))
        )
        parameter_scale = np.where(parameter_scale == 0, 1.0, parameter_scale)

        def scaled_model(x: _Array, *offsets: float) -> ArrayLike:
            """Differentiate in parameter coordinates informed by covariance."""
            return func(x, *(beta + parameter_scale * np.asarray(offsets)))

        derivative = _computed(
            _jacobian(scaled_model, coordinates, np.zeros_like(beta)) / parameter_scale,
            "Rescaled model Jacobian",
        )
    else:
        derivative = _jacobian(func, coordinates, beta, jac)
    factor = eigenvectors * np.sqrt(eigenvalues)
    with np.errstate(over="ignore", invalid="ignore"):
        projected = _computed(derivative @ factor, "Projected covariance factor")
    standard_error = _norms(projected, axis=1)
    if interval_type == "prediction":
        if prediction_variance is not None:
            noise_variance = _array(prediction_variance, "prediction_variance")
            if noise_variance.ndim > 1 or noise_variance.shape not in (
                (),
                fitted.shape,
            ):
                raise ValueError(
                    "prediction_variance must be scalar or match the evaluation output."
                )
            if np.any(noise_variance < 0):
                raise ValueError("prediction_variance must be nonnegative.")
            standard_error = np.hypot(standard_error, np.sqrt(noise_variance))
        elif reference == "normal":
            raise ValueError(
                "Normal-reference prediction requires explicit prediction_variance."
            )
        elif residual is None or fitting_dof is None or fitting_dof <= 0:
            raise ValueError(
                "Residual prediction requires aligned y, "
                "fitting coordinates, and N > p."
            )
        else:
            standard_error = np.hypot(
                standard_error, float(_norms(residual, axis=0)) / np.sqrt(fitting_dof)
            )
    elif prediction_variance is not None:
        raise ValueError("prediction_variance is only valid for prediction intervals.")
    # A singular covariance can still carry positive uncertainty at a point.
    # Check zero-SE gradients against its physical positive-eigenvalue support.
    # Exact float products avoid underflow in this diagnostic too, while exact
    # cancellation preserves legitimate zero uncertainty in the nullspace.
    if any(
        sum(
            (
                Fraction(float(component)) * Fraction(float(coordinate))
                for component, coordinate in zip(gradient, direction, strict=True)
                if component != 0 and coordinate != 0
            ),
            Fraction(0),
        )
        != 0
        for gradient in derivative[standard_error == 0]
        for direction in eigenvectors[:, eigenvalues > 0].T
    ):
        raise FloatingPointError(
            "Positive mean uncertainty underflows float64; rescale the output."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        width = _computed(critical * standard_error, "Interval width")
        lower_band = _computed(fitted - width, "Lower band")
        upper_band = _computed(fitted + width, "Upper band")
    if np.any((standard_error > 0) & (width == 0)) or np.any(
        (width > 0) & (lower_band == upper_band)
    ):
        raise FloatingPointError(
            "Interval width cannot be represented around the fitted mean; "
            "rescale the output."
        )
    return lower_band, upper_band


def audit_fit(
    func: _Model,
    x: ArrayLike,
    y: ArrayLike,
    popt: ArrayLike,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
) -> dict[str, float]:
    """Calculate Gaussian goodness-of-fit and likelihood-consistent AIC/BIC.

    Parameters
    ----------
    func : callable
        Model ``func(x, *popt)``.
    x, y : array_like
        Fitting coordinates and aligned scalar observations.
    popt : array_like
        Estimated parameters.
    sigma : array_like, optional
        Positive standard deviations (scalar/vector) or full positive-definite
        observation covariance, with the same interpretation as curve_fit.
    absolute_sigma : bool
        True treats sigma as known; absent sigma then means known unit variance.
        False estimates one common scale multiplying the supplied covariance.

    Returns
    -------
    dict
        Counts, dof, rss, rmse, residual_variance, chi_square,
        reduced_chi_square, aic, and bic. Counts are returned as floats.

    Raises
    ------
    ValueError
        Invalid observations, parameters, sigma, absolute_sigma, or a numerically
        rank-deficient parameter Jacobian incompatible with regular criteria.
    FloatingPointError
        Residual sums or likelihood quantities are outside float64 range.

    Notes
    -----
    For observation covariance C, Q = r.T @ inv(C) @ r. Known-scale
    deviance is N*log(2*pi) + logdet(C) + Q, with K=p. Unknown-scale
    deviance is N*(log(2*pi)+1+log(Q/N)) + logdet(C), with K=p+1.
    AIC = deviance+2*K; BIC = deviance+K*log(N).

    AIC/BIC assume maximum-likelihood input estimates and count every supplied
    parameter. Fixed/tied parameters require a reduced model/vector; singular
    and boundary models need separate likelihood/parameter-count calibration.
    Arbitrary estimates merely evaluate a likelihood, not valid model criteria.
    A column-normalized, whitened numerical Jacobian must have full rank at
    relative cutoff max(1e-7, max(N,p)*eps); otherwise the audit rejects the
    input rather than reporting regular criteria with an invalid parameter count.
    Rank checks do not verify maximum likelihood or boundary regularity.

    AIC/BIC retain the Gaussian normalization constant for consistent likelihood
    comparisons. Unknown scale counts as an additional fitted parameter. The
    unweighted residual variance is RSS/(N-p), even for weighted fits; it is not
    an estimate of a heteroscedastic covariance. Without known or supplied
    weights, chi-square is undefined (NaN). Nonpositive dof makes residual
    variance and reduced chi-square NaN. Zero weighted RSS with unknown scale
    has an unbounded likelihood and returns NaN AIC/BIC with a warning.
    Nonzero squared residuals outside float64 range raise FloatingPointError
    rather than falsely reporting an exact fit or infinite likelihood metrics.

    Examples
    --------
    >>> result = audit_fit(lambda x, a: a*x, [1., 2., 3.],
    ...                    [2.1, 3.9, 6.0], [2.])
    >>> round(result['rss'], 2)
    0.02
    """
    absolute_sigma = _boolean(absolute_sigma, "absolute_sigma")
    beta = _parameters(popt)
    coordinates, observations, residual = _data(func, x, y, beta)
    n, p = observations.size, beta.size
    dof = n - p
    noise, logdet = _noise(sigma, n)
    if sigma is not None and not absolute_sigma:
        _warn(
            "sigma is interpreted as relative weights with absolute_sigma=False; "
            "SciPy rescales parameter covariance by the fitted residual scale. "
            "This is a valid relative-weighting convention."
        )
    weighted = _whiten(residual, noise)
    rss, wss = _squared_norm(residual), _squared_norm(weighted)
    derivative = _whiten(_jacobian(func, coordinates, beta), noise)
    scales = _norms(derivative, axis=0)
    singular = linalg.svdvals(derivative / np.where(scales == 0, 1.0, scales))
    cutoff = max(1e-7, max(derivative.shape) * np.finfo(float).eps)
    if singular.size < p or np.count_nonzero(singular > cutoff * singular[0]) < p:
        raise ValueError(
            "Regular AIC/BIC and residual degrees of freedom require a full-rank "
            "parameter Jacobian; reduce/reparameterize the model."
        )
    chi_square = wss if sigma is not None or absolute_sigma else float("nan")
    if dof <= 0:
        _warn("N <= p: residual variance and reduced chi-square are undefined.")
    residual_variance = rss / dof if dof > 0 else float("nan")
    reduced = chi_square / dof if dof > 0 else float("nan")
    if absolute_sigma:
        deviance = n * np.log(2 * np.pi) + logdet + wss
        k = p
    elif wss > 0:
        deviance = n * (np.log(2 * np.pi) + 1 + (np.log(wss) - np.log(n))) + logdet
        k = p + 1
    else:
        _warn(
            "Zero residual sum of squares gives an unbounded unknown-scale likelihood; "
            "AIC and BIC are undefined."
        )
        deviance, k = float("nan"), p + 1
    if wss > 0 or absolute_sigma:
        _computed([deviance + 2 * k, deviance + k * np.log(n)], "AIC/BIC")
    return {
        "n_observations": float(n),
        "n_parameters": float(p),
        "dof": float(dof),
        "rss": rss,
        "rmse": float(float(_norms(residual, axis=0)) / np.sqrt(n)),
        "residual_variance": residual_variance,
        "chi_square": chi_square,
        "reduced_chi_square": reduced,
        "aic": float(deviance + 2 * k),
        "bic": float(deviance + k * np.log(n)),
    }


def _names(func: _Model, p: int, supplied: list[str] | None) -> list[str]:
    """Infer only positional model parameter names."""
    if supplied is not None:
        if len(supplied) != p or any(
            not isinstance(name, str) or not name for name in supplied
        ):
            raise ValueError(
                "param_names must contain one nonempty string per parameter."
            )
        return list(supplied)
    try:
        parameters = list(inspect.signature(func).parameters.values())
    except (ValueError, TypeError):
        parameters = []
    positional = [
        item.name
        for item in parameters
        if item.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(item.kind == inspect.Parameter.VAR_POSITIONAL for item in parameters):
        return [f"p{i}" for i in range(p)]
    return positional[1:] if len(positional) == p + 1 else [f"p{i}" for i in range(p)]


def fit_summary(
    func: _Model,
    x: ArrayLike,
    y: ArrayLike,
    popt: ArrayLike,
    pcov: ArrayLike,
    param_names: list[str] | None = None,
    *,
    alpha: float = 0.05,
    precision: int = 5,
) -> str:
    """Produce an aligned ASCII table of approximate nonlinear Wald inference.

    Parameters
    ----------
    func : callable
        Model ``func(x, *popt)``; positional parameter names are inferred.
    x, y : array_like
        Original aligned fitting data, used to determine N-p.
    popt, pcov : array_like
        Parameter estimates and their covariance.
    param_names : list of str, optional
        Explicit parameter names overriding signature inspection.
    alpha : float
        Two-sided significance level.
    precision : int
        Positive number of significant digits in numeric columns.

    Returns
    -------
    str
        Table with estimates, standard errors, t-statistics, two-tailed p-values,
        confidence endpoints, sample size, dof, and confidence level.

    Raises
    ------
    ValueError
        Invalid data, covariance shape/finite PSD values, names, alpha, or precision.
    FloatingPointError
        A roundoff covariance repair produces unrepresentable numerical results.

    Notes
    -----
    SE_i = sqrt(pcov[i,i]); t_i = popt[i]/SE_i. Two-sided p-values are
    2*StudentT.sf(abs(t_i), N-p), and intervals use the Student-t
    upper-tail quantile at alpha/2. This function always uses Student-t,
    including when a caller supplies known-scale or ODR covariance; it does
    not automatically calibrate those alternative estimator conventions.

    These are approximate Wald tests against zero, not exact nonlinear tests.
    Singular covariance or a rank-deficient/unverifiable parameter Jacobian
    produces NaN standard errors, tests, and intervals instead of spurious
    marginal significance. Numerical rank uses column normalization and a
    relative singular-value cutoff of at least 1e-7. Reduce fixed/tied
    parameterizations before requesting unrestricted Student-t inference.
    Invalid finite covariance raises ValueError. Non-finite covariance or
    nonpositive dof warns and displays NaN for undefined inference. Zero standard
    errors do not imply infinite significance: tests are shown as NaN. Active
    bounds, small samples, and non-identifiability invalidate ordinary Wald
    interpretations; supply an appropriate unrestricted covariance. Overflowing
    statistics or unrepresentable confidence endpoints warn and display NaN.

    Examples
    --------
    >>> text = fit_summary(lambda x, slope: slope*x, [1., 2., 3.],
    ...                    [2.1, 3.9, 6.0], [2.], [[0.01]])
    >>> 'slope' in text and 'Pr(>|t|)' in text
    True
    """
    _alpha(alpha)
    precision = _count(precision, "precision")
    beta = _parameters(popt)
    _, observations, _ = _data(func, x, y, beta)
    names = _names(func, beta.size, param_names)
    dof = observations.size - beta.size
    covariance = _real(pcov, "pcov")
    if covariance.shape != (beta.size, beta.size):
        raise ValueError("pcov must have one row and column per parameter.")
    if np.any(np.diag(covariance) < 0):
        raise ValueError("pcov diagonal variances must be nonnegative.")
    standard_errors = np.full(beta.size, np.nan)
    if np.isfinite(covariance).all():
        covariance, eigenvalues, _ = _covariance(covariance, beta.size)
        standard_errors = np.sqrt(np.diag(covariance))
        if eigenvalues[0] <= 100 * np.finfo(float).eps * beta.size:
            reason = (
                "Zero standard errors"
                if np.any(standard_errors == 0)
                else "Singular parameter covariance"
            )
            _warn(
                f"{reason}: unrestricted Wald inference is undefined; reduce the model."
            )
            standard_errors[:] = np.nan
        elif dof > 0:
            try:
                derivative = _jacobian(func, _coordinates(x), beta)
                scales = _norms(derivative, axis=0)
                singular = linalg.svdvals(
                    derivative / np.where(scales == 0, 1.0, scales)
                )
                cutoff = max(1e-7, max(derivative.shape) * np.finfo(float).eps)
                if np.count_nonzero(singular > cutoff * singular[0]) < beta.size:
                    raise ValueError("Parameter Jacobian is rank deficient.")
            except (ValueError, FloatingPointError, linalg.LinAlgError) as error:
                _warn(f"Local Wald inference cannot be validated: {error}")
                standard_errors[:] = np.nan
    else:
        _warn(
            "Non-finite covariance: parameter uncertainty "
            "and Wald inference are undefined."
        )
    statistic = np.full(beta.size, np.nan)
    probability, lower, upper = statistic.copy(), statistic.copy(), statistic.copy()
    if dof > 0:
        valid = np.isfinite(standard_errors) & (standard_errors > 0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            statistic[valid] = beta[valid] / standard_errors[valid]
            half_width = float(stats.t.isf(alpha / 2, dof)) * standard_errors
            lower, upper = beta - half_width, beta + half_width
        overflow = valid & ~np.isfinite(statistic)
        if np.any(overflow):
            _warn(
                "Wald statistics exceed float64 range; "
                "displaying NaN instead of infinite significance."
            )
            statistic[overflow] = np.nan
        probability[valid] = 2 * stats.t.sf(np.abs(statistic[valid]), dof)
        bad_interval = valid & (
            ~np.isfinite(lower)
            | ~np.isfinite(upper)
            | ((half_width > 0) & (lower == upper))
        )
        if np.any(bad_interval):
            _warn(
                "Wald confidence endpoints are not representable in float64; "
                "displaying NaN."
            )
            lower[bad_interval] = np.nan
            upper[bad_interval] = np.nan
        if np.any(standard_errors == 0):
            _warn(
                "Zero standard errors do not support defined Wald tests against zero."
            )
            lower[standard_errors == 0] = np.nan
            upper[standard_errors == 0] = np.nan
    else:
        _warn("Nonpositive residual degrees of freedom: Wald inference is undefined.")
    columns = [
        "Parameter",
        "Estimate",
        "Std. Error",
        "t-value",
        "Pr(>|t|)",
        "CI Lower",
        "CI Upper",
    ]
    rows = [columns]
    for index, name in enumerate(names):
        values = [
            beta[index],
            standard_errors[index],
            statistic[index],
            probability[index],
            lower[index],
            upper[index],
        ]
        rows.append(
            [name]
            + [
                "NaN" if np.isnan(value) else f"{value:.{precision}g}"
                for value in values
            ]
        )
    widths = [max(len(row[j]) for row in rows) for j in range(len(columns))]
    table = [
        " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True))
        for row in rows
    ]
    table.insert(1, "-+-".join("-" * width for width in widths))
    return (
        f"Approximate nonlinear Wald summary\nN = {observations.size}; "
        f"residual dof = {dof}; confidence = {100 * (1 - alpha):g}%\n"
        + "\n".join(table)
    )


def profile_likelihood_errors(
    func: _Model,
    x: ArrayLike,
    y: ArrayLike,
    popt: ArrayLike,
    sigma: ArrayLike | None = None,
    sigma_level: float = 1.0,
    bounds: tuple[Any, Any] = (-np.inf, np.inf),
    *,
    pcov: ArrayLike | None = None,
    max_nfev: int = 10000,
    max_bracket_steps: int = 50,
    tolerance: float = 1e-6,
    absolute_sigma: bool = False,
) -> list[dict[str, float]]:
    """Profile parameters with bounded nuisance fits and likelihood thresholds.

    Parameters
    ----------
    func : callable
        Model ``func(x, *popt)``.
    x, y : array_like
        Aligned fitting coordinates and observations.
    popt : array_like
        Locally optimal parameters; an appreciably suboptimal input is rejected.
    sigma : array_like, optional
        Standard deviations or positive-definite observation covariance.
    sigma_level : float
        Normal-equivalent central coverage; 1 corresponds to about 68.27%.
    bounds : tuple
        Scalar or vector lower and upper parameter bounds.
    pcov : array_like, optional
        Covariance used only to select initial search step sizes.
    max_nfev, max_bracket_steps : int
        Limits for each optimizer run and each directional bracket search.
    tolerance : float
        Positive relative root tolerance. An improvement exceeding
        tolerance*sigma_level**2 on the likelihood-ratio scale rejects the
        baseline or invalidates the affected profile direction.
    absolute_sigma : bool
        True uses known Gaussian covariance (unit covariance when sigma is
        absent). False profiles one unknown scale, including for relative sigma.

    Returns
    -------
    list of dict
        Each parameter has signed ``up`` and ``down`` deviations. An unresolved,
        optimizer-failed, or boundary-limited direction is NaN with a warning;
        a bound itself is never reported as a likelihood threshold crossing.

    Raises
    ------
    ValueError
        Invalid data, options, bounds, covariance, suboptimal starting estimates,
        or zero best SSE with unknown scale.
    RuntimeError
        Baseline nuisance optimization fails. Directional failures instead warn
        and leave the affected endpoint NaN.
    FloatingPointError
        Baseline likelihood calculations are outside float64 range.

    Notes
    -----
    Signed endpoints solve 2*(logL_best-logL_profile) = sigma_level**2
    after reoptimizing nuisance parameters. The calibration assumes a
    regular one-parameter chi-square likelihood-ratio reference.

    Known-scale likelihood ratios are weighted SSE differences. Unknown-scale
    ratios are N*log(SSE_profile/SSE_best); a zero best SSE is undefined.
    Threshold is sigma_level**2, equivalent to a chi-square(1) quantile at the
    specified normal central coverage. Nuisance fits are local, not global.
    Boundary and nonregular models need specialized coverage calibration.
    Root tolerances apply to signed parameter displacements, not absolute
    parameter values. Exactly fixed parameters are never numerically perturbed.
    Non-finite trial evaluations warn and leave that endpoint unresolved.
    Root searches allow at most 1000 iterations. A returned finite endpoint must
    resolve its likelihood threshold within 10*tolerance relative error; an
    oversized covariance seed cannot turn zero displacement into a crossing.
    Optimality checks use the same likelihood-ratio scale and threshold, not a
    fraction of total SSE. Large parameter-independent residuals therefore
    cannot relax a known-scale optimum check. Near-equal unknown-scale SSEs
    use log1p to avoid subtracting nearly equal logarithms.

    Examples
    --------
    >>> intervals = profile_likelihood_errors(lambda x, a: a*x,
    ...     [1., 2., 3.], [2.1, 3.9, 6.0], [1.992857142857],
    ...     sigma=[0.1, 0.1, 0.1], absolute_sigma=True)
    >>> intervals[0]['down'] < 0 < intervals[0]['up']
    True
    """
    absolute_sigma = _boolean(absolute_sigma, "absolute_sigma")
    if not np.isfinite(sigma_level) or sigma_level <= 0:
        raise ValueError("sigma_level must be positive and finite.")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must lie between zero and one.")
    max_nfev = _count(max_nfev, "max_nfev")
    max_bracket_steps = _count(max_bracket_steps, "max_bracket_steps")
    beta = _parameters(popt)
    coordinates, observations, residual = _data(func, x, y, beta)
    lower, upper = _bounds(bounds, beta)
    noise, _ = _noise(sigma, observations.size)
    weighted = _whiten(residual, noise)
    best = _squared_norm(weighted)
    if not absolute_sigma and best == 0:
        raise ValueError(
            "Unknown-scale profile likelihood is undefined at zero residual variance."
        )
    if sigma_level > np.sqrt(np.finfo(float).max):
        raise ValueError("sigma_level is too large to define a finite threshold.")
    threshold = sigma_level**2
    if not np.isfinite(threshold) or threshold == 0:
        raise ValueError(
            "sigma_level**2 must be positive and representable in float64."
        )
    fixed = lower == upper

    def fit_at(full: _Array, free: NDArray[np.bool_]) -> float:
        """Minimize weighted SSE and probe weak Jacobian directions."""

        def residual_function(nuisance: _Array) -> _Array:
            parameters = full.copy()
            parameters[free] = nuisance
            return _whiten(
                observations - _evaluate(func, coordinates, parameters),
                noise,
            )

        if not np.any(free):
            return _squared_norm(residual_function(np.empty(0)))

        lo, hi = lower[free], upper[free]

        solution = optimize.least_squares(
            residual_function,
            full[free],
            bounds=(lo, hi),
            max_nfev=max_nfev,
            x_scale="jac",
        )

        if not solution.success:
            raise RuntimeError(f"Nuisance optimizer failed: {solution.message}")

        point = np.asarray(solution.x, dtype=float)
        values = residual_function(point)
        value = _squared_norm(values)

        # An absolute gradient stopping criterion can report convergence
        # in nearly collinear parameter directions even when a much
        # better least-squares solution exists.
        #
        # Probe a full-rank Gauss-Newton direction independently.
        # Accept only improvements confirmed by the actual residual
        # objective, never by the linear approximation alone.

        for _ in range(3):
            columns: list[_Array] = []

            for j in range(point.size):
                step = max(1.0, abs(float(point[j])))

                plus = point.copy()
                minus = point.copy()

                plus[j] = min(float(hi[j]), point[j] + step)
                minus[j] = max(float(lo[j]), point[j] - step)

                if plus[j] == minus[j]:
                    raise RuntimeError("Cannot verify nuisance Jacobian at a bound.")

                try:
                    column = (residual_function(plus) - residual_function(minus)) / (
                        plus[j] - minus[j]
                    )
                except (ValueError, FloatingPointError):
                    columns = []
                    break

                columns.append(column)

            if not columns:
                break

            jacobian = np.column_stack(columns)

            try:
                displacement, _, rank, _ = linalg.lstsq(
                    jacobian,
                    -values,
                    cond=np.finfo(float).eps,
                )
            except linalg.LinAlgError as error:
                raise RuntimeError("Nuisance Jacobian decomposition failed.") from error

            if rank < point.size:
                break

            improved = False

            for multiplier in (1.0, 0.5, 0.25, 0.125, 0.0625):
                with np.errstate(over="ignore", invalid="ignore"):
                    candidate = np.clip(
                        point + multiplier * displacement,
                        lo,
                        hi,
                    )

                if not np.isfinite(candidate).all() or np.array_equal(candidate, point):
                    continue

                try:
                    candidate_values = residual_function(candidate)
                    candidate_sse = _squared_norm(candidate_values)
                except (ValueError, FloatingPointError):
                    continue

                if candidate_sse < value:
                    point = candidate
                    values = candidate_values
                    value = candidate_sse
                    improved = True
                    break

            if not improved:
                break

        return value

    def likelihood_ratio(sse: float) -> float:
        """Evaluate a signed profile deviance difference from the input fit."""
        change = sse - best
        if absolute_sigma:
            return change
        if sse == 0:
            return float("-inf")
        if abs(change) < best * 0.5:
            return float(observations.size * np.log1p(change / best))
        return float(observations.size * (np.log(sse) - np.log(best)))

    optimized = fit_at(beta, ~fixed)
    if likelihood_ratio(optimized) < -tolerance * threshold:
        raise ValueError("popt is not a local optimum; refit before profiling.")
    if pcov is not None:
        covariance, _, _ = _covariance(pcov, beta.size)
        steps = np.sqrt(np.diag(covariance))
    else:
        # Absolute local curvature provides a physically scaled bracket seed.
        derivative = np.zeros((observations.size, beta.size))
        for free_index in np.flatnonzero(~fixed):
            derivative[:, free_index] = _central_column(
                func, coordinates, beta, int(free_index)
            )
        curvature = _norms(_whiten(derivative, noise), axis=0)
        steps = np.divide(
            1.0, curvature, out=np.zeros_like(curvature), where=curvature > 0
        )
        if not absolute_sigma:
            steps *= np.sqrt(
                best / max(observations.size - int(np.count_nonzero(~fixed)), 1)
            )
    fallback = 0.01 * np.maximum(np.abs(beta), 1.0)
    steps = np.where(np.isfinite(steps) & (steps > 0), steps, fallback)
    results: list[dict[str, float]] = []
    for index in range(beta.size):
        endpoints: dict[str, float] = {}
        free = ~fixed
        free[index] = False
        for key, sign, bound in (("down", -1, lower[index]), ("up", 1, upper[index])):
            if bound == beta[index]:
                _warn(
                    f"Parameter {index} {key} endpoint is boundary-limited; "
                    "returning NaN."
                )
                endpoints[key] = float("nan")
                continue

            def objective(
                displacement: float,
                parameter_index: int = index,
                free_mask: NDArray[np.bool_] = free,
            ) -> float:
                """Evaluate the likelihood-ratio threshold at a fixed parameter."""
                parameters = beta.copy()
                parameters[parameter_index] += displacement
                sse = fit_at(parameters, free_mask)
                ratio = likelihood_ratio(sse)
                if ratio < -tolerance * threshold:
                    raise RuntimeError(
                        "Profile found a better optimum; refit the base model."
                    )
                return float(ratio - threshold)

            step = float(steps[index])
            endpoint = float("nan")
            try:
                for _ in range(max_bracket_steps):
                    with np.errstate(over="ignore", invalid="ignore"):
                        trial = float(beta[index] + sign * step)
                    trial = (
                        max(trial, float(bound))
                        if sign < 0
                        else min(trial, float(bound))
                    )
                    if not np.isfinite(trial):
                        break
                    displacement = float(trial - beta[index])
                    value = objective(displacement)
                    if value >= 0:
                        left, right = sorted((0.0, displacement))
                        root = optimize.brentq(
                            objective,
                            left,
                            right,
                            xtol=np.finfo(float).tiny,
                            rtol=max(tolerance, 4 * np.finfo(float).eps),
                            maxiter=1000,
                        )
                        if abs(objective(float(root))) > 10 * tolerance * threshold:
                            raise RuntimeError(
                                "Endpoint does not resolve the likelihood threshold; "
                                "rescale parameters or tighten nuisance optimization."
                            )
                        endpoint = float(root)
                        break
                    if trial == bound:
                        break
                    step *= 2
            except (
                RuntimeError,
                ValueError,
                FloatingPointError,
                linalg.LinAlgError,
            ) as error:
                _warn(f"Parameter {index} {key} profile failed: {error}")
            if np.isnan(endpoint):
                _warn(
                    f"Parameter {index} {key} threshold was not resolved "
                    "(bound, flat profile, "
                    "or optimization limit); returning NaN."
                )
            endpoints[key] = endpoint
        results.append(endpoints)
    return results


def bootstrap_curve_fit(
    func: _Model,
    x: ArrayLike,
    y: ArrayLike,
    p0: ArrayLike | None = None,
    n_resamples: int = 1000,
    random_state: int | None = None,
    **kwargs: Any,
) -> tuple[_Array, _Array, _Array]:
    """Refit a fixed number of centered, exchangeable residual resamples.

    Parameters
    ----------
    func : callable
        SciPy-style model.
    x, y : array_like
        Original aligned data.
    p0 : array_like, optional
        Initial guess for the base fit; SciPy signature inference otherwise.
    n_resamples : int
        Number of attempted replicates, not a requested number of successes.
    random_state : int, optional
        Seed for a local numpy.random.Generator.
    **kwargs
        Compatible curve_fit options, including bounds and analytic jac.
        sigma, full_output=True, nan_policy, and check_finite=False are rejected.

    Returns
    -------
    mean, covariance, samples : ndarray
        Shapes (p,), (p, p), and (n_successful, p). Covariance uses B-1.

    Raises
    ------
    ValueError
        Invalid data, replicate count, seed, or unsupported bootstrap options.
    RuntimeError
        Base covariance is unusable or fewer than two replicates succeed;
        a base optimizer failure also propagates.
    FloatingPointError
        Empirical mean or covariance is outside float64 range.
    TypeError
        An unsupported keyword reaches scipy.optimize.curve_fit.

    Notes
    -----
    For B successful parameter samples theta_b, the empirical covariance
    is sum(outer(theta_b-mean, theta_b-mean))/(B-1). Returned samples are
    rows, and no residual degrees-of-freedom inflation is applied.

    Fits internally subtract a common response origin and normalize response
    units, including supplied analytic Jacobians. This prevents absolute
    trust-region stopping tolerances from certifying an unchanged initial guess
    in tiny response units. Known-unit base/refit covariance is transformed back
    before reliability checks; returned bootstrap samples retain parameter units.

    Assumes independent, identically distributed observation errors with zero
    mean. Fitted residuals are generally correlated and have leverage-dependent
    variances; their empirical distribution is an approximation to the error law.
    Residuals are centered before resampling; no leverage correction is applied.
    Population covariance consistency requires regular fitting and a suitable
    asymptotic design (for example, vanishing maximum leverage), not just finite
    successful refits. Finite-sample covariance is not generally unbiased.
    Population parameter covariance must exist; ordinary least squares requires
    finite error variance.
    Weighted/correlated noise is not supported by this bootstrap.
    An invalid/non-finite/singular base covariance or rank-deficient parameter
    Jacobian raises RuntimeError before resampling. Each refit is checked as well,
    including trust-region pseudoinverse covariance without OptimizeWarning.
    Numerical Jacobian rank uses a column-normalized relative cutoff of at least
    1e-7; analytic jac uses max(N,p)*eps. Supply analytic jac for models whose
    derivatives cannot be evaluated in an open neighborhood of a bound. A full-
    rank exact fit may return zero empirical spread; this does not establish
    zero population parameter uncertainty.
    Failed fits are excluded with a count warning; fewer than two successes
    raises RuntimeError. Less than 90% success triggers an additional warning.
    This does not provide universal outlier robustness or fix model misspecification.
    Non-finite synthetic observations count as failures. Unrepresentable empirical
    mean/covariance raises FloatingPointError; rescale parameters in that case.

    Examples
    --------
    >>> mean, covariance, samples = bootstrap_curve_fit(lambda x, a: a*x,
    ...     [1., 2., 3., 4.], [2.1, 3.9, 6.1, 7.9], [2.],
    ...     n_resamples=10, random_state=3)
    >>> samples.shape, covariance.shape
    ((10, 1), (1, 1))
    """
    n_resamples = _count(n_resamples, "n_resamples", 2)
    if kwargs.get("sigma") is not None:
        raise ValueError(
            "Residual bootstrap requires unweighted exchangeable errors; "
            "sigma is unsupported."
        )
    if kwargs.get("full_output", False):
        raise ValueError("full_output=True is not supported by bootstrap_curve_fit.")
    if (
        kwargs.get("nan_policy") is not None
        or kwargs.get("check_finite", True) is False
    ):
        raise ValueError("Bootstrap requires finite observations and model outputs.")
    observations = _vector(y, "y")
    coordinates = _coordinates(x, observations.size)
    initial = None if p0 is None else _parameters(p0)
    origin_y = float(observations[0])
    with np.errstate(over="ignore", invalid="ignore"):
        shifted_y = _computed(observations - origin_y, "Centered fitting observations")
    measurement_scale = float(np.max(np.abs(shifted_y)))
    if measurement_scale == 0:
        probe = initial
        if probe is None:
            positional = [
                parameter
                for parameter in inspect.signature(func).parameters.values()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if len(positional) < 2:
                raise ValueError(
                    "Supply p0 when the model parameter count is ambiguous."
                )
            probe = np.ones(len(positional) - 1)
            limits = kwargs.get("bounds", (-np.inf, np.inf))
            if isinstance(limits, optimize.Bounds):
                limits = (limits.lb, limits.ub)
            lower = np.broadcast_to(_real(limits[0], "lower bounds"), probe.shape)
            upper = np.broadcast_to(_real(limits[1], "upper bounds"), probe.shape)
            both = np.isfinite(lower) & np.isfinite(upper)
            probe[both] = lower[both] / 2 + upper[both] / 2
            probe[np.isfinite(lower) & ~both] = lower[np.isfinite(lower) & ~both] + 1
            probe[np.isfinite(upper) & ~both] = upper[np.isfinite(upper) & ~both] - 1
        measurement_scale = (
            float(
                np.max(
                    np.abs(
                        _computed(
                            _evaluate(func, coordinates, probe) - origin_y,
                            "Initial model offsets",
                        )
                    )
                )
            )
            or 1.0
        )

    @wraps(func)
    def normalized_func(x: _Array, *parameters: float) -> NDArray[Any]:
        """Preserve complex-step evaluations while centering response units."""
        return (np.asarray(func(x, *parameters)) - origin_y) / measurement_scale

    fit_options = kwargs.copy()
    if kwargs.get("loss", "linear") != "linear":
        # Preserve the caller's robust-loss threshold in response units.
        fit_options["f_scale"] = float(kwargs.get("f_scale", 1.0)) / measurement_scale
    supplied_jac = kwargs.get("jac")
    if callable(supplied_jac):

        def normalized_jac(x: _Array, *parameters: float) -> NDArray[Any]:
            """Apply the same response scale to an analytical model Jacobian."""
            return np.asarray(supplied_jac(x, *parameters)) / measurement_scale

        fit_options["jac"] = normalized_jac

    def restore_covariance(value: ArrayLike) -> _Array:
        """Undo objective scaling for known unit response variance."""
        covariance = _real(value, "Fit covariance")
        if kwargs.get("absolute_sigma", False):
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                covariance = covariance / measurement_scale / measurement_scale
        return covariance

    scipy_func = cast(Callable[..., _Array], normalized_func)
    base, base_covariance = optimize.curve_fit(
        scipy_func,
        coordinates,
        shifted_y / measurement_scale,
        p0=initial,
        **fit_options,
    )
    base_covariance = restore_covariance(base_covariance)
    beta = _parameters(base)
    predictions = _evaluate(func, coordinates, beta)
    if observations.size <= beta.size:
        raise ValueError("Residual bootstrap requires N > p.")

    def validate_fit(parameters: _Array, covariance: ArrayLike, data: _Array) -> None:
        """Reject finite pseudoinverse covariance from unidentified refits."""
        checked, eigenvalues, _ = _covariance(covariance, parameters.size)
        supplied_jac = kwargs.get("jac")
        derivative = _jacobian(
            func,
            coordinates,
            parameters,
            supplied_jac if callable(supplied_jac) else None,
        )
        scales = _norms(derivative, axis=0)
        singular = linalg.svdvals(derivative / np.where(scales == 0, 1.0, scales))
        cutoff = max(derivative.shape) * np.finfo(float).eps
        if not callable(supplied_jac):
            cutoff = max(cutoff, 1e-7)
        if (
            singular.size < parameters.size
            or np.count_nonzero(singular > cutoff * singular[0]) < parameters.size
        ):
            raise ValueError("Parameter Jacobian is rank deficient.")
        # A noiseless relative-scale fit can have genuinely zero empirical
        # spread. Positive-noise singular covariance cannot certify a refit.
        exact = not np.any(data - _evaluate(func, coordinates, parameters))
        if np.all(checked == 0) and exact and not kwargs.get("absolute_sigma", False):
            return
        if eigenvalues[0] <= 100 * np.finfo(float).eps * parameters.size:
            raise ValueError("Fit covariance is numerically singular.")

    try:
        validate_fit(beta, base_covariance, observations)
    except (ValueError, FloatingPointError, linalg.LinAlgError) as error:
        raise RuntimeError(
            "Base fit covariance is unreliable; "
            "refit an identifiable model before bootstrap."
        ) from error
    with np.errstate(over="ignore", invalid="ignore"):
        residual = _computed(observations - predictions, "Base fit residuals")
        residual_scale = max(float(np.max(np.abs(residual))), np.finfo(float).tiny)
        residual = _computed(
            residual - (residual / residual_scale).mean() * residual_scale,
            "Centered residuals",
        )
    generator = _generator(random_state)
    successful: list[_Array] = []
    failures = 0
    for _ in range(n_resamples):
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                synthetic = _computed(
                    predictions
                    + generator.choice(residual, size=observations.size, replace=True),
                    "Bootstrap observations",
                )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", optimize.OptimizeWarning)
                fitted, fitted_covariance = optimize.curve_fit(
                    scipy_func,
                    coordinates,
                    (synthetic - origin_y) / measurement_scale,
                    p0=beta.copy(),
                    **fit_options,
                )
                fitted_covariance = restore_covariance(fitted_covariance)
            for item in caught:
                if not issubclass(item.category, optimize.OptimizeWarning):
                    warnings.warn(item.message, item.category, stacklevel=2)
            candidate = _parameters(fitted)
            _evaluate(func, coordinates, candidate)
            if any(
                issubclass(item.category, optimize.OptimizeWarning) for item in caught
            ):
                failures += 1
                continue
            validate_fit(candidate, fitted_covariance, synthetic)
        except (RuntimeError, ValueError, FloatingPointError, linalg.LinAlgError):
            failures += 1
            continue
        successful.append(candidate)
    if failures:
        _warn(
            f"{failures} of {n_resamples} bootstrap attempts failed or had "
            "unreliable covariance; excluded."
        )
    if len(successful) < 2:
        raise RuntimeError(
            "At least two successful bootstrap fits are required for covariance."
        )
    if len(successful) / n_resamples < 0.9:
        _warn(
            "Fewer than 90% of bootstrap attempts succeeded; "
            "empirical inference may be unreliable."
        )
    samples = np.stack(successful)
    scale = np.max(np.abs(samples), axis=0)
    safe = np.where(scale == 0, 1.0, scale)
    normalized = samples / safe
    # Subtract an observed origin before scaling: division of large, nearly
    # equal values otherwise quantizes their small, representable differences.
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = samples - samples[0]
    if np.isfinite(offsets).all():
        spread_scale = np.max(np.abs(offsets), axis=0)
        spread_scale = np.where(spread_scale == 0, 1.0, spread_scale)
        spread = offsets / spread_scale
        origin = samples[0]
    else:
        spread_scale = safe
        spread = normalized - normalized[0]
        origin = samples[0]
    with np.errstate(over="ignore", invalid="ignore"):
        mean = _computed(
            origin + spread.mean(axis=0) * spread_scale
            if np.isfinite(offsets).all()
            else normalized.mean(axis=0) * safe,
            "Bootstrap mean",
        )
        centered = (
            (spread - spread.mean(axis=0))
            * spread_scale
            / np.sqrt(samples.shape[0] - 1)
        )
        covariance = _computed(centered.T @ centered, "Bootstrap covariance")
    if np.any((np.diag(covariance) == 0) & np.any(centered != 0, axis=0)):
        raise FloatingPointError(
            "Bootstrap covariance underflows float64; rescale parameters."
        )
    return mean, covariance, samples


def collinearity_diagnostics(
    pcov: ArrayLike,
    *,
    correlation_threshold: float = 0.99,
) -> dict[str, Any]:
    """Diagnose covariance-derived local parameter dependence.

    Parameters
    ----------
    pcov : array_like
        Square parameter covariance matrix.
    correlation_threshold : float
        Absolute off-diagonal correlation flag threshold, strictly between 0 and 1.

    Returns
    -------
    dict
        correlation_matrix, vif, max_abs_correlation, condition_number,
        high_correlation_pairs, potential_identifiability_issue, and rank.
        Rank is None when non-finite covariance prevents its determination.
        Pair indices are zero-based. Condition and rank refer to correlation
        among parameters with positive variance; invalid zero-variance directions
        are excluded and flagged. Singular valid correlation has infinite VIF
        and condition number rather than a finite pseudoinverse substitute.

    Raises
    ------
    ValueError
        Invalid covariance shape, finite PSD values, or correlation threshold.
    FloatingPointError
        A roundoff covariance repair produces unrepresentable numerical results.

    Notes
    -----
    For positive variances, R[i,j] = V[i,j]/sqrt(V[i,i]*V[j,j]),
    vif = diag(inv(R)), and condition_number = max(eig(R))/min(eig(R)).
    These use correlation units, not the raw covariance condition number.

    VIF-like values are inverse correlation diagonals from a stable solver.
    They are not generally classical design-matrix VIFs for nonlinear models.
    Strong correlation indicates local dependence, not structural non-identifiability.

    Examples
    --------
    >>> result = collinearity_diagnostics([[1., 0.5], [0.5, 1.]])
    >>> result['high_correlation_pairs']
    []
    """
    if not np.isfinite(correlation_threshold) or not 0 < correlation_threshold < 1:
        raise ValueError("correlation_threshold must lie between zero and one.")
    supplied = _real(pcov, "pcov")
    if (
        supplied.ndim != 2
        or supplied.shape[0] != supplied.shape[1]
        or supplied.shape[0] == 0
    ):
        raise ValueError("pcov must be a nonempty square matrix.")
    p = supplied.shape[0]
    correlation = np.full((p, p), np.nan)
    vif = np.full(p, np.nan)
    pairs: list[tuple[int, int]] = []
    condition, maximum = float("nan"), float("nan")
    rank: int | None = 0
    if np.any(np.diag(supplied) < 0):
        raise ValueError("pcov diagonal variances must be nonnegative.")
    unreliable = not np.isfinite(supplied).all()
    if unreliable:
        rank = None
        _warn("Non-finite covariance cannot support reliable collinearity diagnostics.")
    else:
        covariance, _, _ = _covariance(supplied, p)
        diagonal = np.diag(covariance)
        valid = diagonal > 0
        unreliable = not bool(np.all(valid))
        if unreliable:
            _warn(
                "Zero parameter variance: correlations and VIF are "
                "undefined for those parameters."
            )
        indices = np.flatnonzero(valid)
        if indices.size:
            standard = np.sqrt(diagonal[valid])
            reduced = (
                covariance[np.ix_(valid, valid)] / standard[:, None] / standard[None, :]
            )
            np.fill_diagonal(reduced, 1.0)
            correlation[np.ix_(valid, valid)] = reduced
            eigenvalues = linalg.eigvalsh(reduced)
            tolerance = np.finfo(float).eps * indices.size * eigenvalues[-1]
            rank = int(np.count_nonzero(eigenvalues > tolerance))
            if rank < indices.size:
                condition = float("inf")
                vif[valid] = np.inf
                unreliable = True
                _warn(
                    "Singular parameter correlation: finite VIF and "
                    "ordinary inference are undefined."
                )
            else:
                condition = float(eigenvalues[-1] / eigenvalues[0])
                vif[valid] = np.diag(
                    linalg.solve(reduced, np.eye(indices.size), assume_a="pos")
                )
            maximum = 0.0
            for i in range(p):
                for j in range(i + 1, p):
                    if np.isfinite(correlation[i, j]):
                        maximum = max(maximum, float(abs(correlation[i, j])))
                        if abs(correlation[i, j]) > correlation_threshold:
                            pairs.append((i, j))
            if pairs:
                _warn(
                    "Strong local parameter correlation exceeds the threshold; "
                    "this does not establish structural non-identifiability."
                )
    return {
        "correlation_matrix": correlation,
        "vif": vif,
        "max_abs_correlation": maximum,
        "condition_number": condition,
        "high_correlation_pairs": pairs,
        "potential_identifiability_issue": unreliable or bool(pairs),
        "rank": rank,
    }


def robust_bounded_pcov(
    func: _Model,
    x: ArrayLike,
    y: ArrayLike,
    popt: ArrayLike,
    bounds: tuple[Any, Any],
    *,
    sigma: ArrayLike | None = None,
    absolute_sigma: bool = False,
    jac: _Jacobian | None = None,
    bound_tolerance: float = 1e-8,
) -> _Array:
    """Estimate conditional local covariance of parameters away from bounds.

    Parameters
    ----------
    func : callable
        Model ``func(x, *popt)``.
    x, y : array_like
        Aligned fitting data.
    popt : array_like
        Parameters within bounds.
    bounds : tuple
        Scalar or vector lower/upper bounds; equality denotes an exactly fixed
        parameter but still produces NaN covariance rows/columns here.
    sigma : array_like, optional
        Standard deviations or positive-definite observation covariance.
    absolute_sigma : bool
        If False, scale free covariance by weighted RSS/(N-p_free).
    jac : callable, optional
        Model Jacobian of shape (N, p); finite differences otherwise.
    bound_tolerance : float
        Nonnegative tolerance multiplied by max(1, |parameter|, |finite bound|)
        separately for each bound. Numerical derivatives of free parameters
        require the model to be defined in a small open neighborhood.

    Returns
    -------
    ndarray
        Full (p, p) matrix. Active rows/columns are NaN. Free covariance is
        conditional on active parameters, not unrestricted marginal uncertainty.
        Rank-deficient free Jacobians or undefined scaling return NaN blocks.

    Raises
    ------
    ValueError
        Invalid data, parameters, bounds, sigma, Jacobian, or activity tolerance.
    FloatingPointError
        Numerical differentiation or whitening produces non-finite results.

    Notes
    -----
    Holding active parameters fixed, the free block is approximately
    inv(J_free.T @ inv(C) @ J_free). Unknown scale multiplies it by
    Q/(N-p_free). Active rows/columns are NaN, not zero uncertainty.

    This does not repair structural non-identifiability, nor infer one-sided
    uncertainty for active parameters. All-active inputs return all NaN.
    Conditional means holding active values fixed in the model; it does not
    calibrate the sampling distribution conditional on selecting that active set.
    Effective dof is N-p_free under the stated local conditional approximation.
    Rank is assessed after Jacobian column normalization, then covariance is
    transformed back to the original parameter units. A covariance outside
    float64 range warns and returns NaN, rather than finite zero uncertainty.

    Examples
    --------
    >>> covariance = robust_bounded_pcov(lambda x, a, b: a*x+b,
    ...     [0., 1., 2.], [1., 3., 5.], [2., 1.],
    ...     ([0., 1.], [3., 1.]), absolute_sigma=True)
    >>> bool(np.isnan(covariance[1]).all())
    True
    """
    absolute_sigma = _boolean(absolute_sigma, "absolute_sigma")
    if not np.isfinite(bound_tolerance) or bound_tolerance < 0:
        raise ValueError("bound_tolerance must be finite and nonnegative.")
    beta = _parameters(popt)
    coordinates, observations, residual = _data(func, x, y, beta)
    lower, upper = _bounds(bounds, beta)
    noise, _ = _noise(sigma, observations.size)
    active = np.zeros(beta.size, dtype=bool)
    for boundary in (lower, upper):
        finite = np.isfinite(boundary)
        scale = np.maximum(
            1.0, np.maximum(np.abs(beta[finite]), np.abs(boundary[finite]))
        )
        active[finite] |= (
            np.abs(beta[finite] / scale - boundary[finite] / scale) <= bound_tolerance
        )
    covariance = np.full((beta.size, beta.size), np.nan)
    if np.any(active):
        _warn(
            "Bound-active parameter rows and columns are NaN; "
            "free covariance is conditional."
        )
    if np.all(active):
        return covariance
    # Do not differentiate fixed columns: models may be undefined outside bounds.
    if jac is None:
        free_indices = np.flatnonzero(~active)
        derivative = np.empty((observations.size, free_indices.size))
        for column, index in enumerate(free_indices):
            derivative[:, column] = _central_column(func, coordinates, beta, int(index))
    else:
        derivative = _jacobian(func, coordinates, beta, jac)[:, ~active]
    free_covariance = _local_covariance(
        _whiten(derivative, noise), _whiten(residual, noise), absolute_sigma
    )
    covariance[np.ix_(~active, ~active)] = free_covariance
    return covariance


def propagate_uncertainty(
    derived_func: Callable[..., Any],
    popt: ArrayLike,
    pcov: ArrayLike,
    *,
    n_samples: int = 10000,
    alpha: float = 0.05,
    random_state: int | None = None,
    min_valid_fraction: float = 0.9,
) -> dict[str, Any]:
    """Propagate Gaussian parameter uncertainty by sample-wise Monte Carlo.

    Parameters
    ----------
    derived_func : callable
        Scalar or fixed-shape vector output from ``derived_func(*parameters)``.
        No vectorization is assumed; arbitrary callable errors propagate.
    popt, pcov : array_like
        Parameter mean and finite positive-semidefinite covariance.
    n_samples : int
        Number of attempted samples, at least two.
    alpha : float
        Two-sided empirical interval significance level.
    random_state : int, optional
        Seed for a local Generator, without modifying global RNG state.
    min_valid_fraction : float
        Minimum fraction of finite outputs, in (0, 1]; default 0.9. At least two
        valid outputs are always required. Lower fractions raise ValueError.

    Returns
    -------
    dict
        mean, median, std (ddof=1), lower, upper, n_samples, n_valid, n_invalid.
        Statistics are Python floats for scalar output and arrays for vectors.
        A vector sample is rejected in full if any component is non-finite.

    Raises
    ------
    ValueError
        Invalid covariance, parameter/sample options, changing output shape,
        or too few valid derived samples.
    FloatingPointError
        Parameter uncertainty is below float64 resolution or summaries are
        unrepresentable. Exceptions raised by derived_func propagate unchanged.

    Notes
    -----
    Draw theta from Normal(popt, pcov), then evaluate g(theta) sample-wise.
    Standard deviation uses the valid sample count minus one; endpoints are
    empirical alpha/2 and 1-alpha/2 quantiles, not simultaneous vector regions.

    Covariance PSD and rank checks use correlation units so parameter units
    cannot hide negative variance or create false singularity. Negative diagonal
    variances are rejected. Correlation eigenvalues in [-100*eps*p, 0) are repaired
    with a warning; zero variance requires an exactly zero row and column.
    Singular covariance warns. Non-finite outputs are counted and warned about;
    retained quantiles
    describe the valid subset and may be biased by rejection. A Gaussian
    approximation may violate physical bounds or miss asymmetric likelihoods.
    Finite ratio outputs do not guarantee that population moments exist.
    Unrepresentable numerical summaries or parameter uncertainty below floating
    point resolution at popt raise FloatingPointError; recenter/rescale instead.
    Seed reproducibility is guaranteed within a fixed NumPy/LAPACK environment,
    not bit-for-bit across backend versions or repeated eigenvalue conventions.

    Examples
    --------
    >>> result = propagate_uncertainty(lambda a, b: a+b, [1., 2.],
    ...     [[0.01, 0.], [0., 0.01]], n_samples=100, random_state=7)
    >>> result['n_valid']
    100
    """
    _alpha(alpha)
    n_samples = _count(n_samples, "n_samples", 2)
    if not np.isfinite(min_valid_fraction) or not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must lie in (0, 1].")
    beta = _parameters(popt)
    _, eigenvalues, eigenvectors = _covariance(pcov, beta.size)
    threshold = (
        100
        * np.finfo(float).eps
        * beta.size
        * max(float(eigenvalues[-1]), np.finfo(float).tiny)
    )
    if eigenvalues[0] <= threshold:
        _warn(
            "Singular or numerically singular parameter covariance: "
            "Gaussian uncertainty may be unreliable."
        )
    standard = np.sqrt(np.diag(_real(pcov, "pcov")))
    if np.any((standard > 0) & (standard <= np.finfo(float).eps * np.abs(beta))):
        raise FloatingPointError(
            "Parameter uncertainty is below float64 resolution at popt; "
            "recenter parameters."
        )
    generator = _generator(random_state)
    factor = eigenvectors * np.sqrt(eigenvalues)
    outputs: list[_Array] = []
    shape: tuple[int, ...] | None = None
    invalid = 0
    # Chunk parameter samples to limit temporary memory; only output values are
    # retained because exact empirical quantiles require them.
    remaining = n_samples
    while remaining:
        count = min(remaining, 1024)
        with np.errstate(over="ignore", invalid="ignore"):
            draws = generator.standard_normal((count, beta.size)) @ factor.T + beta
        for draw in draws:
            if not np.isfinite(draw).all():
                invalid += 1
                continue
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                value = _real(derived_func(*draw), "derived_func output")
            if value.ndim > 1 or value.size == 0:
                raise ValueError(
                    "derived_func must return a scalar or a nonempty vector."
                )
            if shape is None:
                shape = value.shape
            elif shape != value.shape:
                raise ValueError(
                    "derived_func output shape must be constant across samples."
                )
            if not np.isfinite(value).all():
                invalid += 1
            else:
                outputs.append(value.copy())
        remaining -= count
    valid = len(outputs)
    if invalid:
        _warn(
            f"{invalid} of {n_samples} derived samples were non-finite; "
            "summaries condition on valid outputs."
        )
    if valid < 2 or valid / n_samples < min_valid_fraction:
        raise ValueError(
            "Too few valid derived samples for reliable uncertainty summaries."
        )
    values = np.stack(outputs)

    def output(value: _Array | np.float64) -> float | _Array:
        """Preserve vector shapes and return Python floats for scalars."""
        finite = _computed(value, "Monte Carlo summary")
        return float(finite) if finite.ndim == 0 else finite

    # Scaling avoids intermediate overflow in means, variances, and quantiles.
    scale = np.max(np.abs(values), axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    normalized = values / scale
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = values - values[0]
    if np.isfinite(offsets).all():
        spread_scale = np.max(np.abs(offsets), axis=0)
        spread_scale = np.where(spread_scale == 0, 1.0, spread_scale)
        spread = offsets / spread_scale
        origin = values[0]
        summary_values, summary_scale = spread, spread_scale
    else:
        # Opposite extreme signs can overflow subtraction despite finite values.
        spread_scale = scale
        spread = normalized - normalized[0]
        origin = np.zeros_like(values[0])
        summary_values, summary_scale = normalized, scale
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        spread_std = spread.std(axis=0, ddof=1)
        deviation = spread_std * spread_scale
    if np.any((deviation == 0) & (spread_std > 0)):
        raise FloatingPointError(
            "Monte Carlo standard deviation underflows float64; "
            "rescale derived outputs."
        )
    return {
        "mean": output(origin + summary_values.mean(axis=0) * summary_scale),
        "median": output(origin + np.median(summary_values, axis=0) * summary_scale),
        "std": output(deviation),
        "lower": output(
            origin + np.quantile(summary_values, alpha / 2, axis=0) * summary_scale
        ),
        "upper": output(
            origin + np.quantile(summary_values, 1 - alpha / 2, axis=0) * summary_scale
        ),
        "n_samples": n_samples,
        "n_valid": valid,
        "n_invalid": invalid,
    }
