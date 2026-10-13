"""Joint least-squares estimation with explicit shared-parameter mappings.

GlobalFitGroup fits independent datasets jointly; eiv_curve_fit accounts for
independent measurement uncertainty in both axes through SciPy ODR.
Numerical helpers are shared with inference without circular imports.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, overload

import numpy as np
import scipy
from numpy.typing import ArrayLike, NDArray
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import LinearOperator

from .inference import (
    _array,
    _boolean,
    _bounds,
    _computed,
    _coordinates,
    _count,
    _covariance,
    _evaluate,
    _jacobian,
    _local_covariance,
    _noise,
    _norms,
    _real,
    _squared_norm,
    _vector,
    _whiten,
)

__all__ = ["GlobalFitGroup", "eiv_curve_fit"]

_Array = NDArray[np.float64]
_Index = NDArray[np.int64]
_Model = Callable[..., ArrayLike]
_Jacobian = Literal["2-point", "3-point"] | Callable[..., ArrayLike]


@dataclass(frozen=True, slots=True)
class _Dataset:
    """Retain validated input snapshots and a whitening transform."""

    x: _Array
    y: _Array
    noise: _Array | None
    weighted: bool


@dataclass(frozen=True, slots=True)
class _DatasetResult:
    """Expose local parameter estimates and observations in original order."""

    parameters: Mapping[str, float]
    popt: _Array
    predictions: _Array
    residuals: _Array
    weighted_residuals: _Array


@dataclass(frozen=True, slots=True)
class _SolverDiagnostics:
    """Expose SciPy convergence information without claiming a global optimum."""

    success: bool
    status: int | None
    message: str
    nfev: int
    njev: int | None
    cost: float
    optimality: float


@dataclass(frozen=True, slots=True)
class _FitDiagnostics:
    """Distinguish optimized parameters from covariance's conditional subspace."""

    n_observations: int
    n_global_parameters: int
    n_parameters: int
    dof: int
    rss: float
    rmse: float
    weighted_sse: float
    chi_square: float
    reduced_chi_square: float
    rank: int
    free_rank: int
    condition_number: float
    identifiable: bool
    covariance_dof: int
    covariance_kind: Literal["ordinary", "conditional", "undefined"]


@dataclass(frozen=True, slots=True)
class _GlobalFitResult:
    """Return global and per-dataset estimates with solver and inference status.

    Arrays are independent output copies. Frozen dataclass fields do not make
    the arrays themselves immutable. The Jacobian differentiates y-f and is
    whitened; exactly fixed columns are zero because they were not optimized.
    active_mask uses -1 for lower, +1 for upper, 2 for exactly fixed, and 0 for
    inactive parameters. Dataset and parameter mappings are read-only.
    """

    popt: _Array
    pcov: _Array
    parameter_names: tuple[str, ...]
    datasets: Mapping[str, _DatasetResult]
    jacobian: _Array
    weighted_residuals: _Array
    active_mask: _Index
    solver: _SolverDiagnostics
    diagnostics: _FitDiagnostics

    @property
    def success(self) -> bool:
        """Return convergence or successful evaluation of an exactly fixed model."""
        return self.solver.success


def _parameter_names(model: _Model, explicit: Sequence[str] | None) -> tuple[str, ...]:
    """Extract positional fitted parameters, ignoring keyword-only model options."""
    try:
        signature = inspect.signature(model)
    except (TypeError, ValueError):
        signature = None
    inferred: list[str] | None = None
    if signature is not None:
        positional = [
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        variadic = any(
            parameter.kind == inspect.Parameter.VAR_POSITIONAL
            for parameter in signature.parameters.values()
        )
        if not variadic and positional:
            inferred = positional[1:]
    if explicit is None:
        if not inferred:
            raise ValueError(
                "Cannot infer fitted parameter names; "
                "supply parameter_names explicitly."
            )
        names = tuple(inferred)
    else:
        if isinstance(explicit, str):
            raise ValueError(
                "parameter_names must be a sequence of names, not a string."
            )
        names = tuple(explicit)
        if inferred is not None and len(inferred) != len(names):
            raise ValueError(
                "parameter_names length must match positional fitted model parameters."
            )
    if not names or any(
        not isinstance(name, str) or not name.isidentifier() for name in names
    ):
        raise ValueError("Parameter names must be nonempty Python identifiers.")
    if len(set(names)) != len(names):
        raise ValueError("Parameter names must be unique.")
    return names


def _rank(jacobian: _Array, tolerance: float) -> tuple[int, float]:
    """Measure numerical rank after normalizing parameter columns."""
    if jacobian.shape[1] == 0:
        return 0, float("nan")
    scales = _norms(jacobian, axis=0)
    normalized = jacobian / np.where(scales == 0, 1.0, scales)
    singular = linalg.svdvals(normalized)
    if singular.size == 0 or singular[0] == 0:
        return 0, float("inf")
    rank = int(np.count_nonzero(singular > tolerance * singular[0]))
    condition = (
        float(singular[0] / singular[-1]) if rank == jacobian.shape[1] else float("inf")
    )
    return rank, condition


class GlobalFitGroup:
    """Fit named datasets with shared and dataset-specific model parameters.

    Parameters
    ----------
    model : callable
        Common model ``model(x, *parameters)``. Keyword-only options are not
        fitted. Different model functions per dataset are not supported.
    datasets : mapping
        Nonempty mapping of dataset names to dictionaries containing x, y, and
        optional sigma. Scalar/vector y and x with observations on the last axis
        are supported. sigma is a positive scalar/vector standard deviation or
        a positive-definite observation covariance. Inputs are copied.
    shared_parameters : set or mapping, optional
        Set of local model parameter names shared across all datasets, or a map
        from local names to shared global aliases. Explicit repeated aliases
        tie those model parameters to one global variable. Unlisted parameters
        receive independent dataset-specific variables.
    parameter_names : sequence of str, optional
        Explicit names in model argument order, required for ambiguous/variadic
        model signatures. Overrides names but must match an inferable count.

    Returns
    -------
    GlobalFitGroup
        Configured fitting group; call fit to obtain parameter estimates,
        covariance, predictions, residuals, and diagnostics.

    Raises
    ------
    ValueError
        Invalid dataset names/shapes, uncertainties, parameter names, or sharing
        mappings. Optimizer validation is deferred to fit.

    Notes
    -----
    The joint objective is sum_d r_d.T @ inv(C_d) @ r_d, using one
    optimized variable for each shared global degree of freedom. Global
    covariance is obtained from the joint whitened residual Jacobian.

    Shared globals follow first occurrence in model argument order. Dataset
    names are sorted lexically; their local globals follow model argument order
    and are named 'dataset.parameter'. Shared aliases are bare identifiers.
    The public parameter_indices property exposes local-to-global integer maps.

    Errors are independent between datasets. sigma=None means unit relative
    weight, or known unit variance when absolute_sigma=True. With False one
    unknown scale multiplies the entire block-diagonal observation covariance;
    separate unknown variance scales per dataset are not estimated. Cross-dataset
    error covariance is unsupported. The solver finds a local optimum, and rank
    diagnoses local numerical dependence rather than structural identifiability.

    Examples
    --------
    >>> import numpy as np
    >>> model = lambda x, slope, intercept: slope*x+intercept
    >>> group = GlobalFitGroup(model, {
    ...     'a': {'x': np.arange(3.), 'y': [1., 3., 5.]},
    ...     'b': {'x': np.arange(4.), 'y': [3., 5., 7., 9.]},
    ... }, shared_parameters={'slope'})
    >>> result = group.fit([1., 0., 0.], absolute_sigma=True)
    >>> result.parameter_names
    ('slope', 'a.intercept', 'b.intercept')
    >>> np.allclose(result.popt, [2., 1., 3.])
    True
    >>> result.success
    True

    """

    def __init__(
        self,
        model: _Model,
        datasets: Mapping[str, Mapping[str, ArrayLike | None]],
        shared_parameters: Set[str] | Mapping[str, str] = frozenset(),
        *,
        parameter_names: Sequence[str] | None = None,
    ) -> None:
        """Validate data and build an explicit deterministic parameter mapping."""
        if not callable(model):
            raise ValueError("model must be callable.")
        if not isinstance(datasets, Mapping) or not datasets:
            raise ValueError("datasets must be a nonempty mapping.")
        if any(not isinstance(name, str) or not name for name in datasets):
            raise ValueError("Dataset names must be nonempty strings.")
        self._model = model
        self._local_names = _parameter_names(model, parameter_names)
        if isinstance(shared_parameters, Mapping):
            sharing = dict(shared_parameters)
        elif isinstance(shared_parameters, Set):
            sharing = {name: name for name in shared_parameters}
        else:
            raise ValueError(
                "shared_parameters must be a set or local-name-to-alias mapping."
            )
        if set(sharing) - set(self._local_names):
            raise ValueError("Sharing map contains unknown model parameters.")
        if any(
            not isinstance(alias, str) or not alias.isidentifier()
            for alias in sharing.values()
        ):
            raise ValueError("Shared global aliases must be Python identifiers.")
        self._dataset_names = tuple(sorted(datasets))
        self._data: dict[str, _Dataset] = {}
        names = list(
            dict.fromkeys(
                sharing[name] for name in self._local_names if name in sharing
            )
        )
        index_by_name = {name: index for index, name in enumerate(names)}
        self._indices: dict[str, _Index] = {}
        for name in self._dataset_names:
            supplied = datasets[name]
            if not isinstance(supplied, Mapping) or not {"x", "y"} <= supplied.keys():
                raise ValueError(f"Dataset {name!r} must provide x and y.")
            if set(supplied) - {"x", "y", "sigma"}:
                raise ValueError(
                    f"Dataset {name!r} has unsupported fields; use x, y, sigma."
                )
            if supplied["x"] is None or supplied["y"] is None:
                raise ValueError(f"Dataset {name!r} requires non-null x and y.")
            y = _vector(supplied["y"], f"{name}.y").copy()
            x = _coordinates(supplied["x"], y.size).copy()
            noise, _ = _noise(supplied.get("sigma"), y.size)
            self._data[name] = _Dataset(
                x, y, None if noise is None else noise.copy(), noise is not None
            )
            indices: list[int] = []
            for parameter in self._local_names:
                if parameter in sharing:
                    global_name = sharing[parameter]
                else:
                    global_name = f"{name}.{parameter}"
                    index_by_name[global_name] = len(names)
                    names.append(global_name)
                indices.append(index_by_name[global_name])
            self._indices[name] = np.asarray(indices, dtype=np.int64)
        self._names = tuple(names)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return global parameter names in their optimization vector order."""
        return self._names

    @property
    def model_parameter_names(self) -> tuple[str, ...]:
        """Return local names in model argument order."""
        return self._local_names

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return datasets in residual-concatenation order."""
        return self._dataset_names

    @property
    def parameter_indices(self) -> Mapping[str, _Index]:
        """Return independent local-to-global index copies for each dataset."""
        return MappingProxyType(
            {name: indices.copy() for name, indices in self._indices.items()}
        )

    def _predictions(self, name: str, theta: _Array) -> _Array:
        """Evaluate one dataset's local model coordinates."""
        data = self._data[name]
        return _evaluate(self._model, data.x, theta[self._indices[name]], data.y.size)

    def _residuals(self, theta: _Array) -> _Array:
        """Concatenate whitened y-f residuals in deterministic dataset order."""
        pieces: list[_Array] = []
        for name in self._dataset_names:
            data = self._data[name]
            with np.errstate(over="ignore", invalid="ignore"):
                residual = _computed(
                    data.y - self._predictions(name, theta), "Global fit residuals"
                )
            pieces.append(_whiten(residual, data.noise))
        return np.concatenate(pieces)

    def _analytic_jacobian(
        self, theta: _Array, jac: Callable[..., ArrayLike]
    ) -> _Array:
        """Map weighted local derivatives into global columns, adding alias ties."""
        pieces: list[_Array] = []
        for name in self._dataset_names:
            data = self._data[name]
            indices = self._indices[name]
            local = -_whiten(
                _jacobian(self._model, data.x, theta[indices], jac), data.noise
            )
            block = np.zeros((data.y.size, len(self._names)))
            for local_index, global_index in enumerate(indices):
                block[:, global_index] += local[:, local_index]
            pieces.append(block)
        return _computed(np.concatenate(pieces), "Global parameter Jacobian")

    def fit(
        self,
        p0: ArrayLike | Mapping[str, float] | None = None,
        bounds: tuple[Any, Any] | Mapping[str, tuple[float, float]] = (-np.inf, np.inf),
        *,
        absolute_sigma: bool = False,
        jac: _Jacobian = "3-point",
        rank_tolerance: float | None = None,
        **kwargs: Any,
    ) -> _GlobalFitResult:
        """Optimize one global parameter vector and estimate its joint covariance.

        Parameters
        ----------
        p0 : array_like or mapping, optional
            Vector in parameter_names order or complete mapping from global names
            to values. Defaults to ones for unbounded variables and feasible
            midpoints/interior points for bounded variables.
        bounds : tuple or mapping
            Scalar/vector (lower, upper) or global-name-to-(lower, upper) map.
            Omitted names are unbounded. Equal limits denote exactly fixed
            parameters; they are removed from the optimizer vector.
        absolute_sigma : bool
            True uses known observation variances. False scales covariance by
            weighted SSE/(N-p), counting each estimated shared variable once.
        jac : {'2-point', '3-point'} or callable
            Finite differences obey optimization bounds. A callable returns the
            local model Jacobian, shape (n_dataset, n_model_parameters), from
            jac(x, *local_parameters). Shared/tied columns are assembled and
            whitened automatically. Complex-step differentiation is unsupported.
        rank_tolerance : float, optional
            Relative singular-value cutoff on column-normalized Jacobians.
            Default is max(N,p)*eps for analytic derivatives and at least 1e-7
            for numerical derivatives. Supply an analytic jac to resolve very
            close directions; rank deficiency never produces a finite pseudoinverse.
        **kwargs
            Supported scipy.optimize.least_squares options. Only loss='linear'
            is allowed, preserving Gaussian least-squares covariance semantics.
            Array x_scale, diff_step, and jac_sparsity refer to non-fixed variables
            in global order. Solver args/kwargs overrides are not supported.

        Returns
        -------
        result : typed dataclass
            popt, pcov, parameter_names, datasets, weighted_residuals, jacobian,
            active_mask, solver, and diagnostics. Each datasets[name] contains
            parameters, local popt, predictions, and raw/weighted residuals.
            result.success reflects solver convergence, not identifiability.

        Raises
        ------
        ValueError
            Invalid guesses, bounds, solver options, uncertainties, or Jacobians.
        FloatingPointError
            Weighted residuals or numerical Jacobian calculations are non-finite.

        Notes
        -----
        For an interior full-rank global Jacobian J_r, covariance is
        inv(J_r.T @ J_r), multiplied by weighted SSE/(N-p) only for
        relative error scales. Equal bounds remove exactly fixed variables.

        Optimization internally divides residuals/Jacobians by a common initial
        residual magnitude to avoid dependence on arbitrary common error units.
        Returned residuals, Jacobian, solver cost and optimality are restored to
        the original weighted units. SciPy gtol applies to the normalized
        objective; ftol and xtol retain their relative meanings. This improves
        common-scale invariance, not global-optimum guarantees.

        Covariance comes from the actual global residual Jacobian. With active
        bounds it is conditional on active values: active rows/columns are NaN,
        and the free block uses N-p_free scaling. diagnostics.dof remains N-p
        for all optimized variables; covariance_dof explicitly reports N-p_free.
        Failed convergence or a rank-deficient optimized Jacobian gives all-NaN
        covariance. Full rank does not prove structural or global identifiability.
        When all parameters are fixed, success indicates model evaluation only:
        solver.status is None and nfev/njev are zero; SciPy options are not run.
        Results can still be inspected when success=False. Ordinary nonlinear
        Wald uncertainty is only a local approximation.

        Examples
        --------
        >>> group = GlobalFitGroup(lambda x, a: a*x, {
        ...     'run': {'x': [1., 2., 3.], 'y': [2.1, 3.9, 6.0]},
        ... }, {'a'})
        >>> result = group.fit({'a': 1.}, bounds={'a': (0., 5.)})
        >>> abs(result.datasets['run'].parameters['a']-2) < 0.02
        True
        """
        absolute_sigma = _boolean(absolute_sigma, "absolute_sigma")
        n = sum(data.y.size for data in self._data.values())
        p = len(self._names)
        if not callable(jac) and jac not in ("2-point", "3-point"):
            raise ValueError(
                "jac must be '2-point', '3-point', or a local model Jacobian."
            )
        allowed = set(inspect.signature(optimize.least_squares).parameters) - {
            "fun",
            "x0",
            "bounds",
            "jac",
            "args",
            "kwargs",
        }
        if set(kwargs) - allowed:
            raise ValueError(
                f"Unsupported least_squares options: {sorted(set(kwargs) - allowed)}."
            )
        if kwargs.get("loss", "linear") != "linear":
            raise ValueError("Only loss='linear' is supported for Gaussian covariance.")
        kwargs.setdefault("x_scale", "jac")
        if rank_tolerance is None:
            rank_tolerance = max(n, p) * np.finfo(float).eps
            if not callable(jac):
                rank_tolerance = max(rank_tolerance, 1e-7)
        if not np.isfinite(rank_tolerance) or not 0 <= rank_tolerance < 1:
            raise ValueError("rank_tolerance must be finite and in [0, 1).")
        if isinstance(bounds, Mapping):
            if set(bounds) - set(self._names):
                raise ValueError("Bounds contain unknown global parameter names.")
            limits = [bounds.get(name, (-np.inf, np.inf)) for name in self._names]
            if any(len(limit) != 2 for limit in limits):
                raise ValueError("Each bound must be a (lower, upper) pair.")
            bounds = ([limit[0] for limit in limits], [limit[1] for limit in limits])
        # Normalize bounds independently of a guess; finite fixed coordinates
        # and infeasible infinities are checked again with the real initial point.
        if len(bounds) != 2:
            raise ValueError("bounds must be a (lower, upper) pair.")
        lower = np.broadcast_to(_real(bounds[0], "lower bounds"), (p,)).copy()
        upper = np.broadcast_to(_real(bounds[1], "upper bounds"), (p,)).copy()
        if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper):
            raise ValueError("Bounds must be ordered and contain no NaN.")
        if p0 is None:
            initial = np.ones(p)
            both = np.isfinite(lower) & np.isfinite(upper)
            initial[both] = lower[both] / 2 + upper[both] / 2
            only_lower = np.isfinite(lower) & ~np.isfinite(upper)
            only_upper = ~np.isfinite(lower) & np.isfinite(upper)
            with np.errstate(over="ignore"):
                initial[only_lower] = np.maximum(
                    1.0,
                    lower[only_lower]
                    + np.maximum(1.0, 0.1 * np.abs(lower[only_lower])),
                )
                initial[only_upper] = np.minimum(
                    1.0,
                    upper[only_upper]
                    - np.maximum(1.0, 0.1 * np.abs(upper[only_upper])),
                )
        elif isinstance(p0, Mapping):
            if set(p0) != set(self._names):
                raise ValueError(
                    "p0 mapping must contain exactly all global parameter names."
                )
            initial = _array([p0[name] for name in self._names], "p0").copy()
        else:
            initial = _vector(p0, "p0").copy()
        if initial.shape != (p,):
            raise ValueError(f"p0 must have {p} global parameters.")
        initial = _array(initial, "p0")
        lower, upper = _bounds((lower, upper), initial)
        fixed = lower == upper
        optimized = ~fixed
        p_optimized = int(np.count_nonzero(optimized))
        # Least-squares gtol is absolute in objective units. A common change
        # of relative sigma must not turn an arbitrary guess into convergence.
        initial_residual = self._residuals(initial)
        optimizer_scale = float(np.max(np.abs(initial_residual))) or 1.0

        def reconstruct(values: _Array) -> _Array:
            """Restore fixed coordinates without mutating initial guesses."""
            full = initial.copy()
            full[optimized] = values
            return full

        def residual_function(values: _Array) -> _Array:
            """Evaluate the residuals of the actual global optimization problem."""
            with np.errstate(over="ignore", invalid="ignore"):
                return _computed(
                    self._residuals(reconstruct(values)) / optimizer_scale,
                    "Normalized solver residuals",
                )

        def jacobian_function(values: _Array) -> _Array:
            """Return the global model derivative on optimized coordinates."""
            assert callable(jac)
            with np.errstate(over="ignore", invalid="ignore"):
                return _computed(
                    self._analytic_jacobian(reconstruct(values), jac)[:, optimized]
                    / optimizer_scale,
                    "Normalized solver Jacobian",
                )

        full_jacobian = np.zeros((n, p))
        active = np.zeros(p, dtype=np.int64)
        active[fixed] = 2
        if p_optimized:
            solution = optimize.least_squares(
                residual_function,
                initial[optimized],
                bounds=(lower[optimized], upper[optimized]),
                jac=jacobian_function if callable(jac) else jac,
                **kwargs,
            )
            theta = reconstruct(np.asarray(solution.x, dtype=float))
            if sparse.issparse(solution.jac):
                jacobian = sparse.csr_array(solution.jac).toarray()
            elif isinstance(solution.jac, LinearOperator):
                jacobian = solution.jac.matmat(np.eye(p_optimized))
            else:
                jacobian = solution.jac
            with np.errstate(over="ignore", invalid="ignore"):
                full_jacobian[:, optimized] = _computed(
                    _array(jacobian, "Solver Jacobian") * optimizer_scale,
                    "Original-unit solver Jacobian",
                )
            active[optimized] = np.asarray(solution.active_mask, dtype=np.int64)
            solver = _SolverDiagnostics(
                bool(solution.success),
                int(solution.status),
                str(solution.message),
                int(solution.nfev),
                None if solution.njev is None else int(solution.njev),
                0.5 * _squared_norm(self._residuals(theta)),
                float(solution.optimality) * optimizer_scale * optimizer_scale,
            )
        else:
            theta = initial.copy()
            solver = _SolverDiagnostics(
                True,
                None,
                "All parameters are fixed; no optimization performed.",
                0,
                0,
                0.5 * _squared_norm(self._residuals(theta)),
                0.0,
            )
        weighted_residual = self._residuals(theta)
        rank, condition = _rank(full_jacobian[:, optimized], rank_tolerance)
        free = active == 0
        free_rank, _ = _rank(full_jacobian[:, free], rank_tolerance)
        p_free = int(np.count_nonzero(free))
        covariance = np.full((p, p), np.nan)
        kind: Literal["ordinary", "conditional", "undefined"] = "undefined"
        if not solver.success:
            warnings.warn(
                "Global optimizer did not converge; covariance is undefined.",
                UserWarning,
                stacklevel=2,
            )
        elif rank < p_optimized:
            warnings.warn(
                "Global Jacobian is rank deficient; local parameters are not "
                "identified and covariance is undefined.",
                UserWarning,
                stacklevel=2,
            )
        else:
            kind = "conditional" if np.any(active) else "ordinary"
            if np.any(active):
                warnings.warn(
                    "Bound-active/fixed rows and columns are NaN; "
                    "free covariance is conditional.",
                    UserWarning,
                    stacklevel=2,
                )
            if p_free:
                covariance[np.ix_(free, free)] = _local_covariance(
                    full_jacobian[:, free], weighted_residual, absolute_sigma
                )
                if not np.isfinite(covariance[np.ix_(free, free)]).all():
                    kind = "undefined"
            if np.isfinite(condition) and condition > 1e6:
                warnings.warn(
                    "Global Jacobian is ill-conditioned; "
                    "local uncertainty may be unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
        results: dict[str, _DatasetResult] = {}
        raw: list[_Array] = []
        for name in self._dataset_names:
            data = self._data[name]
            local = theta[self._indices[name]].copy()
            predicted = self._predictions(name, theta).copy()
            residual = _computed(data.y - predicted, "Final residuals")
            raw.append(residual)
            parameters = MappingProxyType(
                {
                    parameter: float(value)
                    for parameter, value in zip(self._local_names, local, strict=True)
                }
            )
            results[name] = _DatasetResult(
                parameters,
                local,
                predicted,
                residual,
                _whiten(residual, data.noise).copy(),
            )
        raw_residual = np.concatenate(raw)
        rss, wss = _squared_norm(raw_residual), _squared_norm(weighted_residual)
        dof = n - p_optimized
        chi = (
            wss
            if absolute_sigma or any(data.weighted for data in self._data.values())
            else float("nan")
        )
        diagnostics = _FitDiagnostics(
            n,
            p,
            p_optimized,
            dof,
            rss,
            float(float(_norms(raw_residual, axis=0)) / np.sqrt(n)),
            wss,
            chi,
            chi / dof if dof > 0 else float("nan"),
            rank,
            free_rank,
            condition,
            rank == p_optimized,
            n - p_free,
            kind,
        )
        return _GlobalFitResult(
            theta.copy(),
            covariance,
            self._names,
            MappingProxyType(results),
            full_jacobian.copy(),
            weighted_residual.copy(),
            active.copy(),
            solver,
            diagnostics,
        )


@dataclass(frozen=True, slots=True)
class _ODRDiagnostics:
    """Expose backend termination and scaling without conflating their meanings."""

    success: bool
    identifiable: bool
    info: int
    stopreason: tuple[str, ...]
    scipy_version: str
    method: Literal["odr", "ols"]
    n_observations: int
    n_parameters: int
    dof: int | None
    rank: int | None
    residual_variance: float
    weighted_sse: float
    sum_square_x: float
    sum_square_y: float
    inverse_condition_number: float
    delta: _Array
    eps: _Array
    adjusted_x: _Array
    fitted_y: _Array
    unscaled_covariance: _Array
    backend_standard_errors: _Array
    standard_errors: _Array
    covariance_kind: Literal["absolute", "scaled", "undefined"]


def _odr_standard_deviations(
    value: ArrayLike | None, n: int, name: str
) -> _Array | None:
    """Require positive independent standard deviations usable as ODR weights."""
    if value is None:
        return None
    supplied = _array(value, name)
    if supplied.ndim == 0:
        supplied = np.full(n, float(supplied))
    if supplied.shape != (n,) or np.any(supplied <= 0):
        raise ValueError(
            f"{name} must be a positive scalar or one standard deviation "
            "per observation; covariance matrices are unsupported."
        )
    with np.errstate(over="ignore", divide="ignore", under="ignore", invalid="ignore"):
        squared = supplied**2
        weights = 1 / squared
    if (
        np.any(~np.isfinite(squared))
        or np.any(~np.isfinite(weights))
        or np.any(weights == 0)
    ):
        raise FloatingPointError(
            f"{name} produces unrepresentable float64 ODR weights; "
            "rescale measurements."
        )
    return supplied.copy()


def _odr_parameter_rank(
    func: _Model,
    x: _Array,
    beta: _Array,
    sx: _Array | None,
    sy: _Array | None,
) -> int:
    """Assess local parameter rank after eliminating scalar latent coordinates."""
    derivative = _jacobian(func, x, beta)
    error_y = np.ones(x.size) if sy is None else sy
    if sx is not None:
        # ODR assumes pointwise responses. Perturb coordinates independently in
        # one vector evaluation to compute the diagonal coordinate derivative.
        step = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x))
        previous: _Array | None = None
        stabilized = False
        for _ in range(40):
            try:
                plus, minus = x + step, x - step
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    coordinate_jac = (
                        _evaluate(func, plus, beta) - _evaluate(func, minus, beta)
                    ) / (plus - minus)
                if np.isfinite(coordinate_jac).all():
                    if previous is not None:
                        scale = np.maximum(
                            np.maximum(np.abs(coordinate_jac), np.abs(previous)),
                            np.finfo(float).tiny,
                        )
                        if np.all(
                            np.abs(coordinate_jac / scale - previous / scale) <= 1e-6
                        ):
                            stabilized = True
                            break
                    previous = coordinate_jac
                else:
                    previous = None
            except ValueError:
                previous = None
            step *= 0.5
        if not stabilized:
            raise ValueError(
                "Coordinate Jacobian did not stabilize; "
                "local ODR covariance cannot be validated."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            error_y = _computed(
                np.hypot(error_y, sx * coordinate_jac), "Effective ODR noise"
            )
    rank, _ = _rank(
        derivative / error_y[:, None],
        max(1e-7, max(derivative.shape) * np.finfo(float).eps),
    )
    return rank


@overload
def eiv_curve_fit(
    func: _Model,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    *,
    sx: ArrayLike | None = None,
    sy: ArrayLike | None = None,
    absolute_sigma: bool = False,
    full_output: Literal[False] = False,
    **kwargs: Any,
) -> tuple[_Array, _Array]: ...


@overload
def eiv_curve_fit(
    func: _Model,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    *,
    sx: ArrayLike | None = None,
    sy: ArrayLike | None = None,
    absolute_sigma: bool = False,
    full_output: Literal[True],
    **kwargs: Any,
) -> tuple[_Array, _Array, _ODRDiagnostics]: ...


@overload
def eiv_curve_fit(
    func: _Model,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    *,
    sx: ArrayLike | None = None,
    sy: ArrayLike | None = None,
    absolute_sigma: bool = False,
    full_output: bool,
    **kwargs: Any,
) -> tuple[_Array, _Array] | tuple[_Array, _Array, _ODRDiagnostics]: ...


def eiv_curve_fit(
    func: _Model,
    xdata: ArrayLike,
    ydata: ArrayLike,
    p0: ArrayLike | None = None,
    *,
    sx: ArrayLike | None = None,
    sy: ArrayLike | None = None,
    absolute_sigma: bool = False,
    full_output: bool = False,
    **kwargs: Any,
) -> tuple[_Array, _Array] | tuple[_Array, _Array, _ODRDiagnostics]:
    """Fit scalar errors-in-variables models through SciPy's ODRPACK backend.

    Parameters
    ----------
    func : callable
        Pointwise model written as func(x, *parameters). The adapter reverses
        arguments for ODR internally. Non-vectorized scalar callables and
        responses coupling observations are unsupported.
    xdata, ydata : array_like
        Finite, nonempty one-dimensional observations of identical length.
        Only one independent variable and one scalar response are supported.
    p0 : array_like, optional
        Finite initial parameters. When omitted, explicit positional parameter
        names are inferred and initialized to ones, as in curve_fit. Ambiguous
        or variadic signatures require p0; keyword-only options are not fitted.
    sx : array_like, optional
        Positive scalar/vector independent-variable standard deviations.
        None treats x as exact using ODR's least-squares mode, not unit x error.
        Zero elements are rejected; fixing selected x observations is unsupported.
    sy : array_like, optional
        Positive scalar/vector dependent-variable standard deviations. None
        means unit y weighting, or known unit y variance with absolute_sigma=True.
    absolute_sigma : bool
        True returns unscaled Output.cov_beta for known absolute error scales.
        False returns cov_beta*Output.res_var, matching ODR's sd_beta convention.
        One common unknown scale multiplies both axis covariance specifications.
    full_output : bool
        False returns (popt, pcov). True additionally returns a typed diagnostics
        dataclass with termination, rank, residual variance, weighted objective,
        coordinate corrections, and scaled/unscaled uncertainty information.
    **kwargs
        ODR controls maxit (default 200), sstol, partol, ndigit, and taufac only.
        maxit may be zero for diagnostic evaluation. sigma, jac, bounds, method,
        maxfev, robust losses, and other curve_fit options are rejected.

    Returns
    -------
    popt, pcov : ndarray
        Parameter vector and covariance of shapes (p,) and (p, p).
    diagnostics : typed dataclass, optional
        Returned with full_output=True. adjusted_x=xdata+delta and fitted_y is
        func(adjusted_x, *popt). eps is fitted_y-ydata, the ODR sign convention.
        standard_errors correspond to returned pcov; backend_standard_errors
        always follow cov_beta*res_var. dof=N-p only for identified solutions.

    Raises
    ------
    ValueError
        Invalid data, uncertainties, parameter guesses, controls, or N <= p.
    RuntimeError
        Missing/unsupported backend or nonconvergence with full_output=False.
    FloatingPointError
        Unrepresentable weights or non-finite fitted parameters/results.

    Notes
    -----
    Supports SciPy >=1.12,<1.19. scipy.odr is deprecated in 1.17 and removed in
    1.19. Calls on deprecated versions emit a FutureWarning; the backend import
    deprecation is translated into that explicit compatibility warning. No
    replacement dependency is installed. This is not a full curve_fit substitute:
    bounds, correlated axis errors, multiple predictors, fixed parameters, and
    analytic ODR derivatives are outside this interface.

    With full_output=True, nonconvergence warns and returns current estimates,
    all-NaN pcov, and success=False. Rank deficiency similarly warns and returns
    NaN covariance even when the optimizer converged. Full rank is a local
    numerical diagnostic, not proof of structural identifiability. A numerical
    relative rank cutoff of at least 1e-7 avoids finite uncertainties fabricated
    by finite-difference noise in dependent parameter columns.

    ODR covariance is a local Gauss-Newton approximation. For independent errors
    the weighted objective is sum((delta/sx)**2)+sum((eps/sy)**2); the first term
    is zero in exact-x mode. For identified fits res_var=objective/(N-p). Zero
    residual variance makes scaled covariance degenerate and warns. At least
    one residual degree of freedom is required, including absolute-scale fits.
    Outputs are copies; inputs are not mutated.

    Examples
    --------
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', FutureWarning)
    ...     popt, pcov = eiv_curve_fit(lambda x, slope: slope*x,
    ...         [1., 2., 3., 4.], [2.1, 3.9, 6.1, 7.9], [2.],
    ...         sx=0.05, sy=0.1, absolute_sigma=True)
    >>> abs(popt[0]-2) < 0.02 and pcov.shape == (1, 1)
    True
    """
    if not callable(func):
        raise ValueError("func must be callable.")
    absolute_sigma = _boolean(absolute_sigma, "absolute_sigma")
    full_output = _boolean(full_output, "full_output")
    x = _array(xdata, "xdata").copy()
    y = _array(ydata, "ydata").copy()
    if x.ndim != 1 or y.ndim != 1 or not x.size or x.shape != y.shape:
        raise ValueError(
            "xdata and ydata must be nonempty one-dimensional arrays "
            "of identical shape."
        )
    if p0 is None:
        try:
            names = _parameter_names(func, None)
        except ValueError as error:
            raise ValueError(
                "Cannot infer parameter count; supply p0 explicitly."
            ) from error
        beta0 = np.ones(len(names))
    else:
        beta0 = _vector(p0, "p0").copy()
    if x.size <= beta0.size:
        raise ValueError(
            "ODR inference requires N > p and positive residual degrees of freedom."
        )
    try:
        _evaluate(func, x, beta0)
    except TypeError as error:
        raise ValueError("Model signature and p0 are incompatible.") from error
    error_x = _odr_standard_deviations(sx, x.size, "sx")
    error_y = _odr_standard_deviations(sy, x.size, "sy")
    allowed = {"maxit", "sstol", "partol", "ndigit", "taufac"}
    if set(kwargs) - allowed:
        raise ValueError(
            f"Unsupported ODR options: {sorted(set(kwargs) - allowed)}; "
            f"supported controls are {sorted(allowed)}."
        )
    controls = dict(kwargs)
    controls["maxit"] = _count(controls.get("maxit", 200), "maxit", 0)
    if "ndigit" in controls:
        controls["ndigit"] = _count(controls["ndigit"], "ndigit", 0)
    for name in ("sstol", "partol", "taufac"):
        if name in controls:
            value = float(controls[name])
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be positive, finite, and at most one.")
            controls[name] = value
    version = tuple(int(part) for part in scipy.__version__.split(".")[:2])
    if not (1, 12) <= version < (1, 19):
        raise RuntimeError(
            "eiv_curve_fit requires scipy.odr in SciPy >=1.12,<1.19; "
            "no replacement backend is configured."
        )
    if version >= (1, 17):
        warnings.warn(
            "scipy.odr is deprecated in SciPy 1.17 and removed in 1.19; "
            "eiv_curve_fit currently supports SciPy >=1.12,<1.19.",
            FutureWarning,
            stacklevel=2,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="`scipy.odr` is deprecated.*", category=DeprecationWarning
        )
        try:
            from scipy import odr
        except ImportError as error:
            raise RuntimeError(
                "The scipy.odr backend is unavailable; "
                "install a supported SciPy version."
            ) from error

    def parameter_first(parameters: _Array, coordinates: _Array) -> _Array:
        """Adapt the caller's ordinary model to the parameter-first ODR API."""
        return _evaluate(
            func,
            np.asarray(coordinates, dtype=float),
            _array(parameters, "ODR parameters"),
        )

    method: Literal["odr", "ols"] = "ols" if error_x is None else "odr"
    data = odr.RealData(x, y, sx=error_x, sy=error_y)
    runner = odr.ODR(data, odr.Model(parameter_first), beta0=beta0, **controls)
    runner.set_job(fit_type=2 if method == "ols" else 0)
    try:
        output = runner.run()
    except (odr.OdrError, odr.OdrStop) as error:
        raise RuntimeError(f"ODR backend failed: {error}") from error
    info = int(output.info)
    converged = info in (1, 2, 3, 11, 12, 13)
    reasons = tuple(str(reason) for reason in output.stopreason)
    if not converged and not full_output:
        raise RuntimeError(f"ODR did not converge (info={info}): {'; '.join(reasons)}")
    popt = _computed(output.beta, "ODR fitted parameters").copy()
    delta = _computed(output.delta, "ODR x corrections").copy()
    eps = _computed(output.eps, "ODR y corrections").copy()
    adjusted_x = _computed(x + delta, "ODR adjusted coordinates")
    predicted = _evaluate(func, adjusted_x, popt)
    raw_covariance = _real(output.cov_beta, "ODR covariance").copy()
    residual_variance = float(output.res_var)
    covariance = np.full((popt.size, popt.size), np.nan)
    rank: int | None = None
    kind: Literal["absolute", "scaled", "undefined"] = "undefined"
    if converged:
        try:
            rank = _odr_parameter_rank(func, adjusted_x, popt, error_x, error_y)
            validated_covariance, _, _ = _covariance(raw_covariance, popt.size)
        except (ValueError, FloatingPointError, linalg.LinAlgError) as error:
            warnings.warn(
                f"ODR local covariance could not be validated: {error}",
                UserWarning,
                stacklevel=2,
            )
        else:
            if rank < popt.size or info >= 10:
                warnings.warn(
                    "ODR parameter Jacobian is rank deficient; "
                    "covariance is undefined.",
                    UserWarning,
                    stacklevel=2,
                )
            elif not np.isfinite(residual_variance) or residual_variance < 0:
                warnings.warn(
                    "ODR reported invalid residual variance; covariance is undefined.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    covariance = (
                        validated_covariance
                        if absolute_sigma
                        else validated_covariance * residual_variance
                    )
                _computed(covariance, "ODR returned covariance")
                if (
                    np.any(
                        (np.diag(validated_covariance) > 0) & (np.diag(covariance) == 0)
                    )
                    and residual_variance > 0
                ):
                    raise FloatingPointError(
                        "ODR covariance underflows float64; rescale parameters."
                    )
                kind = "absolute" if absolute_sigma else "scaled"
                if not absolute_sigma and residual_variance == 0:
                    warnings.warn(
                        "Zero residual variance: scaled ODR covariance is degenerate; "
                        "ordinary Wald inference is unreliable.",
                        UserWarning,
                        stacklevel=2,
                    )
                if float(output.inv_condnum) < np.sqrt(np.finfo(float).eps):
                    warnings.warn(
                        "ODR reports ill-conditioning; "
                        "local covariance may be unreliable.",
                        UserWarning,
                        stacklevel=2,
                    )
    else:
        warnings.warn(
            f"ODR did not converge (info={info}); "
            "returning current estimates with undefined covariance.",
            UserWarning,
            stacklevel=2,
        )
    diagnostics = _ODRDiagnostics(
        converged,
        rank == popt.size and info < 10,
        info,
        reasons,
        scipy.__version__,
        method,
        x.size,
        popt.size,
        x.size - popt.size if rank == popt.size and info < 10 else None,
        rank,
        residual_variance,
        float(output.sum_square),
        float(output.sum_square_delta),
        float(output.sum_square_eps),
        float(output.inv_condnum),
        delta,
        eps,
        adjusted_x.copy(),
        predicted.copy(),
        raw_covariance,
        _real(output.sd_beta, "ODR sd_beta").copy(),
        np.sqrt(np.diag(covariance)),
        kind,
    )
    return (popt, covariance, diagnostics) if full_output else (popt, covariance)
