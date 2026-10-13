"""Independent joint-regression references and GlobalFitGroup API tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from scipy import linalg

from curvefit_stats import estimators
from curvefit_stats.estimators import GlobalFitGroup

_Array = NDArray[np.float64]


def _line(x: _Array, slope: float, intercept: float) -> _Array:
    """Evaluate a common linear model."""
    return slope * x + intercept


def _line_jac(x: _Array, slope: float, intercept: float) -> _Array:
    """Return its symbolic Jacobian."""
    return np.column_stack((x, np.ones_like(x)))


def _linear_data(noise: str = "none") -> tuple[dict[str, Any], _Array, _Array, _Array]:
    """Build an independent three-column shared-slope design and block covariance."""
    x1, x2 = np.linspace(-1, 2, 13), np.linspace(0.2, 4, 19)
    design = np.zeros((32, 3))
    design[:13, 0] = x1
    design[13:, 0] = x2
    design[:13, 1] = 1
    design[13:, 2] = 1
    covariance = np.eye(32)
    sigma1: Any = None
    sigma2: Any = None
    if noise == "vector":
        sigma1, sigma2 = np.linspace(0.1, 0.2, 13), np.linspace(0.2, 0.3, 19)
        covariance = np.diag(np.r_[sigma1**2, sigma2**2])
    elif noise == "matrix":
        sigma1 = 0.02 * np.eye(13) + 0.01 * np.ones((13, 13))
        sigma2 = 0.03 * np.eye(19) + 0.004 * np.ones((19, 19))
        covariance = linalg.block_diag(sigma1, sigma2)
    elif noise == "mixed":
        sigma1, sigma2 = 0.2, None
        covariance[:13, :13] *= 0.04
    y = design @ np.array([2.0, 0.7, -1.2]) + np.random.default_rng(372).normal(
        0, 0.04, 32
    )
    # Reverse insertion order deliberately; parameter/residual order must still
    # be deterministic and agree with our independently assembled design.
    data = {
        "b": {"x": x2, "y": y[13:], "sigma": sigma2},
        "a": {"x": x1, "y": y[:13], "sigma": sigma1},
    }
    return data, design, y, covariance


@pytest.mark.parametrize("noise", ["none", "vector", "matrix", "mixed"])
@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("analytic", [True, False])
def test_joint_linear_reference(noise: str, absolute: bool, analytic: bool) -> None:
    """Compare parameters, covariance, Jacobian, and diagnostics to joint GLS."""
    data, design, y, observation_covariance = _linear_data(noise)
    information = design.T @ np.linalg.solve(observation_covariance, design)
    reference = np.linalg.solve(
        information, design.T @ np.linalg.solve(observation_covariance, y)
    )
    residual = y - design @ reference
    wss = residual @ np.linalg.solve(observation_covariance, residual)
    covariance = np.linalg.solve(information, np.eye(3))
    if not absolute:
        covariance *= wss / (32 - 3)
    group = GlobalFitGroup(_line, data, {"slope"})
    jac: Any = _line_jac if analytic else "3-point"
    result = group.fit(
        {"slope": 1.0, "a.intercept": 0.0, "b.intercept": 0.0},
        absolute_sigma=absolute,
        jac=jac,
    )
    assert result.success
    assert (
        group.parameter_names
        == result.parameter_names
        == ("slope", "a.intercept", "b.intercept")
    )
    assert group.dataset_names == ("a", "b")
    assert_array_equal(group.parameter_indices["a"], [0, 1])
    assert_array_equal(group.parameter_indices["b"], [0, 2])
    assert_allclose(result.popt, reference, rtol=1e-8, atol=1e-9)
    assert_allclose(result.pcov, covariance, rtol=1e-8, atol=1e-12)
    whitening = linalg.cholesky(observation_covariance, lower=True)
    expected_jac = -linalg.solve_triangular(whitening, design, lower=True)
    assert_allclose(result.jacobian, expected_jac, rtol=1e-8, atol=1e-9)
    assert_allclose(
        result.weighted_residuals,
        linalg.solve_triangular(whitening, residual, lower=True),
        atol=1e-8,
    )
    assert_allclose(result.solver.cost, 0.5 * wss)
    assert result.solver.nfev > 0
    assert result.solver.status is not None and result.solver.status > 0
    assert result.diagnostics.n_observations == 32
    assert result.diagnostics.n_parameters == result.diagnostics.rank == 3
    assert result.diagnostics.dof == result.diagnostics.covariance_dof == 29
    assert result.diagnostics.covariance_kind == "ordinary"
    assert result.diagnostics.identifiable
    assert_allclose(result.diagnostics.weighted_sse, wss)
    assert_allclose(result.diagnostics.rss, residual @ residual)
    if noise == "none" and not absolute:
        assert np.isnan(result.diagnostics.chi_square)
    else:
        assert_allclose(result.diagnostics.reduced_chi_square, wss / 29)
    for name, rows, intercept in [("a", slice(0, 13), 1), ("b", slice(13, None), 2)]:
        local = result.datasets[name]
        assert_allclose(local.popt, reference[[0, intercept]])
        assert_allclose(local.predictions, (design @ reference)[rows])
        assert_allclose(local.residuals, residual[rows])
        assert_allclose(local.parameters["slope"], result.popt[0])
        assert local.predictions.shape == local.residuals.shape == data[name]["y"].shape


def test_three_datasets_multiple_shared_nonlinear_parameters() -> None:
    """Recover shared rate/offset and independent amplitudes on unequal grids."""

    def decay(x: _Array, amplitude: float, rate: float, offset: float) -> _Array:
        """Evaluate an exponential with a background."""
        return amplitude * np.exp(-rate * x) + offset

    def jacobian(x: _Array, amplitude: float, rate: float, offset: float) -> _Array:
        """Return exact parameter derivatives."""
        z = np.exp(-rate * x)
        return np.column_stack((z, -amplitude * x * z, np.ones_like(x)))

    generator = np.random.default_rng(27)
    datasets: dict[str, Any] = {}
    for name, n, amplitude in [("c", 41, 4.0), ("b", 29, 3.0), ("a", 35, 2.0)]:
        x = np.linspace(0.1, 5 + amplitude / 2, n)
        datasets[name] = {
            "x": x,
            "y": decay(x, amplitude, 0.7, 0.3) + generator.normal(0, 0.01, n),
            "sigma": 0.01,
        }
    group = GlobalFitGroup(decay, datasets, {"rate", "offset"})
    assert group.parameter_names == (
        "rate",
        "offset",
        "a.amplitude",
        "b.amplitude",
        "c.amplitude",
    )
    result = group.fit([0.5, 0.2, 1.5, 2.5, 3.5], absolute_sigma=True, jac=jacobian)
    assert result.success
    assert_allclose(result.popt, [0.7, 0.3, 2.0, 3.0, 4.0], atol=0.02)
    blocks = []
    for index, name in enumerate(["a", "b", "c"]):
        x = datasets[name]["x"]
        amplitude = result.popt[index + 2]
        z = np.exp(-result.popt[0] * x)
        design = np.zeros((x.size, 5))
        design[:, 0] = -amplitude * x * z
        design[:, 1] = 1
        design[:, index + 2] = z
        blocks.append(design / 0.01)
        assert_array_equal(group.parameter_indices[name], [index + 2, 0, 1])
    design = np.concatenate(blocks)
    assert_allclose(
        result.pcov, np.linalg.solve(design.T @ design, np.eye(5)), rtol=1e-10
    )
    assert result.diagnostics.n_parameters == 5
    assert result.diagnostics.dof == 35 + 29 + 41 - 5
    # A genuine global covariance contains cross-dataset amplitude dependence.
    assert abs(result.pcov[2, 3]) > 1e-7


@pytest.mark.parametrize("shared", [frozenset(), frozenset({"slope", "intercept"})])
def test_independent_and_shared_only_groups(shared: frozenset[str]) -> None:
    """Exercise independent blocks and one fully shared parameter vector."""
    data, _, _, _ = _linear_data()
    if shared:
        for dataset in data.values():
            dataset["y"] = _line(dataset["x"], 2.0, 1.0)
    group = GlobalFitGroup(_line, data, shared)
    result = group.fit(absolute_sigma=True)
    assert result.success
    if shared:
        assert group.parameter_names == ("slope", "intercept")
        assert result.diagnostics.n_parameters == 2
        assert_allclose(result.popt, [2.0, 1.0], atol=1e-8)
        assert_array_equal(group.parameter_indices["a"], group.parameter_indices["b"])
    else:
        assert group.parameter_names == (
            "a.slope",
            "a.intercept",
            "b.slope",
            "b.intercept",
        )
        assert result.diagnostics.n_parameters == 4
        assert_allclose(result.pcov[:2, 2:], 0, atol=1e-14)


def test_alias_mapping_ties_and_keyword_only_names() -> None:
    """Explicit alias ties must sum derivative columns and optimize one variable."""

    def model(x: _Array, a: float, b: float, *, scale: float = 1.0) -> _Array:
        """Evaluate two explicitly tied linear coefficients."""
        return (a + b) * x * scale

    datasets = {
        "b": {"x": [1.0, 2.0, 3.0], "y": [4.0, 8.0, 12.0]},
        "a": {"x": [1.0, 2.0], "y": [4.0, 8.0]},
    }
    group = GlobalFitGroup(model, datasets, {"a": "level", "b": "level"})
    result = group.fit(
        {"level": 1.0}, absolute_sigma=True, jac=lambda x, a, b: np.column_stack((x, x))
    )
    assert group.model_parameter_names == ("a", "b")
    assert group.parameter_names == ("level",)
    assert result.diagnostics.n_parameters == 1
    assert result.diagnostics.dof == 4
    assert_allclose(result.popt, [2.0])
    assert_allclose(result.pcov, [[1 / (4 * (5 + 14))]])
    assert result.datasets["a"].parameters["a"] == result.datasets["b"].parameters["b"]


@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("fixed", [True, False])
def test_bounds_and_conditional_covariance(absolute: bool, fixed: bool) -> None:
    """Distinguish global fitted dof from free conditional covariance dof."""
    data, _, _, _ = _linear_data()
    group = GlobalFitGroup(_line, data, {"slope"})
    bounds = {"slope": (1.0, 1.0) if fixed else (0.0, 1.0)}
    with pytest.warns(UserWarning, match="conditional"):
        result = group.fit(
            [1.0 if fixed else 0.5, 0.0, 0.0],
            bounds=bounds,
            absolute_sigma=absolute,
            jac=_line_jac,
        )
    assert result.success and result.popt[0] <= 1.0
    assert result.active_mask[0] == (2 if fixed else 1)
    assert np.isnan(result.pcov[0]).all() and np.isnan(result.pcov[:, 0]).all()
    covariance = np.diag([1 / 13, 1 / 19])
    if not absolute:
        covariance *= result.diagnostics.weighted_sse / (32 - 2)
    assert_allclose(result.pcov[1:, 1:], covariance, rtol=1e-10)
    assert result.diagnostics.n_parameters == (2 if fixed else 3)
    assert result.diagnostics.dof == (30 if fixed else 29)
    assert result.diagnostics.covariance_dof == 30
    assert result.diagnostics.covariance_kind == "conditional"


@pytest.mark.parametrize("analytic", [False, True])
def test_rank_deficiency_is_not_hidden_by_pseudoinverse(analytic: bool) -> None:
    """Return undefined covariance while preserving optimizer status and estimates."""

    def model(x: _Array, a: float, b: float) -> _Array:
        """Evaluate deliberately unidentifiable coefficients."""
        return (a + b) * x

    group = GlobalFitGroup(
        model,
        {
            "a": {"x": [1.0, 2.0, 3.0], "y": [3.1, 5.9, 9.0]},
            "b": {"x": [2.0, 3.0, 4.0], "y": [6.0, 9.1, 11.9]},
        },
        {"a", "b"},
    )
    jac: Any = (lambda x, a, b: np.column_stack((x, x))) if analytic else "3-point"
    with pytest.warns(UserWarning, match="rank deficient"):
        result = group.fit([1.0, 1.0], jac=jac)
    assert result.success
    assert result.diagnostics.rank == 1
    assert not result.diagnostics.identifiable
    assert np.isnan(result.pcov).all()
    assert np.isinf(result.diagnostics.condition_number)


def test_solver_failure_not_convergence() -> None:
    """A real evaluation limit must not produce apparently valid covariance."""
    data, _, _, _ = _linear_data()
    group = GlobalFitGroup(_line, data, {"slope"})
    with pytest.warns(UserWarning, match="did not converge"):
        result = group.fit([100.0, -100.0, 100.0], max_nfev=1)
    assert not result.success and result.solver.status == 0
    assert np.isnan(result.pcov).all()
    assert result.diagnostics.covariance_kind == "undefined"


def test_fixed_all_parameters_and_nonpositive_dof() -> None:
    """Evaluate fixed models and keep undefined residual scaling explicit."""
    group = GlobalFitGroup(
        lambda x, a: a * x, {"a": {"x": [1.0, 2.0], "y": [2.1, 3.9]}}, {"a"}
    )
    with pytest.warns(UserWarning, match="conditional"):
        fixed = group.fit(bounds={"a": (2.0, 2.0)})
    assert fixed.success and fixed.diagnostics.n_parameters == 0
    assert fixed.solver.status is None and fixed.solver.nfev == 0
    assert fixed.diagnostics.dof == 2
    assert fixed.active_mask[0] == 2
    assert np.isnan(fixed.pcov).all()
    singleton = GlobalFitGroup(
        lambda x, a: a * x, {"a": {"x": [1.0], "y": [2.0]}}, {"a"}
    )
    with pytest.warns(UserWarning, match="degrees of freedom"):
        fitted = singleton.fit([1.0])
    assert fitted.diagnostics.dof == 0 and np.isnan(fitted.pcov).all()
    absolute = singleton.fit([1.0], absolute_sigma=True)
    assert_allclose(absolute.pcov, [[1.0]])


def test_ambiguous_names_need_explicit_mapping() -> None:
    """Handle variadic signatures only with explicit local parameter order."""

    def model(x: _Array, *parameters: float) -> _Array:
        """Evaluate a variadic line."""
        return parameters[0] * x + parameters[1]

    datasets = {"a": {"x": [1.0, 2.0, 3.0], "y": [3.0, 5.0, 7.0]}}
    with pytest.raises(ValueError, match="parameter_names"):
        GlobalFitGroup(model, datasets)
    group = GlobalFitGroup(model, datasets, {"a"}, parameter_names=["a", "b"])
    assert group.parameter_names == ("a", "a.b")
    result = group.fit([1.0, 0.0], absolute_sigma=True)
    assert_allclose(result.popt, [2.0, 1.0], atol=1e-8)


@pytest.mark.parametrize(
    "sharing", [{"missing"}, {"slope": "bad.alias"}, {"slope": ""}, ["slope"]]
)
def test_malformed_sharing(sharing: Any) -> None:
    """Reject unknown parameters and malformed or ambiguous sharing schemas."""
    data, _, _, _ = _linear_data()
    with pytest.raises(ValueError):
        GlobalFitGroup(_line, data, sharing)


@pytest.mark.parametrize("names", [[], ["a"], ["a", "a"], ["a", "not.valid"], "ab"])
def test_invalid_explicit_names(names: Any) -> None:
    """Require one unique identifier per fitted positional parameter."""
    with pytest.raises(ValueError):
        GlobalFitGroup(_line, {"a": {"x": [1.0], "y": [2.0]}}, parameter_names=names)


@pytest.mark.parametrize(
    "data",
    [
        {},
        {1: {"x": [1.0], "y": [2.0]}},
        {"a": {}},
        {"a": {"x": [1.0]}},
        {"a": {"x": None, "y": [1.0]}},
        {"a": {"x": [1.0], "y": [2.0], "weights": [1.0]}},
        {"a": {"x": [1.0], "y": [np.nan]}},
        {"a": {"x": [1.0], "y": [1.0, 2.0]}},
        {"a": {"x": [], "y": []}},
    ],
)
def test_invalid_dataset_schema(data: Any) -> None:
    """Reject missing, non-finite, empty, or inconsistent observations."""
    with pytest.raises(ValueError):
        GlobalFitGroup(_line, data)


@pytest.mark.parametrize(
    "sigma",
    [
        0.0,
        -1.0,
        [1.0],
        [1.0, np.inf],
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 2.0], [2.0, 1.0]],
        [[1.0, 0.1], [0.2, 1.0]],
    ],
)
def test_invalid_observation_uncertainty(sigma: Any) -> None:
    """Reject invalid standard deviations and singular or asymmetric covariance."""
    with pytest.raises(ValueError):
        GlobalFitGroup(_line, {"a": {"x": [1.0, 2.0], "y": [2.0, 4.0], "sigma": sigma}})


@pytest.mark.parametrize(
    "options",
    [
        {"p0": [1.0]},
        {"p0": {"slope": 1.0}},
        {"p0": [np.nan, 0.0, 0.0]},
        {"bounds": {"missing": (0.0, 1.0)}},
        {"bounds": ([2.0, 0.0, 0.0], [1.0, 1.0, 1.0])},
        {"bounds": ([np.nan, 0.0, 0.0], [3.0, 1.0, 1.0])},
        {"bounds": (0.0, 1.0), "p0": [2.0, 0.0, 0.0]},
        {"jac": "cs"},
        {"loss": "soft_l1"},
        {"rank_tolerance": -1},
        {"rank_tolerance": np.nan},
        {"absolute_sigma": "true"},
        {"bogus": 1},
    ],
)
def test_invalid_fit_configuration(options: dict[str, Any]) -> None:
    """Fail clearly rather than silently accepting incompatible statistical options."""
    data, _, _, _ = _linear_data()
    with pytest.raises(ValueError):
        GlobalFitGroup(_line, data, {"slope"}).fit(**options)


def test_inputs_mapping_and_results_are_independent() -> None:
    """Snapshot inputs and protect mapping metadata across repeated fits."""
    data, _, _, _ = _linear_data()
    initial = np.array([1.0, 0.0, 0.0])
    before = initial.copy()
    group = GlobalFitGroup(_line, data, {"slope"})
    indices = group.parameter_indices
    indices["a"][0] = 99
    assert group.parameter_indices["a"][0] == 0
    first = group.fit(initial, absolute_sigma=True)
    assert_array_equal(initial, before)
    data["a"]["y"][:] = 0
    second = group.fit(initial, absolute_sigma=True)
    assert_array_equal(first.popt, second.popt)
    first.popt[:] = 0
    assert np.any(second.popt != 0)
    assert isinstance(second.datasets, Mapping)
    with pytest.raises(TypeError):
        second.datasets["a"].parameters["slope"] = 0  # type: ignore[index]
    assert estimators.__all__ == ["GlobalFitGroup", "eiv_curve_fit"]
    assert callable(estimators.eiv_curve_fit)


@pytest.mark.parametrize("analytic", [True, False])
def test_parameter_units_preserve_global_rank_and_covariance(analytic: bool) -> None:
    """Recover shared parameters when predictor units differ by twenty orders."""
    data, design, y, _ = _linear_data()
    information = design.T @ design
    reference = np.linalg.solve(information, design.T @ y)
    covariance = np.linalg.solve(information, np.eye(3))
    for dataset in data.values():
        dataset["x"] = dataset["x"] * 1e20
    jac: Any = _line_jac if analytic else "3-point"
    result = GlobalFitGroup(_line, data, {"slope"}).fit(
        [0.0, 0.0, 0.0], jac=jac, absolute_sigma=True
    )
    units = np.array([1e20, 1.0, 1.0])
    assert result.success and result.diagnostics.rank == 3
    assert_allclose(result.popt * units, reference, atol=1e-8)
    assert_allclose(
        result.pcov * units[:, None] * units[None, :], covariance, rtol=1e-8
    )


def test_multidimensional_coordinates_and_sparse_solver_jacobian() -> None:
    """Support multiple predictors and assemble sparse numerical global derivatives."""
    from scipy import sparse

    def model(x: _Array, a: float, b: float) -> _Array:
        """Evaluate two predictors on a dataset's own grid."""
        return np.asarray(a * x[0] + b * x[1], dtype=float)

    x1 = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]])
    x2 = np.array([[0.1, 0.5, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 1.0, 3.0]])
    datasets = {
        "a": {"x": x1, "y": model(x1, 2.0, 1.0)},
        "b": {"x": x2, "y": model(x2, 2.0, 3.0)},
    }
    design = np.zeros((9, 3))
    design[:4, 0] = x1[0]
    design[4:, 0] = x2[0]
    design[:4, 1] = x1[1]
    design[4:, 2] = x2[1]
    result = GlobalFitGroup(model, datasets, {"a"}).fit(
        [1.0, 0.0, 0.0],
        absolute_sigma=True,
        jac_sparsity=sparse.csr_matrix(design != 0),
    )
    assert result.success
    assert_allclose(result.popt, [2.0, 1.0, 3.0], atol=1e-8)
    assert_allclose(
        result.pcov, np.linalg.solve(design.T @ design, np.eye(3)), rtol=1e-8
    )
    assert result.jacobian.shape == (9, 3)


def test_zero_jacobian_and_active_domain_boundary() -> None:
    """Distinguish flat models from a feasible model defined only inside bounds."""
    group = GlobalFitGroup(
        lambda x, a: np.ones_like(x), {"a": {"x": [1.0, 2.0], "y": [1.1, 0.9]}}, {"a"}
    )
    with pytest.warns(UserWarning, match="rank deficient"):
        result = group.fit([1.0])
    assert result.diagnostics.rank == 0 and np.isnan(result.pcov).all()

    def model(x: _Array, a: float) -> _Array:
        """Restrict the parameter to its valid physical domain."""
        if a < 0:
            raise ValueError("a must be nonnegative")
        return a * x

    constrained = GlobalFitGroup(
        model, {"a": {"x": [1.0, 2.0, 3.0], "y": [-1.0, -2.0, -3.0]}}, {"a"}
    )
    with pytest.warns(UserWarning, match="conditional"):
        result = constrained.fit([0.1], bounds=(0.0, np.inf), absolute_sigma=True)
    assert result.success and result.active_mask[0] == -1
    assert result.popt[0] >= 0
    assert np.isnan(result.pcov).all()


def test_indefinite_bounds_and_wrong_jacobian_shape() -> None:
    """Reject complex bounds and inconsistent local analytical Jacobians."""
    data, _, _, _ = _linear_data()
    group = GlobalFitGroup(_line, data, {"slope"})
    with pytest.raises(ValueError, match="real-valued"):
        group.fit(bounds=(0 + 1j, 5))
    with pytest.raises(ValueError, match="jac must have shape"):
        group.fit(jac=lambda x, a, b: np.zeros((x.size, 3)))
    invalid_bounds: Any = (0, 1, 2)
    with pytest.raises(ValueError, match="bounds"):
        group.fit(bounds=invalid_bounds)


def test_nonfinite_models_and_no_extra_public_features() -> None:
    """Reject non-finite model predictions and preserve the approved feature scope."""
    group = GlobalFitGroup(
        lambda x, a: np.full_like(x, np.inf),
        {"a": {"x": [1.0, 2.0], "y": [2.0, 4.0]}},
        {"a"},
    )
    with pytest.raises(ValueError, match="finite"):
        group.fit([1.0])
    assert estimators.__all__ == ["GlobalFitGroup", "eiv_curve_fit"]
