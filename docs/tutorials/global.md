# Shared-parameter experiments

Suppose two experiments measure the same first-order decay rate but begin with
different concentrations. Joint fitting uses one rate degree of freedom and
one amplitude per experiment, even when their time grids differ.

$$
C_d(t)=A_d e^{-kt},\qquad
Q=\sum_d r_d^T C_d^{-1}r_d.
$$

## Fit once, project carefully

```python
import numpy as np
from curvefit_stats import GlobalFitGroup, generate_error_bands

def decay(time, amplitude, rate):
    return amplitude * np.exp(-rate * time)

rng = np.random.default_rng(52)
time_a = np.linspace(0, 4, 35)
time_b = np.linspace(0.1, 6, 52)
datasets = {
    "control": {"x": time_a, "y": decay(time_a, 3.0, 0.7) + rng.normal(0, 0.08, 35), "sigma": 0.08},
    "treated": {"x": time_b, "y": decay(time_b, 4.5, 0.7) + rng.normal(0, 0.1, 52), "sigma": 0.1},
}
group = GlobalFitGroup(decay, datasets, shared_parameters={"rate"})
result = group.fit(
    {"rate": 0.5, "control.amplitude": 2.5, "treated.amplitude": 4.0},
    bounds={"rate": (0.01, 3), "control.amplitude": (0, np.inf), "treated.amplitude": (0, np.inf)},
    absolute_sigma=True,
)
assert result.success and result.diagnostics.identifiable
assert result.parameter_names == ("rate", "control.amplitude", "treated.amplitude")
assert result.pcov.shape == (3, 3)

indices = group.parameter_indices["control"]
local_parameters = result.popt[indices]
local_covariance = result.pcov[np.ix_(indices, indices)]
grid = np.linspace(0, 5, 80)
lower, upper = generate_error_bands(
    decay, grid, local_parameters, local_covariance, reference="normal",
)
assert lower.shape == upper.shape == grid.shape
print(result.diagnostics)
```

The result includes per-dataset parameters, predictions, residuals, explicit
global/local mappings and solver diagnostics. Covariance is calculated from
the **joint Jacobian**, never by concatenating independent dataset covariances.
Mapping aliases can tie local arguments to a shared variable; tied derivative
contributions add into its one global column.

Known absolute measurement errors justify normal-reference bands here. For an
interior relative-weight fit, pass the **global** dof explicitly with no local
`y`; per-dataset N-p does not describe joint parameter uncertainty. There is one
common unknown variance multiplier, not separate unknown dataset scales.

For prediction intervals, supply the future observation variance from the same
global noise model. A per-dataset audit is not global AIC/BIC. Rank failure or
optimizer failure invalidates covariance; active bounds leave NaN active rows
and a conditional free block. Check these before extracting local bands.

See [joint-fit integration](../integration.md#global-fits) and
[`GlobalFitGroup`][curvefit_stats.estimators.GlobalFitGroup] for the exact mapping,
solver and result contracts.
