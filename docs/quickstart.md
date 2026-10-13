# Quickstart: concentration decay

Fit a first-order decay process, then quantify the uncertainty in its fitted
mean. Time is in hours, concentration in mg/L and the rate in h⁻¹:

$$
C(t)=A e^{-kt},\qquad t_{1/2}=\frac{\log 2}{k}.
$$

## Fit and summarize

The following complete example assumes IID Gaussian observation errors with
an **unknown common variance**. It runs after [installation](installation.md).

```python
import numpy as np
from scipy.optimize import curve_fit
from curvefit_stats import audit_fit, fit_summary, generate_error_bands

def decay(time, amplitude, rate):
    return amplitude * np.exp(-rate * time)

rng = np.random.default_rng(2026)
time = np.linspace(0, 5, 45)
concentration = decay(time, 3.0, 0.7) + rng.normal(0, 0.08, time.size)
popt, pcov = curve_fit(decay, time, concentration, p0=[2.5, 0.5])
print(fit_summary(decay, time, concentration, popt, pcov))
print(audit_fit(decay, time, concentration, popt))

grid = np.linspace(0, 5, 101)
lower, upper = generate_error_bands(
    decay, grid, popt, pcov, concentration, x_fit=time,
)
assert lower.shape == upper.shape == (101,)
assert np.all(lower <= upper)
```

The table's p-values test zero-valued parameters under approximate nonlinear
Student-t/Wald inference; they are not tests of model adequacy. The default
confidence bands use $N-p=43$ residual degrees of freedom from the original fit,
not the 101-point prediction grid.

## Predict a new measurement

Continue the same example:

```python
prediction_lower, prediction_upper = generate_error_bands(
    decay, grid, popt, pcov, concentration, x_fit=time,
    interval_type="prediction",
)
assert np.all(prediction_upper - prediction_lower >= upper - lower)
```

This adds independent future noise estimated as $\mathrm{RSS}/(N-p)$.
It is appropriate here because the example assumes homoscedastic IID errors.
Known or heteroscedastic future errors need an explicit `prediction_variance`.
Neither interval covers the entire curve simultaneously.

## If the measurement errors are known

Refit under the different noise convention before using a normal reference:

```python
popt_known, pcov_known = curve_fit(
    decay, time, concentration, p0=[2.5, 0.5],
    sigma=0.08, absolute_sigma=True,
)
known_lower, known_upper = generate_error_bands(
    decay, grid, popt_known, pcov_known, reference="normal",
    interval_type="prediction", prediction_variance=0.08**2,
)
assert np.isfinite(known_lower).all() and np.isfinite(known_upper).all()
```

Scalar/vector `sigma` is a **standard deviation**; `prediction_variance` is a
**variance**. Do not change the reference law without retaining the covariance's
noise interpretation. [Weighting conventions](user-guide.md#weighting-and-covariance-conventions)
explain the distinction between relative and absolute errors.

**Next:** [compare inference methods](tutorials/fitting.md),
[choose a workflow](workflows.md), or consult the
[source-generated API](reference/index.md).
