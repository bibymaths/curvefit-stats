# From a fit to uncertainty

Start with [the quickstart](../quickstart.md), then keep three different
questions separate: uncertainty in a parameter, uncertainty in a fitted mean,
and variation in a future measurement.

## Mean confidence versus observation prediction

At a chosen coordinate, the Delta approximation gives

$$
v_{\mathrm{mean}}(x)=j(x)Vj(x)^T,\qquad
v_{\mathrm{prediction}}(x)=v_{\mathrm{mean}}(x)+\sigma_{\mathrm{future}}^2(x).
$$

Here $V$ is parameter covariance and $j$ the model's parameter gradient.
The added future noise must be independent of parameter estimation.
This self-contained example uses **known** concentration uncertainty:

```python
import numpy as np
from scipy.optimize import curve_fit
from curvefit_stats import audit_fit, generate_error_bands, propagate_uncertainty

def decay(time, amplitude, rate):
    return amplitude * np.exp(-rate * time)

time = np.linspace(0, 5, 45)
sigma = 0.08
rng = np.random.default_rng(2026)
observed = decay(time, 3.0, 0.7) + rng.normal(0, sigma, time.size)
popt, pcov = curve_fit(
    decay, time, observed, p0=[2.5, 0.5], sigma=sigma, absolute_sigma=True,
)
grid = np.linspace(0, 5, 101)
mean_lo, mean_hi = generate_error_bands(
    decay, grid, popt, pcov, reference="normal",
)
pred_lo, pred_hi = generate_error_bands(
    decay, grid, popt, pcov, reference="normal",
    interval_type="prediction", prediction_variance=sigma**2,
)
assert np.all(pred_hi - pred_lo > mean_hi - mean_lo)
metrics = audit_fit(decay, time, observed, popt, sigma=sigma, absolute_sigma=True)
assert metrics["dof"] == 43

half_life = propagate_uncertainty(
    lambda amplitude, rate: np.log(2) / rate,
    popt, pcov, n_samples=20000, random_state=2026,
)
assert half_life["n_valid"] == 20000
print(half_life["lower"], half_life["upper"])
```

The normal reference matches the known-error covariance here. `fit_summary`
always uses approximate Student-t with N-p, so it is not an automatically
calibrated known-error table. Both bands remain pointwise, and linearization may
be poor under strong curvature, weak identification or constraints.

The half-life calculation samples a Gaussian parameter approximation. Its finite
Monte Carlo summary is not a proof of finite population moments: a Gaussian rate
can be arbitrarily near zero. Physical bounds and likelihood asymmetry may call
for a different uncertainty model.

## What changes at an active bound?

A nonnegative background can reach its lower constraint. Ordinary full-matrix
propagation must then give way to explicitly **conditional** inference:

```python
import warnings
from curvefit_stats import robust_bounded_pcov

def with_background(time, amplitude, rate, background):
    return amplitude * np.exp(-rate * time) + background

rng = np.random.default_rng(19)
observed_bound = with_background(time, 3.0, 0.7, -0.15)
observed_bound = observed_bound + rng.normal(0, 0.04, time.size)
bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
bounded, _ = curve_fit(with_background, time, observed_bound, [3, 0.7, 0.05], bounds=bounds)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", UserWarning)
    conditional = robust_bounded_pcov(
        with_background, time, observed_bound, bounded, bounds,
    )
assert any("Bound-active parameter rows" in str(item.message) for item in caught)
assert np.isnan(conditional[2]).all()

def fixed_background(time, amplitude, rate):
    return with_background(time, amplitude, rate, bounded[2])

free_lo, free_hi = generate_error_bands(
    fixed_background, grid, bounded[:2], conditional[:2, :2], dof=time.size - 2,
)
assert np.isfinite(free_lo).all() and np.isfinite(free_hi).all()
```

The synthetic true background conflicts deliberately with the imposed bound.
Computing a finite free covariance does not establish model adequacy. These
bands hold the selected background fixed; they do not quantify its one-sided
uncertainty or calibrate selection of the active set.

Read [conditional covariance](../user-guide.md#covariance-at-active-bounds),
[diagnostic assumptions](../user-guide.md#fit-diagnostics-and-information-criteria)
and the [API contracts](../reference/index.md) before adapting this workflow.
