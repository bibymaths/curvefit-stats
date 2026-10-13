# Scientific numerical stress validation

This verification phase preserves all ten public signatures and the two-module
architecture. It adds 25 reproducible stress cases in `tests/test_stress.py`;
existing analytical/reference tests remain part of the full validation.

## Corrected software defect

Normalizing large, nearly equal values before subtracting their mean introduced
rounding into otherwise representable differences. In a seeded 20,000-draw
experiment, the derived output `1e16 + Normal(0, 1)` had represented sample
standard deviation 1.1490996569058207. The implementation reported
1.3621062287418582, about 19% too large. The expected value here is the variance
of the represented outputs, including their unavoidable float64 quantization,
rather than the theoretical standard deviation of an unquantized normal.

Monte Carlo standard deviations and residual-bootstrap covariance now subtract
an observed origin before scaling and centering. If subtraction would overflow
for finite values with opposite extreme signs, scaling precedes subtraction.
Regression tests use independently centered represented samples. The bootstrap
regression supplies prescribed successful fits to isolate aggregation from
optimizer errors. Public signatures, sampling, and covariance conventions are
unchanged.

## Benchmarks and independent references

| Feature | Reference and stress coverage |
| --- | --- |
| `generate_error_bands` | Analytical linear/nonlinear Jacobians; response-unit invariance; independent Gaussian and Student-t quantiles; pointwise confidence/prediction coverage experiment in existing reference tests. |
| `audit_fit` | Normalized Gaussian density, weighted normal equations, likelihood scale parameter counting; heteroscedastic response/uncertainty rescaling by `1e-100` through `1e100`. |
| `fit_summary` | Existing incomplete-beta Student-t tail and nonlinear Wald references; additional exponential, Michaelis–Menten, and logistic benchmark summaries. |
| `profile_likelihood_errors` | Independently eliminate amplitude and verify likelihood-ratio increase of one at each shape endpoint for all three nonlinear models. Existing tests also cover unknown-scale likelihoods, bounds, and failed nuisance fits. |
| `bootstrap_curve_fit` | 1,500 seeded residual resamples; empirical residual variance with divisor N and independent linear normal equations; large-offset covariance regression. Existing tests cover failed attempts and reproducibility. |
| `collinearity_diagnostics` | Independent inverse-correlation VIF calculations, strongly correlated columns, near-singularity, and parameter units differing by 12 orders of magnitude. |
| `robust_bounded_pcov` | Known-noise conditional intercept covariance with active lower/upper slope bounds; weighted normal equations across response scales. Existing tests cover rank deficiency and undefined residual scaling. |
| `propagate_uncertainty` | 20,000 seeded Gaussian draws; derived ratio Delta approximation with sampling-error tolerance; represented-value variance at offsets up to `1e16`. Existing references compare ratio quantiles against independently integrated distributions. |
| `GlobalFitGroup` | Unequal dataset lengths and different grids; shared shape with separate amplitudes for exponential decay, Michaelis–Menten, and logistic growth; independent amplitude elimination; analytic Jacobians across parameter/response scales. |
| `eiv_curve_fit` | Independent joint least-squares optimization over parameters and latent x coordinates, including the parameter block of the joint covariance; heteroscedastic y uncertainty and errors in both axes across response units. Existing references include Deming regression, negligible x uncertainty, nonlinear models, and iteration limits. |

Near-singular Jacobians use column separations `1e-3`, `1e-9`, and zero.
The latter two correctly produce undefined covariance under the documented
numerical-Jacobian rank threshold. Forced global optimization failure likewise
returns undefined covariance rather than a fabricated estimate. Invalid zero,
negative, infinite, and NaN uncertainties have separate rejection tests. Existing
edge tests retain coverage for zero residual variance, insufficient degrees of
freedom, singular covariance, and numerical overflow/underflow.

Covariance comparisons use normalized parameter units with an absolute tolerance
for analytically zero cross terms. Relative tolerances otherwise scale with the
reference magnitude. Iterative ODR comparisons allow `3e-6` parameter and `2e-5`
covariance relative error; independent profile endpoints allow `3e-5` relative
and `3e-6` absolute error in their unit likelihood-ratio increase. Bootstrap
variance tolerance is five approximate Monte Carlo standard errors,
`5*sqrt(2/(B-1))`. Gaussian-derived standard deviation comparisons use
`5/sqrt(2*(M-1))` plus a stated small allowance for the nonlinear ratio
approximation. These tolerances do not demand exact stochastic agreement.
Exact aggregation regressions hold the represented draws fixed to test numerical
arithmetic, rather than sampling accuracy.

## Scientific and backend limitations

- Local covariance, Delta bands, and nonlinear Wald inference are local
  approximations; optimizer success does not prove a unique global minimum.
  Numerical rank thresholds cannot establish structural identifiability.
- Confidence/prediction bands are pointwise. Residual-based prediction variance
  assumes independent homoscedastic observations; heteroscedastic or EIV
  predictions require an appropriate explicit observation-error model.
- Residual bootstrap assumes exchangeable unweighted residuals. Its covariance
  uses the empirical residual distribution, not a silently applied residual
  degrees-of-freedom correction.
- Bound-active covariance is conditional on active parameters. It cannot provide
  their one-sided sampling uncertainty, and ordinary interior Wilks/Wald
  approximations may fail at boundaries.
- Gaussian parameter sampling can violate bounds. Ratios with a Gaussian
  denominator can lack population moments even when all sampled outputs are
  finite. Large offsets still quantize outputs in float64; corrected arithmetic
  cannot recover precision already lost in the model's returned values.
- Finite differences, extreme dynamic ranges, and native optimizers remain
  sensitive to parameterization. Analytic derivatives and rescaling are advisable;
  values or covariance outside float64 range are rejected or marked undefined.
- ODR covariance describes errors in both axes and is not interchangeable with
  vertical-residual curve-fit inference. The supported backend remains
  SciPy `>=1.12,<1.19`; SciPy 1.17+ emits the documented deprecation warning,
  and bounds/full constrained curve-fit compatibility remain unsupported.

No unresolved software defect was reproduced in these benchmark cases after the
aggregation correction. This is finite benchmark evidence, not a guarantee for
arbitrary user models.

## Executed final checks

All commands ran from `/workspace/curvefit-stats`.

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests -W error --cov=curvefit_stats --cov-report=term-missing --cov-fail-under=90
PYTHONPATH=src /workspace/setup-checks/curvefit-stats/compat-venv/bin/python -m pytest -q tests -W error
PYTHONPATH=src .venv/bin/python -m pytest -q --doctest-modules src/curvefit_stats
.venv/bin/ruff check --isolated --select E,F,I,B,UP,D --config 'lint.pydocstyle.convention="numpy"' src tests
.venv/bin/ruff format --isolated --check src tests
.venv/bin/mypy --strict src tests
git diff --check
```

The main environment uses Python 3.12.14, NumPy 2.5.4, and SciPy 1.18.1.
The lower-version environment uses NumPy 1.26.4 and SciPy 1.12.0.
The main and lower-version suites each pass 353 tests; main line coverage is
96.66%. Eleven doctests pass with one expected bound-active covariance warning.
Ruff lint/format checks, strict mypy, and `git diff --check` pass.
