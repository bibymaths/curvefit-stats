
## Unreleased



### Other Changes

- Initial commit ([6be2cc4](https://github.com/bibymaths/curvefit-stats/commit/6be2cc4513eda5b6b222624d5d7211da039b009a))

- Implement curvefit-stats inference, estimators, validation and release tooling ([dc3677f](https://github.com/bibymaths/curvefit-stats/commit/dc3677f59754dd8891e425c14472fd3af03aa669))

- Specify UTF-8 encoding when reading documentation files in tests. ([fde1530](https://github.com/bibymaths/curvefit-stats/commit/fde15302e578e5be31974cf721ed1e750442f498))

- Add Zensical documentation site and validated GitHub Pages deployment ([05b853d](https://github.com/bibymaths/curvefit-stats/commit/05b853d28deb12e40d92452704b53f02b568a66e))

- Fix confidence-band underflow for singular covariance

Check zero-error gradients against positive covariance eigen-directions using exact diagnostic products. Preserve true nullspace bands and representable subnormal widths. Add ten independent regression cases and regenerate the API snapshot for PR #1 review 4263103099. ([d9b16f5](https://github.com/bibymaths/curvefit-stats/commit/d9b16f588a96d20b466a535ce31d179da3e3f4db))

- Scale profile optimum checks to the likelihood-ratio threshold

Use the same signed deviance statistic for baseline validation and directional checks. Compare improvements with tolerance times sigma_level squared; use log1p for nearby unknown-scale SSEs. Add fifteen analytical regression cases and regenerate the API snapshot. ([1bfc36e](https://github.com/bibymaths/curvefit-stats/commit/1bfc36e992a0191d27c829e91da8df242314c5e3))

- Improve nuisance optimization process and test Jacobian behavior near parameter bounds

- Enhanced nuisance optimizer to handle near-collinear directions in Jacobian.
- Added safeguards to prevent false convergence in gradient-based methods.
- Introduced a new test case to validate rejection of suboptimal solutions with nearly collinear parameters.
- Updated minimum Python version to 3.12 in `pyproject.toml` and MyPy configuration. ([5d789a0](https://github.com/bibymaths/curvefit-stats/commit/5d789a084b5302fed8d3bcc695e3ed5b65eaab10))

- Lower Python version requirement to 3.11 in `pyproject.toml` and MyPy configuration. ([3378264](https://github.com/bibymaths/curvefit-stats/commit/3378264d78f8ca8554b46f99a9379ce67189b139))

- **Update `README.md`, add visual elements, refine tests, and enforce compatibility constraints**

- Added logo and improved visual elements in `README.md`.
- Extended decay.svg to include confidence bands, legend, and observation uncertainty.
- Refactored test cases with improved validation for suboptimal parameter rejection.
- Enforced Python 3.11+ and SciPy <1.19 compatibility in `pyproject.toml`. ([9d77e3b](https://github.com/bibymaths/curvefit-stats/commit/9d77e3b7b14074672e36576ebadf999454490029))

- Merge pull request #1 from bibymaths/release/prepare-0.1.0

## Summary

Implement `curvefit-stats` 0.1.0 as a typed scientific Python package for nonlinear fitting and statistically qualified inference. The initial repository contained a placeholder README; this PR supplies the complete package, independent numerical reference tests, executable documentation, reproducible development configuration, distribution validation, and CI.

The package retains exactly two functional submodules, `curvefit_stats.inference` and `curvefit_stats.estimators`. All ten approved features are available through both their individual modules and consistent package-level exports. NumPy and SciPy are the only third-party runtime dependencies.

## Implemented features

| Feature | Behavior |
| --- | --- |
| `generate_error_bands` | Delta-method pointwise confidence and prediction intervals, with explicit normal/Student-t reference and future-noise conventions. |
| `audit_fit` | Gaussian fit diagnostics and normalized AIC/BIC, including known versus estimated common variance and correct parameter counting. |
| `fit_summary` | R-inspired parameter tables with approximate nonlinear Wald statistics and Student-t tail probabilities. |
| `profile_likelihood_errors` | Asymmetric errors from profile likelihood with nuisance-parameter reoptimization and noise-model-specific thresholds. |
| `bootstrap_curve_fit` | Seeded, fixed-attempt residual bootstrap with empirical parameter covariance and explicit failed/invalid replicate handling. |
| `collinearity_diagnostics` | Parameter correlation, VIF-like local diagnostics, rank and covariance conditioning. |
| `robust_bounded_pcov` | Free-parameter conditional covariance with NaN rows/columns for active constraints. |
| `propagate_uncertainty` | Seeded Gaussian Monte Carlo propagation to scalar/vector derived quantities, with validated covariance and invalid-output accounting. |
| `GlobalFitGroup` | Joint least-squares fitting of named, unequal-length datasets with shared/tied and dataset-specific parameters; covariance from the global Jacobian. |
| `eiv_curve_fit` | SciPy-like x-first model adapter to ODR, supporting `sx`, `sy`, absolute/relative covariance scaling and optional typed diagnostics. |

## Correctness and numerical safeguards

- Validate shapes, uncertainty signs/finiteness, covariance symmetry/positive semidefiniteness, residual degrees of freedom, parameter mappings and bounds.
- Use independent analytical/numerical references instead of package self-comparisons: linear regression, exponential decay, Michaelis–Menten kinetics, logistic growth, strongly correlated models, Deming regression and explicit latent-coordinate EIV optimization.
- Exercise nearly singular Jacobians, extreme parameter/response/uncertainty scales, active lower/upper/fixed bounds, heteroscedastic observations and large floating-point offsets.
- Avoid presenting optimizer failures, pseudoinverse covariance for unidentified fits, unresolved profile endpoints, or collapsed numerical uncertainty as usable inference.
- Normalize optimizer coordinates/objectives where needed and restore physical-unit outputs; retain covariance provenance and distinguish conditional, ODR, bootstrap and ordinary least-squares interpretations.
- Include regressions for the nine defects identified by the independent scientific review, including false convergence under relative-weight scaling, unreliable bootstrap refits, spurious singular Wald inference, profile seed-dependent root errors and large-offset/underflow arithmetic.

## Documentation and packaging

- Comprehensive NumPy-style API docstrings plus README, user guide, API reference, integration guidance, stress-test report, scientific review and final release-readiness report.
- Eleven executable scientific workflows covering all requested inference and estimator use cases; no plotting or additional runtime dependencies.
- PEP 517 setuptools build, MIT metadata, `py.typed`, wheel/sdist content checks and isolated installation/import smoke tools.
- Strict Ruff, mypy and pytest configuration; warnings are errors except explicitly expected statistical/backend cases. Combined statement/branch coverage must remain at least 90%.
- Hash-locked Python 3.12 development dependencies and pinned build requirements.
- GitHub Actions stages for strict quality checks, distribution generation/installation, and wheel-based Linux/Windows/macOS compatibility tests.

## Executed validation

All local checks below passed. The installed-wheel suite was executed in every Linux row:

| Platform / CPython | NumPy | SciPy | Tests | Statement/branch coverage |
| --- | --- | --- | --- | --- |
| Linux / 3.11.17 | 1.26.4 | 1.12.0 | 405 passed | 95.22% |
| Linux / 3.12.14 | 1.26.4 | 1.12.0 | 405 passed | 95.22% |
| Linux / 3.12.14 | 2.5.4 | 1.18.1 | 405 passed | 95.27% |
| Linux / 3.13.5 | 2.5.4 | 1.18.1 | 405 passed | 95.27% |
| Linux / 3.14.8 | 2.5.4 | 1.18.1 | 405 passed | 95.27% |

The fresh hash-locked development environment also passed 405 tests with 95.27% coverage. No skipped or expected-failure tests occurred.

- Ruff lint and formatting: passed for all 15 Python files.
- Strict mypy: passed for all 15 source/test/example/tool files.
- Doctests: 13 passed in the source environment and each Linux wheel environment, with one explicitly expected conditional-bound covariance warning.
- All eleven workflows and README/user-guide/integration Python examples: passed.
- Wheel and sdist builds, strict Twine validation and archive checks: passed.
- Isolated installed-package imports and all-ten-feature smokes: passed for all Linux wheel environments and separate runtime-only wheel/direct-sdist environments.
- `pip check`: passed in all eight checked environments. Runtime-only environments contained pip, this package, NumPy and SciPy only.
- Independent fixed-epoch wheel builds: byte-identical.
- Actionlint 1.7.7 and staged whitespace validation: passed.

Exact commands and evidence are recorded in `docs/release-readiness.md`; scientific correction details are in `docs/review.md`.

## Scientific and compatibility limitations

Pointwise bands are not simultaneous regions. Nonlinear Wald/Delta and profile/Wilks inference require their documented local/regular/asymptotic assumptions. Numerical rank is not structural identifiability. Residual bootstrap assumes IID observation errors at fixed predictors, and failed-fit selection can bias retained samples. Bound covariance holds active values fixed and is not unrestricted or selection-calibrated uncertainty. Gaussian propagation may violate constraints; finite ratio samples do not establish finite population moments. Global fitting uses one common unknown variance multiplier.

EIV is a separate ODR statistical model: response-only likelihood, bootstrap and Student-t routines must not be treated as ODR inference. The ODR adapter supports one predictor/response and scalar/vector axis errors, without bounds or correlated axis errors. ODR is deprecated in SciPy 1.17 and removed in 1.19, so supported SciPy is explicitly restricted to `>=1.12,<1.19`; deprecated backends emit a documented compatibility warning. Runtime Python is >=3.11, NumPy >=1.26.4,<3.

## Release gate

The final local recommendation is **NOT READY for public release until the actual GitHub Actions workflow passes on this candidate**, including configured Windows/macOS Python 3.12 jobs. Workflow syntax and all Linux equivalents have passed; Windows/macOS runner execution has not yet been verified. This PR prepares review and CI execution; it does not publish a package or assert cross-platform readiness prematurely. ([71886a1](https://github.com/bibymaths/curvefit-stats/commit/71886a1a66b72f62bf0a7364d7642003fa4d1bcc))

