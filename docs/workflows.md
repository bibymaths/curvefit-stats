# Choose a workflow

Choose the statistical model before choosing the uncertainty calculation.
Every function accepts ordinary NumPy-compatible arrays, but a covariance matrix
alone cannot tell you where it came from.

| Question | Feature | Assumption to check |
| --- | --- | --- |
| How uncertain is the mean curve? | [`generate_error_bands`][curvefit_stats.inference.generate_error_bands] | Local Delta approximation; compatible covariance, reference law and fitting dof. |
| Where might one new observation fall? | [`generate_error_bands`][curvefit_stats.inference.generate_error_bands] | Independent future noise; explicitly supplied for known/heteroscedastic errors. |
| How do regular response-only models compare? | [`audit_fit`][curvefit_stats.inference.audit_fit] | Same observations/units, Gaussian likelihood and maximum-likelihood estimates. |
| What does a parameter table say? | [`fit_summary`][curvefit_stats.inference.fit_summary] | Approximate Student-t/Wald tests of zero parameters, using N-p. |
| Are parameter errors asymmetric? | [`profile_likelihood_errors`][curvefit_stats.inference.profile_likelihood_errors] | Regular profile/Wilks calibration and successful local nuisance reoptimization. |
| How variable are empirical refits? | [`bootstrap_curve_fit`][curvefit_stats.inference.bootstrap_curve_fit] | IID observation errors, adequate mean model, fixed predictors; inspect failed attempts. |
| Which parameters are locally correlated? | [`collinearity_diagnostics`][curvefit_stats.inference.collinearity_diagnostics] | Local covariance-based evidence, not structural identifiability. |
| What covariance is available at a bound? | [`robust_bounded_pcov`][curvefit_stats.inference.robust_bounded_pcov] | Active parameters held fixed; only the free block has conditional covariance. |
| How uncertain is a derived quantity? | [`propagate_uncertainty`][curvefit_stats.inference.propagate_uncertainty] | Gaussian local parameter approximation; domain validity and finite moments. |
| Can experiments share parameters? | [`GlobalFitGroup`][curvefit_stats.estimators.GlobalFitGroup] | One joint residual model and one common unknown scale when relative weighting is used. |
| Are both axes measured with error? | [`eiv_curve_fit`][curvefit_stats.estimators.eiv_curve_fit] | Independent scalar/vector axis errors, ODR covariance and supported SciPy backend. |

## A useful starting sequence

1. Fit the appropriate response-only, joint or EIV model.
2. Inspect convergence, model adequacy, numerical rank and constraint activity.
3. Retain the noise/covariance convention and residual degrees of freedom.
4. Choose an interval or propagation method whose assumptions match that fit.
5. Inspect diagnostics and uncertainty failures before interpreting results.

!!! warning "Covariance is not interchangeable metadata"

    EIV covariance does not make a vertical-response likelihood audit an ODR
    likelihood. Bootstrap covariance does not establish a Student-t pivot.
    Conditional covariance at a bound is not unrestricted marginal uncertainty.

See [integration guidance](integration.md) for valid projections between modules,
[the concepts guide](user-guide.md) for equations and limitations, and
[executable examples](examples.md) for reproducible complete workflows.
