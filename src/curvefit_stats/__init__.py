"""Statistical inference and estimation for SciPy-style nonlinear models.

All ten features are re-exported here without wrappers. Direct imports from
:mod:`curvefit_stats.inference` and :mod:`curvefit_stats.estimators` retain the
same objects and signatures. These are the only functional submodules.

Covariance conventions
----------------------
A covariance array alone cannot encode its estimator, noise model, degrees of
freedom, or boundary status. Callers must preserve that context:

* SciPy curve_fit with an unknown common Gaussian scale supports approximate
  Student-t/Wald inference using the original fitting N-p. Known absolute scales
  use ``generate_error_bands(..., reference='normal')`` instead.
* GlobalFitGroup uses one global degree of freedom per estimated shared
  parameter. For per-dataset bands, project the global covariance using the
  group's integer mapping and pass the global dof explicitly, without supplying
  per-dataset y to infer a new local dof. Use explicit prediction_variance from
  the global noise model rather than reestimating it from a single dataset.
* EIV covariance comes from ODR, not vertical-response least squares. It can
  support Delta Method mean bands at exact evaluation coordinates and Gaussian
  parameter propagation. audit_fit, profile_likelihood_errors, fit_summary,
  residual bootstrap, and robust_bounded_pcov use response-only least-squares
  assumptions and do not automatically become ODR methods when passed ODR
  estimates. Future uncertain predictors require their own observation model.
* Bootstrap covariance is an empirical parameter covariance, not a Student-t
  pivot. Gaussian propagation from it is a further approximation; bootstrap
  percentile intervals should be calculated directly from bootstrap samples.
* Bound-active covariance has NaN rows and columns. Full-matrix propagation is
  rejected. Extracting a finite free block only permits inference conditional
  on the active parameters, never unrestricted marginal uncertainty for them.

See ``docs/integration.md`` for complete examples and limitations. No estimator
provenance is inferred from plain parameter/covariance arrays.
"""

from .estimators import GlobalFitGroup, eiv_curve_fit
from .inference import (
    audit_fit,
    bootstrap_curve_fit,
    collinearity_diagnostics,
    fit_summary,
    generate_error_bands,
    profile_likelihood_errors,
    propagate_uncertainty,
    robust_bounded_pcov,
)

__all__ = [
    "GlobalFitGroup",
    "audit_fit",
    "bootstrap_curve_fit",
    "collinearity_diagnostics",
    "eiv_curve_fit",
    "fit_summary",
    "generate_error_bands",
    "profile_likelihood_errors",
    "propagate_uncertainty",
    "robust_bounded_pcov",
]
