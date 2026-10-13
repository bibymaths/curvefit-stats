# API reference

The reference below is generated on every build from the Python source and its
NumPy-style docstrings. Signatures are never maintained by hand. Zensical's
supported API navigation discovers the package; `mkdocstrings` renders its
members. Source changes are also watched during local preview.

All ten features are available from `curvefit_stats` and directly from their
functional modules. There are exactly two functional submodules:

- [Inference](curvefit_stats/inference.md): bands, diagnostics, parameter tables,
  profile likelihood, bootstrap, collinearity, conditional covariance and propagation.
- [Estimators](curvefit_stats/estimators.md): shared-parameter global fitting and EIV.

## Read the statistical contract

A covariance array does not retain its noise model, estimator, residual degrees
of freedom or active constraints. Keep that context when combining APIs. Start
with [the workflow guide](../workflows.md) and
[cross-module integration](../integration.md).

Type aliases denote float64 NumPy arrays or callables accepting
`func(x, *parameters)`. Result dataclass names beginning with an underscore are
implementation details; access the documented result attributes rather than
importing private classes. Exceptions raised by your own model or derived
callables can propagate in addition to each API's listed errors.
