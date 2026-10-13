# Installation

`curvefit-stats` is a lightweight package built on NumPy and SciPy. No pandas,
plotting library or notebook runtime is required.

## Supported runtime

| Component | Supported range |
| --- | --- |
| Python | >=3.11; Linux compatibility tested on 3.11–3.14 |
| NumPy | >=1.26.4,<3 |
| SciPy | >=1.12,<1.19 |

The tested numerical stacks are recorded in [release verification](release-readiness.md).
Version ranges are compatibility policy, not an exhaustive test of every combination.

!!! warning "ODR compatibility"

    `scipy.odr` is deprecated from SciPy 1.17 and removed in 1.19. EIV calls emit
    an explicit `FutureWarning` on deprecated backends. The dependency upper
    bound excludes removal; no replacement backend is currently configured.

## Install the source checkout

Until a registry release is published, use the repository:

=== "uv"

    ```bash
    git clone https://github.com/bibymaths/curvefit-stats.git
    cd curvefit-stats
    uv venv --python 3.12
    uv pip install --python .venv/bin/python .
    .venv/bin/python -c "import curvefit_stats; print(curvefit_stats.__all__)"
    ```

=== "pip"

    ```bash
    git clone https://github.com/bibymaths/curvefit-stats.git
    cd curvefit-stats
    python -m venv .venv
    python -m pip --python .venv/bin/python install .
    .venv/bin/python -c "import curvefit_stats; print(curvefit_stats.__all__)"
    ```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.
An editable development checkout can use `uv pip install -e .` after activating
its environment. [Development instructions](development.md) define the separate,
locked quality-check environment; installing the base package installs no dev tools.

## Install built artifacts

For a wheel or source distribution you have built locally:

```bash
uv pip install --python .venv/bin/python dist/curvefit_stats-0.1.0-py3-none-any.whl
# Or build/install from the source distribution:
uv pip install --python .venv/bin/python dist/curvefit_stats-0.1.0.tar.gz
```

See [packaging checks](development.md#build-and-inspect-artifacts) before sharing
artifacts. Proceed to [the quickstart](quickstart.md) once imports succeed.
