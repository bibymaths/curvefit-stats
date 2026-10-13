"""Verify executable documentation against the actual public implementation."""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import curvefit_stats as cs

_ROOT = Path(__file__).resolve().parents[1]


def test_all_documented_workflows_execute() -> None:
    """Execute every workflow while treating undocumented warnings as errors."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_ROOT / "src")
    command = [
        sys.executable,
        "-W",
        "error",
        "-W",
        "default:Strong local parameter correlation:UserWarning",
        "-W",
        "default:Bound-active parameter rows:UserWarning",
        "-W",
        "default:scipy.odr is deprecated in SciPy:FutureWarning",
        str(_ROOT / "examples" / "workflows.py"),
    ]
    result = subprocess.run(
        command, cwd=_ROOT, env=environment, capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name in [
        "confidence",
        "intervals",
        "diagnostics",
        "summary",
        "profile",
        "bootstrap",
        "collinearity",
        "bounds",
        "propagation",
        "global",
        "eiv",
    ]:
        assert f"\n{name.upper()}\n" in result.stdout


def test_readme_and_user_guide_python_examples_execute() -> None:
    """Run the quick start followed by the explicitly dependent prediction example."""
    namespace: dict[str, Any] = {}
    for relative in ["README.md", "docs/user-guide.md"]:
        source = (_ROOT / relative).read_text()
        for index, snippet in enumerate(
            re.findall(r"```python\n(.*?)```", source, flags=re.DOTALL)
        ):
            exec(
                compile(snippet, f"{relative}:python-example-{index}", "exec"),
                namespace,
            )
    assert namespace["lower"].shape == (101,)
    assert namespace["prediction_lower"].shape == (101,)


def test_integration_guide_python_examples_execute() -> None:
    """Execute the existing integration snippets in their documented sequence."""
    namespace: dict[str, Any] = {}
    source = (_ROOT / "docs" / "integration.md").read_text()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="scipy.odr is deprecated in SciPy", category=FutureWarning
        )
        for index, snippet in enumerate(
            re.findall(r"```python\n(.*?)```", source, flags=re.DOTALL)
        ):
            exec(
                compile(snippet, f"integration.md:python-example-{index}", "exec"),
                namespace,
            )
    assert namespace["result"].success
    assert namespace["odr"].success


def test_api_reference_matches_source_docstrings_and_signatures() -> None:
    """Guard the checked-in reference against signature and docstring drift."""
    reference = (_ROOT / "docs" / "api.md").read_text()
    for name in cs.__all__:
        obj = getattr(cs, name)
        assert f"{name}{inspect.signature(obj)}" in reference
        docstring = inspect.getdoc(obj)
        assert docstring is not None and docstring in reference
        for section in ["Parameters", "Returns", "Raises", "Notes", "Examples"]:
            assert f"\n{section}\n" in docstring
    assert f"fit{inspect.signature(cs.GlobalFitGroup.fit)}" in reference
    fit_docstring = inspect.getdoc(cs.GlobalFitGroup.fit)
    assert fit_docstring is not None and fit_docstring in reference
