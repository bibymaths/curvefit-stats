"""Check installed distribution metadata and the advertised typing contract."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import tomllib
from pathlib import Path
from typing import get_type_hints

import curvefit_stats as cs


def test_distribution_runtime_metadata_and_extras() -> None:
    """Keep numerical runtime requirements separate from optional development tools."""
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    distribution = importlib.metadata.metadata("curvefit-stats")
    assert distribution["Version"] == project["version"]
    assert distribution["License-Expression"] == project["license"]
    assert distribution["Requires-Python"] == project["requires-python"]
    requirements = distribution.get_all("Requires-Dist", [])
    runtime = {requirement for requirement in requirements if ";" not in requirement}
    assert runtime == {"numpy<3,>=1.26.4", "scipy<1.19,>=1.12"}
    assert set(distribution.get_all("Provides-Extra", [])) == {"dev", "test"}
    assert all(
        'extra == "' in requirement
        for requirement in requirements
        if ";" in requirement
    )


def test_typing_marker_and_resolvable_public_annotations() -> None:
    """Ensure installed typing data and annotations are usable by package consumers."""
    assert importlib.resources.files(cs).joinpath("py.typed").is_file()
    for name in cs.__all__:
        obj = getattr(cs, name)
        callable_object = obj.__init__ if name == "GlobalFitGroup" else obj
        annotations = get_type_hints(callable_object)
        assert "return" in annotations, name
    assert "return" in get_type_hints(cs.GlobalFitGroup.fit)
