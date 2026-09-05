"""The top-level package surface: ``__all__`` and the names the README promises."""

import re
from pathlib import Path

import pytest

import wcc_etc

README = Path(__file__).resolve().parents[2] / "README.md"


def readme_imported_names():
    """Names the README's fenced Python blocks import from ``wcc_etc``."""
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    names = set()
    for block in blocks:
        for match in re.finditer(r"^from wcc_etc import (.+)$", block, re.M):
            names.update(n.strip() for n in match.group(1).split(","))
    return sorted(names)


@pytest.mark.parametrize("name", wcc_etc.__all__)
def test_every_all_name_is_importable(name):
    """Every name in ``__all__`` actually exists on the package."""
    assert hasattr(wcc_etc, name)


@pytest.mark.parametrize("name", readme_imported_names())
def test_readme_name_is_exported(name):
    """Every name the README imports from wcc_etc is in ``__all__``."""
    assert name in wcc_etc.__all__
