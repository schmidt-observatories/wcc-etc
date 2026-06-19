from pathlib import Path

import lazuli_transit


def test_package_imports():
    assert lazuli_transit.__version__


def test_no_wcc_etc_dependency_in_source():
    pkg_dir = Path(lazuli_transit.__file__).parent
    offenders = [p.name for p in pkg_dir.rglob("*.py") if "wcc_etc" in p.read_text()]
    assert not offenders, f"lazuli_transit must not reference wcc_etc: {offenders}"
