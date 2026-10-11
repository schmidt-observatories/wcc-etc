"""Fail a release build whose metadata PyPI would reject or whose tag disagrees
with pyproject.toml (#108). On a non-tag run only the dependency check applies."""

import os
import sys
import tomllib


def main():
    with open("pyproject.toml", "rb") as fh:
        project = tomllib.load(fh)["project"]
    bad = False

    direct = [d for d in project["dependencies"] if " @ " in d]
    if direct:
        print("FAIL: PyPI refuses direct-URL dependencies:", *direct, sep="\n  ")
        bad = True

    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/tags/v"):
        tag = ref.removeprefix("refs/tags/v")
        if tag != project["version"]:
            print(f"FAIL: tag v{tag} != pyproject version {project['version']}")
            bad = True
        else:
            print(f"tag v{tag} matches pyproject version")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
