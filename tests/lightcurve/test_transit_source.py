"""Guard that `lazuli_transit` resolves to the copy vendored in this repo.

A standalone lazuli-transit checkout installed editable elsewhere will shadow
`packages/lazuli-transit/` unless pytest's `pythonpath` puts the in-repo src
first. When that happened, local runs and CI silently tested different source.
"""

from pathlib import Path

import lazuli_transit

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestTransitSource:
    def test_lazuli_transit_resolves_inside_this_repo(self):
        """The imported package must come from packages/lazuli-transit/src."""
        resolved = Path(lazuli_transit.__file__).resolve()
        expected = REPO_ROOT / "packages" / "lazuli-transit" / "src" / "lazuli_transit"
        assert resolved.parent == expected, (
            f"lazuli_transit resolved to {resolved.parent}, not {expected}; "
            "an editable install of the standalone repo is shadowing it"
        )
