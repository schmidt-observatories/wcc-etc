"""Tests for the legacy wcc_etc.wcc_etc module (deprecated interface)."""

import warnings
import numpy as np


class TestLegacyWCCETC:
    def _import(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from wcc_etc.wcc_etc import get_wcc_snr_and_simulation
        return get_wcc_snr_and_simulation

    def test_blackbody_returns_finite_snr(self):
        fn = self._import()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr, sim = fn(mag=15, texp=60, source_type="blackbody", teff=5777)
        assert np.isfinite(snr)
        assert snr > 0

    def test_file_source_returns_finite_snr(self, tmp_path):
        fn = self._import()
        specfile = tmp_path / "source.dat"
        wave = np.linspace(3000.0, 11000.0, 40)
        flux = np.full_like(wave, 1e-16)
        np.savetxt(specfile, np.column_stack([wave, flux]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            snr, sim = fn(
                mag=15, texp=60, source_type="file", source_file=str(specfile)
            )
        assert np.isfinite(snr)
        assert snr > 0
        assert sim.scene.source.meta["spectrum"] == "file"
