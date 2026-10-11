"""Fail the build if dist/ artifacts carry local junk or blow the size ceiling."""

import glob
import os
import re
import sys
import tarfile
import zipfile

JUNK = re.compile(r"\.DS_Store$|\.sw[op]$|_report\.pdf$|\.fgd$|/scatter/|/flats/|/dat_uvi/|data/psfs/.*\.txt$")
MAX_BYTES = 20 * 1024**2  # clean build is ~12 MB; the Pickles + Brown atlases dominate


def entries(path):
    if path.endswith(".whl"):
        return zipfile.ZipFile(path).namelist()
    with tarfile.open(path) as tar:
        return tar.getnames()


def main():
    paths = sorted(glob.glob("dist/*"))
    if not paths:
        print("FAIL: dist/ is empty")
        return 1

    bad = False
    for path in paths:
        names = entries(path)
        size = os.path.getsize(path)
        print(f"{path}: {size / 1024**2:.1f} MB, {len(names)} entries")
        if size > MAX_BYTES:
            print(f"  FAIL: over the {MAX_BYTES / 1024**2:.0f} MB ceiling")
            bad = True
        for name in names:
            if JUNK.search(name):
                print(f"  FAIL: junk in artifact: {name}")
                bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
