#!/usr/bin/env python
"""Download and extract the PhysioPain dataset for local work.

You do NOT need this on Kaggle -- just attach `orvile/physiopain-dataset` to
the notebook and it appears under /kaggle/input.

The full archive is 1.26 GB zipped / 5.36 GB extracted, but the EEG tier we
actually model on is only 19 MB. So by default we pull the zip, extract just
the EEG + survey, and leave the 4.9 GB watch tier alone until you ask for it.

    python scripts/get_data.py                # EEG + survey  (~19 MB on disk)
    python scripts/get_data.py --watch        # ... plus the wristband tier
    python scripts/get_data.py --keep-zip     # don't delete the 1.26 GB zip

No Kaggle credentials needed -- this dataset is served unauthenticated.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/orvile/physiopain-dataset"
EXPECTED_BYTES = 1_264_163_874

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
ZIP = RAW / "physiopain.zip"

# what to pull out of the archive
EEG_AND_SURVEY = ("RAW EEG DATA (1Hz)", "SURVEY DATA")
WATCH = ("PROCESSED WATCH DATA",)


def _progress(done: int, total: int) -> None:
    pct = 100 * done / total if total else 0
    bar = "#" * int(pct // 2.5)
    sys.stdout.write(f"\r  [{bar:<40}] {pct:5.1f}%  {done/1e6:7.0f} / {total/1e6:.0f} MB")
    sys.stdout.flush()


def download() -> None:
    if ZIP.exists() and ZIP.stat().st_size == EXPECTED_BYTES:
        print(f"  zip already present ({ZIP.stat().st_size/1e9:.2f} GB), skipping download")
        return

    RAW.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(RAW).free
    if free < 3e9:
        print(f"  WARNING: only {free/1e9:.1f} GB free. The zip alone needs 1.3 GB.")

    print(f"  downloading {URL}")
    with urllib.request.urlopen(URL) as r, open(ZIP, "wb") as out:
        total = int(r.headers.get("Content-Length") or EXPECTED_BYTES)
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            _progress(done, total)
    print()


def extract(prefixes: tuple[str, ...]) -> None:
    z = zipfile.ZipFile(ZIP)
    want = [
        i for i in z.infolist()
        if not i.is_dir() and any(p in i.filename for p in prefixes)
    ]
    mb = sum(i.file_size for i in want) / 1e6
    print(f"  extracting {len(want)} files ({mb:.1f} MB) ...")
    for i in want:
        z.extract(i, RAW)
    print("  done")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", action="store_true",
                    help="also extract the 4.9 GB processed wristband tier")
    ap.add_argument("--keep-zip", action="store_true",
                    help="keep the 1.26 GB zip after extracting")
    args = ap.parse_args()

    print("PhysioPain dataset fetch")
    download()

    prefixes = EEG_AND_SURVEY + (WATCH if args.watch else ())
    extract(prefixes)

    if not args.keep_zip:
        # keep it if we might still need the watch tier later
        if args.watch:
            ZIP.unlink(missing_ok=True)
            print("  removed zip")
        else:
            print("  keeping zip (you'll need it for --watch); pass --keep-zip to silence")

    # sanity check: did we get the 83 subject files?
    eeg = RAW / "PhysioPain Dataset" / "Multimodal Pain Dataset" / "RAW EEG DATA (1Hz)" / "All"
    n = len(list(eeg.glob("S*.csv"))) if eeg.exists() else 0
    print(f"\n  EEG subject files found: {n} (expected 83)")
    return 0 if n == 83 else 1


if __name__ == "__main__":
    raise SystemExit(main())
