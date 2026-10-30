#!/usr/bin/env python3
"""Setup script for LAFAN1 motion CSVs in reference_motions/data_lafan1/g1/.

Usage:
    python reference_motions/scripts/download_lafan1.py
    python reference_motions/scripts/download_lafan1.py --src /path/to/extracted/lafan1_csvs
    python reference_motions/scripts/download_lafan1.py --url https://.../lafan1_g1.tar.gz
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_DIR = ROOT / "reference_motions" / "data_lafan1" / "g1"
MANIFEST_FILE = ROOT / "reference_motions" / "lists" / "lafan1_motions.txt"


def check_existing_clips() -> int:
    if TARGET_DIR.is_dir():
        return len(list(TARGET_DIR.glob("*.csv")))
    return 0


def copy_from_local_source(src_dir: Path) -> int:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for csv_file in sorted(src_dir.rglob("*.csv")):
        shutil.copy2(csv_file, TARGET_DIR / csv_file.name)
        count += 1
    return count


def download_and_extract_archive(url: str) -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    tmp_archive = TARGET_DIR / "temp_download_archive"
    print(f"[*] Downloading archive from: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(tmp_archive, "wb") as f:
        shutil.copyfileobj(resp, f)

    count = 0
    try:
        if zipfile.is_zipfile(tmp_archive):
            with zipfile.ZipFile(tmp_archive) as z:
                for member in z.infolist():
                    if member.filename.endswith(".csv"):
                        filename = Path(member.filename).name
                        with z.open(member) as s_in, open(TARGET_DIR / filename, "wb") as s_out:
                            shutil.copyfileobj(s_in, s_out)
                        count += 1
        elif tarfile.is_tarfile(tmp_archive):
            with tarfile.open(tmp_archive) as t:
                for member in t:
                    if member.isfile() and member.name.endswith(".csv"):
                        filename = Path(member.name).name
                        f_in = t.extractfile(member)
                        if f_in:
                            with open(TARGET_DIR / filename, "wb") as s_out:
                                shutil.copyfileobj(f_in, s_out)
                            count += 1
        else:
            raise ValueError("Unsupported archive format. Expected .zip or .tar.gz/.tar")
    finally:
        if tmp_archive.exists():
            tmp_archive.unlink()
    return count


def main():
    parser = argparse.ArgumentParser(description="Setup LAFAN1 motion clips locally")
    parser.add_argument("--src", type=str, default=None, help="Path to local folder containing LAFAN1 CSVs")
    parser.add_argument("--url", type=str, default=None, help="Direct URL to archive (.zip / .tar.gz) containing LAFAN1 CSVs")
    args = parser.parse_args()

    num_existing = check_existing_clips()
    if num_existing >= 40:
        print(f"[+] All 40 LAFAN1 motion clips are already present in: {TARGET_DIR}")
        return

    if args.src:
        src_path = Path(args.src)
        print(f"[*] Copying LAFAN1 CSVs from local source: {src_path}...")
        copied = copy_from_local_source(src_path)
        print(f"[+] Successfully copied {copied} CSV clips to {TARGET_DIR}")
        return

    if args.url:
        print(f"[*] Downloading archive from URL: {args.url}...")
        extracted = download_and_extract_archive(args.url)
        print(f"[+] Successfully extracted {extracted} CSV clips to {TARGET_DIR}")
        return

    print("=" * 70)
    print("LAFAN1 Motion Clips Setup")
    print("=" * 70)
    print(f"Current clips present: {num_existing}/40 in {TARGET_DIR}")
    print("\nTo populate the LAFAN1 motion clips:")
    print("1. From a local folder:")
    print("   python reference_motions/scripts/download_lafan1.py --src /path/to/csvs")
    print("\n2. From an external URL (zip or tar.gz):")
    print("   python reference_motions/scripts/download_lafan1.py --url https://.../lafan1.zip")
    print("\n3. Or copy your 40 LAFAN1 CSVs directly into:")
    print(f"   {TARGET_DIR}/")
    print("   (Refer to reference_motions/lists/lafan1_motions.csv for required clip names)")
    print("=" * 70)


if __name__ == "__main__":
    main()
