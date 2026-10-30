#!/usr/bin/env python3
"""Download and stream-extract BONES-SEED motion clips from Hugging Face.

Ingests G1 humanoid joint angle CSVs into:
    reference_motions/data_bones_seed/g1/

Requirements:
    1. Hugging Face account with access granted on:
       https://huggingface.co/datasets/bones-studio/seed
    2. Hugging Face User Access Token (read permission).

Usage:
    # Automatic (uses ~/.cache/huggingface/token or HF_TOKEN env var):
    python reference_motions/scripts/download_bones_seed.py

    # Or provide token explicitly:
    python reference_motions/scripts/download_bones_seed.py --token hf_...
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TARGET_DIR = ROOT / "reference_motions" / "data_bones_seed" / "g1"
REPO_URL = "https://huggingface.co/datasets/bones-studio/seed/resolve/main/g1.tar.gz"
API_CHECK_URL = "https://huggingface.co/api/datasets/bones-studio/seed"
DEFAULT_SESSION = "230509"


def get_hf_token(cli_token: str | None = None) -> str | None:
    if cli_token:
        return cli_token.strip()
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    cache_token = Path.home() / ".cache" / "huggingface" / "token"
    if cache_token.is_file():
        token = cache_token.read_text().strip()
        if token:
            return token
    return None


def verify_repo_access(token: str) -> bool:
    req = urllib.request.Request(API_CHECK_URL, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print("\n[!] HTTP 403 Forbidden: You do not have permission to access 'bones-studio/seed'.")
            print("    Please log in to Hugging Face and click 'Agree and access repository' on:")
            print("    https://huggingface.co/datasets/bones-studio/seed\n")
        elif e.code == 401:
            print("\n[!] HTTP 401 Unauthorized: Invalid Hugging Face token provided.\n")
        else:
            print(f"\n[!] Hugging Face API check failed with status {e.code}: {e.reason}\n")
        return False
    except Exception as e:
        print(f"[!] Network error verifying repo access: {e}")
        return False


def stream_extract_bones_seed(token: str, session: str | None = DEFAULT_SESSION, force: bool = False):
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    session_dir = TARGET_DIR / session if session else TARGET_DIR

    if not force and session_dir.is_dir():
        existing = list(session_dir.rglob("*.csv"))
        if len(existing) >= 1000:
            print(f"[+] BONES-SEED clips already present ({len(existing)} CSVs in {session_dir}).")
            print("    Use --force to re-download.")
            return

    print(f"[*] Connecting to Hugging Face repository...")
    if not verify_repo_access(token):
        sys.exit(1)

    print(f"[*] Access verified! Streaming 'g1.tar.gz' from Hugging Face...")
    prefix_filter = f"g1/csv/{session}/" if session else "g1/csv/"
    print(f"[*] Extracting CSV clips matching prefix '{prefix_filter}' into: {TARGET_DIR}")

    req = urllib.request.Request(
        REPO_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "allmimic-downloader/1.0",
        },
    )

    extracted_count = 0
    try:
        with urllib.request.urlopen(req) as resp:
            with tarfile.open(mode="r|gz", fileobj=resp) as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".csv"):
                        continue

                    if session and not member.name.startswith(prefix_filter):
                        # If we passed the targeted session, break early to save bandwidth
                        if extracted_count > 0 and not member.name.startswith("g1/csv/"):
                            break
                        continue

                    # Strip 'g1/csv/' prefix so files land under reference_motions/data_bones_seed/g1/<session>/
                    rel_path = member.name[len("g1/csv/"):] if member.name.startswith("g1/csv/") else member.name
                    out_path = TARGET_DIR / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    f_in = tar.extractfile(member)
                    if f_in:
                        with open(out_path, "wb") as f_out:
                            f_out.write(f_in.read())
                        extracted_count += 1
                        if extracted_count % 50 == 0:
                            print(f"  [+] Extracted {extracted_count} CSV clips...", end="\r", flush=True)

        print(f"\n[+] Successfully extracted {extracted_count} BONES-SEED motion clips into: {TARGET_DIR}")
    except KeyboardInterrupt:
        print(f"\n[!] Download interrupted by user. Extracted {extracted_count} clips so far.")
    except Exception as e:
        print(f"\n[!] Error streaming archive: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download BONES-SEED motion dataset from Hugging Face")
    parser.add_argument("--token", default=None, help="Hugging Face user access token")
    parser.add_argument("--session", default=DEFAULT_SESSION, help=f"Session subfolder to extract (default: '{DEFAULT_SESSION}' for 1,388 catalog clips). Set to empty for all.")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if files exist")
    args = parser.parse_args()

    token = get_hf_token(args.token)
    if not token:
        print("[!] No Hugging Face token detected.")
        print("    Please provide your token using --token <HF_TOKEN>, set the HF_TOKEN environment variable,")
        print("    or run 'huggingface-cli login'.")
        sys.exit(1)

    session = args.session if args.session.strip() else None
    stream_extract_bones_seed(token, session=session, force=args.force)


if __name__ == "__main__":
    main()
