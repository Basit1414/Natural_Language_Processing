#!/usr/bin/env python3
"""Download and verify the BIS speech archive snapshot used by this project."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests


URL = "https://www.bis.org/pages/download-central-bankers-speeches/speeches.zip"
ZIP_SHA256 = "a8f26d5f6b17c19aa754ed67281e0a98a0c93b0f00e355b4aa8e842143ca01da"
CSV_SHA256 = "78b9ea67c2d75dfe77bd08d437f8a48ab8ea7c4efdeebe611b43d694156ad319"
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def main() -> None:
    zip_target = ROOT / "speeches.zip"
    csv_target = ROOT / "BIS_speeches.csv"
    if csv_target.exists() and sha256(csv_target) == CSV_SHA256:
        print(f"Verified existing {csv_target.name} ({CSV_SHA256})")
        return

    with tempfile.TemporaryDirectory(prefix="bis-speeches-") as temp_name:
        temp = Path(temp_name)
        downloaded = temp / "speeches.zip"
        print(f"Downloading {URL}")
        download(URL, downloaded)
        observed_zip_hash = sha256(downloaded)
        if observed_zip_hash != ZIP_SHA256:
            raise RuntimeError(
                "The mutable BIS archive has changed since this analysis snapshot. "
                f"Expected {ZIP_SHA256}, received {observed_zip_hash}. Review the new snapshot "
                "before updating the pinned hashes."
            )
        with zipfile.ZipFile(downloaded) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise RuntimeError(f"ZIP CRC check failed for {bad_member}")
            extracted = temp / "speeches.csv"
            with archive.open("speeches.csv") as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
        observed_csv_hash = sha256(extracted)
        if observed_csv_hash != CSV_SHA256:
            raise RuntimeError(
                f"CSV hash mismatch: expected {CSV_SHA256}, received {observed_csv_hash}"
            )
        shutil.copy2(downloaded, zip_target)
        shutil.copy2(extracted, csv_target)
    print(f"Wrote and verified {csv_target}")


if __name__ == "__main__":
    main()

