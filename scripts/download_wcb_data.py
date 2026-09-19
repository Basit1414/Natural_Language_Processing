#!/usr/bin/env python3
"""Download the pinned World Central Banks stance-label parquet splits."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import requests


BASE = "https://huggingface.co/datasets/gtfintechlab/all_annotated_sentences_25000/resolve/main/5768"
FILES = {
    "train": (
        "train-00000-of-00001.parquet",
        "3087b4c87872942a1699b241865138484dc13da3b0a540638b3654d6d6256b7f",
    ),
    "validation": (
        "val-00000-of-00001.parquet",
        "a0040d76fedb454ec5e0a6e2a5c00ef4ddd5b623584e9a830167f58772af557a",
    ),
    "test": (
        "test-00000-of-00001.parquet",
        "5974572ec5e523a5f4ae849b89f50e33cc517b46369829c854c5f54ed525b1c1",
    ),
}
TARGET = Path(__file__).resolve().parents[1] / "data" / "external" / "wcb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for local_name, (remote_name, expected_hash) in FILES.items():
        destination = TARGET / f"{local_name}.parquet"
        if destination.exists() and sha256(destination) == expected_hash:
            print(f"Verified existing {destination.name}")
            continue
        url = f"{BASE}/{remote_name}?download=true"
        with tempfile.NamedTemporaryFile(prefix=f"wcb-{local_name}-", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            print(f"Downloading {url}")
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            observed = sha256(temporary)
            if observed != expected_hash:
                raise RuntimeError(
                    f"Hash mismatch for {local_name}: expected {expected_hash}, received {observed}"
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"WCB splits ready in {TARGET}")


if __name__ == "__main__":
    main()
