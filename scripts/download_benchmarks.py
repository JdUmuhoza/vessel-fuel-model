"""Download the two public benchmarks from their official sources and verify them.

Shifts (Malinin et al., 2022): Zenodo record 7684813, licence CC BY-NC-SA 4.0.
FuelCast (Viga et al., 2026): Hugging Face dataset ``krohnedigital/FuelCast`` at a pinned
revision, licence CC BY-NC-ND 4.0.

Neither dataset is redistributed with this code. This script fetches them on the user's
machine, checks every file against the SHA-256 of the copy that produced the paper's
results, and places them where the experiment stages read them:

    data/external/shifts_vessel/v2/power_consumption_upload/real_data/{train,dev_in,dev_out}.csv
    data/external/fuelcast/{CPS_Poseidon,CPS_Triton,OSS_Ceto}.parquet

Usage:
    python scripts/download_benchmarks.py [--only shifts|fuelcast] [--dest .] [--timeout 120]
Exit codes: 0 every file present and verified, 1 a download failed or a checksum differs.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import socket
import sys
import time
import zipfile
from urllib.error import URLError
from urllib.request import urlopen

RETRIES = 3
RETRY_PAUSE_S = 5
_CHUNK = 1 << 20
_HF = "https://huggingface.co/datasets/krohnedigital/FuelCast/resolve/eb6a6ec011c1c9a2cbce21459e22be4c77ef84dd"
_REAL = "data/external/shifts_vessel/v2/power_consumption_upload/real_data"

# SHA-256 values computed on 2026-11-20 from the copies that produced the paper's results;
# they equal the files served by Zenodo (zip MD5 1010039484f827e7cf1ff6aa2f75bc94) and by
# Hugging Face (LFS SHA-256 of the pinned revision).
SOURCES: dict = {
    "shifts": {
        "url": "https://zenodo.org/api/records/7684813/files/power_consumption_upload.zip/content",
        "archive": "data/external/shifts_vessel/power_consumption_v2.zip",
        "archive_sha256": "160b852ec525d7bee1599455795457c26eceba59bfe36b9016cca82ed4a6c82f",
        "members": {
            "power_consumption_upload/real_data/train.csv": {
                "path": f"{_REAL}/train.csv",
                "sha256": "9ed9da98990d02236e06df34b6412f7241b3b39e6e7b641e5581f6dc6eef16f3"},
            "power_consumption_upload/real_data/dev_in.csv": {
                "path": f"{_REAL}/dev_in.csv",
                "sha256": "75419bfde8e0ffa238225e0af11347597d3aa4c788f05402f8f55f77c94599c3"},
            "power_consumption_upload/real_data/dev_out.csv": {
                "path": f"{_REAL}/dev_out.csv",
                "sha256": "fd38c26cd07ecec710fab65e1f81dd6a9f3133fea7e85805f542f1d428f766f4"},
        },
    },
    "fuelcast": {
        "files": {
            "CPS_Poseidon.parquet": {
                "url": f"{_HF}/CPS_Poseidon.parquet", "path": "data/external/fuelcast/CPS_Poseidon.parquet",
                "sha256": "fcc4fff3e86de3d02b034376429e4cd98c794befb4e8f610187643d52552571b"},
            "CPS_Triton.parquet": {
                "url": f"{_HF}/CPS_Triton.parquet", "path": "data/external/fuelcast/CPS_Triton.parquet",
                "sha256": "327581b0d35c756db207e6992a1d78f6f7e4f68dcfd299c77188912fdbd6aab8"},
            "OSS_Ceto.parquet": {
                "url": f"{_HF}/OSS_Ceto.parquet", "path": "data/external/fuelcast/OSS_Ceto.parquet",
                "sha256": "a81eefff2bb6bedbb85455e4fb408906a1de5356f0633fda450d7e794410b7ef"},
        },
    },
}


class DownloadError(RuntimeError):
    """A file could not be fetched or does not match its pinned checksum."""


def _sha256(path: pathlib.Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _verified(path: pathlib.Path, sha256: str) -> bool:
    """True when the file exists and matches the pinned checksum."""
    return path.is_file() and _sha256(path) == sha256


def _fetch(url: str, path: pathlib.Path, sha256: str, timeout_s: int) -> None:
    """Download ``url`` to ``path`` with bounded retries, then verify; remove it on mismatch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urlopen(url, timeout=timeout_s) as resp, open(tmp, "wb") as out:
                for block in iter(lambda: resp.read(_CHUNK), b""):
                    out.write(block)
            break
        except (socket.timeout, TimeoutError, URLError, OSError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            if attempt < RETRIES:
                time.sleep(RETRY_PAUSE_S)
    else:
        raise DownloadError(f"{url}: failed after {RETRIES} attempts ({last})")
    tmp.replace(path)
    if _sha256(path) != sha256:
        path.unlink()
        raise DownloadError(f"{path}: checksum differs from the pinned value; file removed")


def download(name: str, dest: pathlib.Path, timeout_s: int) -> pathlib.Path:
    """Fetch one benchmark into ``dest`` and verify every file.

    Parameters
    ----------
    name : str
        ``"shifts"`` or ``"fuelcast"``.
    dest : pathlib.Path
        Root under which the ``data/external/...`` layout is written (the repository root).
    timeout_s : int
        Per-request network timeout in seconds; each request is tried ``RETRIES`` times.

    Returns
    -------
    pathlib.Path
        The folder the experiment stages read for this benchmark.
    """
    if name not in SOURCES:
        raise ValueError(f"unknown benchmark {name!r}; expected one of {sorted(SOURCES)}")
    dest = pathlib.Path(dest)
    src = SOURCES[name]
    if name == "fuelcast":
        for f in src["files"].values():
            target = dest / f["path"]
            if not _verified(target, f["sha256"]):
                _fetch(f["url"], target, f["sha256"], timeout_s)
        return (dest / next(iter(src["files"].values()))["path"]).parent
    members = src["members"]
    if all(_verified(dest / m["path"], m["sha256"]) for m in members.values()):
        return (dest / next(iter(members.values()))["path"]).parent
    archive = dest / src["archive"]
    if not _verified(archive, src["archive_sha256"]):
        _fetch(src["url"], archive, src["archive_sha256"], timeout_s)
    with zipfile.ZipFile(archive) as z:
        for member, m in members.items():
            target = dest / m["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(member))   # explicit member names: no path from the archive is trusted
            if _sha256(target) != m["sha256"]:
                target.unlink()
                raise DownloadError(f"{target}: checksum differs from the pinned value; file removed")
    return (dest / next(iter(members.values()))["path"]).parent


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=sorted(SOURCES))
    ap.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args(argv)
    rc = 0
    for name in [a.only] if a.only else sorted(SOURCES):
        try:
            folder = download(name, a.dest, a.timeout)
            print(f"OK    {name:9s} {folder}")
        except DownloadError as exc:
            print(f"FAIL  {name:9s} {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
