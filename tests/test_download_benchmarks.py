"""Tests for the benchmark downloader (scripts/download_benchmarks.py); the network is mocked."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import pathlib
import socket
import zipfile

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "download_benchmarks",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "download_benchmarks.py")
dl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dl)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class _Resp:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _fake_sources(monkeypatch, served: dict[str, bytes], pins: dict[str, bytes] | None = None):
    """Point every source at fake URLs whose payloads are `served`; pin `pins` (default: served)."""
    pins = served if pins is None else pins
    rows = {"a.csv": b"x,y\n1,2\n", "b.csv": b"x,y\n3,4\n"}
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as z:
        for name, data in rows.items():
            z.writestr(f"power_consumption_upload/real_data/{name}", data)
        z.writestr("power_consumption_upload/synthetic_data/a.csv", b"ignored")
    served = {**served, "zip": zbuf.getvalue()}
    sources = {
        "shifts": {"url": "https://example.org/shifts.zip", "archive": "data/external/shifts_vessel/power_consumption_v2.zip",
                   "archive_sha256": _sha(served["zip"]),
                   "members": {f"power_consumption_upload/real_data/{n}": {
                       "path": f"data/external/shifts_vessel/v2/power_consumption_upload/real_data/{n}",
                       "sha256": _sha(d)} for n, d in rows.items()}},
        "fuelcast": {"files": {n: {"url": f"https://example.org/{n}", "path": f"data/external/fuelcast/{n}",
                                   "sha256": _sha(pins[n])} for n in ("P.parquet",)}},
    }
    monkeypatch.setattr(dl, "SOURCES", sources)
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        key = "zip" if url.endswith("shifts.zip") else url.rsplit("/", 1)[1]
        return _Resp(served[key])

    monkeypatch.setattr(dl, "urlopen", fake_urlopen)
    return calls


def test_layout_matches_stage_paths(tmp_path, monkeypatch):
    _fake_sources(monkeypatch, {"P.parquet": b"parquet-bytes"})
    dl.download("shifts", tmp_path, timeout_s=5)
    dl.download("fuelcast", tmp_path, timeout_s=5)
    real = tmp_path / "data/external/shifts_vessel/v2/power_consumption_upload/real_data"
    assert sorted(p.name for p in real.iterdir()) == ["a.csv", "b.csv"]
    assert (tmp_path / "data/external/fuelcast/P.parquet").read_bytes() == b"parquet-bytes"
    assert not (tmp_path / "data/external/shifts_vessel/v2/power_consumption_upload/synthetic_data").exists()


def test_layout_matches_the_paths_the_stages_read():
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    from benchmark_measured.common import FC_DIR, REAL
    shifts = {pathlib.Path(m["path"]).parent for m in dl.SOURCES["shifts"]["members"].values()}
    fuel = {pathlib.Path(f["path"]).parent for f in dl.SOURCES["fuelcast"]["files"].values()}
    assert shifts == {REAL} and fuel == {FC_DIR}


def test_checksum_mismatch_exits_1_and_removes_file(tmp_path, monkeypatch):
    _fake_sources(monkeypatch, {"P.parquet": b"tampered"}, pins={"P.parquet": b"original"})
    rc = dl.main(["--only", "fuelcast", "--dest", str(tmp_path), "--timeout", "5"])
    assert rc == 1
    assert not (tmp_path / "data/external/fuelcast/P.parquet").exists()


def test_existing_verified_file_not_downloaded_again(tmp_path, monkeypatch):
    calls = _fake_sources(monkeypatch, {"P.parquet": b"parquet-bytes"})
    dl.download("fuelcast", tmp_path, timeout_s=5)
    dl.download("shifts", tmp_path, timeout_s=5)
    n = len(calls)
    dl.download("fuelcast", tmp_path, timeout_s=5)
    dl.download("shifts", tmp_path, timeout_s=5)
    assert len(calls) == n


def test_timeout_is_reported(tmp_path, monkeypatch, capsys):
    _fake_sources(monkeypatch, {"P.parquet": b"x"})

    def slow(url, timeout):
        raise socket.timeout("timed out")

    monkeypatch.setattr(dl, "urlopen", slow)
    monkeypatch.setattr(dl, "RETRY_PAUSE_S", 0)
    rc = dl.main(["--only", "fuelcast", "--dest", str(tmp_path), "--timeout", "5"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "timed out" in err and str(dl.RETRIES) in err


def test_unknown_benchmark_is_refused(tmp_path):
    with pytest.raises(ValueError):
        dl.download("other", tmp_path, timeout_s=5)
