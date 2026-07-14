#!/usr/bin/env python3
"""Runtime utilities for rawseq 1m feature-family evolution.

These helpers are deliberately small and dependency-light. They provide
optional matrix telemetry, bounded immutable-array caching, memory snapshots,
and atomic JSON/CSV writes without changing experiment semantics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def matrix_footprint(array: np.ndarray, context: dict[str, Any] | None = None) -> dict[str, Any]:
    arr = np.asarray(array)
    base = arr
    owns = bool(arr.flags.owndata)
    while getattr(base, "base", None) is not None and isinstance(base.base, np.ndarray):
        base = base.base
    return {
        **(context or {}),
        "shape": "x".join(str(x) for x in arr.shape),
        "ndim": int(arr.ndim),
        "dtype": str(arr.dtype),
        "estimated_bytes": int(arr.nbytes),
        "owns_memory": owns,
        "c_contiguous": bool(arr.flags.c_contiguous),
        "f_contiguous": bool(arr.flags.f_contiguous),
        "base_estimated_bytes": int(getattr(base, "nbytes", arr.nbytes)),
    }


def process_memory_snapshot() -> dict[str, Any]:
    out = {
        "process_private_bytes": math.nan,
        "process_working_set_bytes": math.nan,
        "system_commit_percent": math.nan,
        "system_committed_bytes": math.nan,
        "system_commit_limit_bytes": math.nan,
        "available_ram_bytes": math.nan,
    }
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        out["process_working_set_bytes"] = int(getattr(info, "rss", 0))
        out["process_private_bytes"] = int(getattr(info, "private", getattr(info, "rss", 0)))
        vm = psutil.virtual_memory()
        out["available_ram_bytes"] = int(getattr(vm, "available", 0))
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                committed = int(stat.ullTotalPageFile - stat.ullAvailPageFile)
                limit = int(stat.ullTotalPageFile)
                out["system_committed_bytes"] = committed
                out["system_commit_limit_bytes"] = limit
                out["system_commit_percent"] = float(committed / limit) if limit else math.nan
                out["available_ram_bytes"] = int(stat.ullAvailPhys)
        except Exception:
            pass
    return out


@dataclass
class MatrixTelemetry:
    enabled: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, array: np.ndarray, context: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self.rows.append(
            {
                "event": event,
                "created_at_monotonic": time.perf_counter(),
                **matrix_footprint(array, context),
                **process_memory_snapshot(),
            }
        )

    def write_csv(self, path: Path) -> None:
        if not self.rows:
            return
        write_csv_atomic(path, self.rows)


class BoundedContentCache:
    def __init__(self, max_entries: int = 128, max_bytes: int = 512 * 1024 * 1024):
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> np.ndarray | None:
        if key in self._items:
            self.hits += 1
            value = self._items.pop(key)
            self._items[key] = value
            return value
        self.misses += 1
        return None

    def put(self, key: str, value: np.ndarray) -> None:
        if self.max_entries <= 0 or self.max_bytes <= 0:
            return
        arr = np.asarray(value)
        if key in self._items:
            old = self._items.pop(key)
            self._bytes -= int(old.nbytes)
        self._items[key] = arr
        self._bytes += int(arr.nbytes)
        while self._items and (len(self._items) > self.max_entries or self._bytes > self.max_bytes):
            _, old = self._items.popitem(last=False)
            self._bytes -= int(old.nbytes)
            self.evictions += 1

    def stats(self) -> dict[str, Any]:
        return {
            "cache_entries": len(self._items),
            "cache_bytes": int(self._bytes),
            "cache_hits": int(self.hits),
            "cache_misses": int(self.misses),
            "cache_evictions": int(self.evictions),
            "cache_max_entries": int(self.max_entries),
            "cache_max_bytes": int(self.max_bytes),
        }


def frame_identity(frame: pd.DataFrame) -> dict[str, Any]:
    if "timestamp_ms" in frame.columns and len(frame):
        ts = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
        return {
            "rows": int(len(frame)),
            "first_timestamp_ms": int(ts.iloc[0]) if pd.notna(ts.iloc[0]) else "",
            "last_timestamp_ms": int(ts.iloc[-1]) if pd.notna(ts.iloc[-1]) else "",
            "timestamp_sha256": stable_hash(ts.fillna(-1).astype(np.int64).tolist()),
        }
    return {"rows": int(len(frame)), "columns": list(frame.columns)}


def matrix_cache_key(kind: str, frame: pd.DataFrame, columns: list[str], dtype: str, semantic_contract: dict[str, Any] | None = None) -> str:
    return stable_hash(
        {
            "kind": kind,
            "frame": frame_identity(frame),
            "columns": list(columns),
            "dtype": str(dtype),
            "semantic_contract": semantic_contract or {},
        }
    )


def extract_frame_matrix(
    frame: pd.DataFrame,
    columns: list[str],
    dtype: np.dtype | type = np.float64,
    cache: BoundedContentCache | None = None,
    telemetry: MatrixTelemetry | None = None,
    context: dict[str, Any] | None = None,
    semantic_contract: dict[str, Any] | None = None,
) -> np.ndarray:
    dtype_obj = np.dtype(dtype)
    key = matrix_cache_key("frame_matrix", frame, columns, dtype_obj.name, semantic_contract)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            if telemetry:
                telemetry.record("cache_hit_frame_matrix", cached, {**(context or {}), "cache_key": key})
            return cached
    values = frame.reindex(columns=columns).to_numpy(dtype=dtype_obj, copy=False)
    values = np.ascontiguousarray(values)
    if telemetry:
        telemetry.record("extract_frame_matrix", values, {**(context or {}), "cache_key": key})
    if cache is not None:
        cache.put(key, values)
    return values


def extract_series_vector(
    series: pd.Series,
    dtype: np.dtype | type = np.float64,
    cache: BoundedContentCache | None = None,
    telemetry: MatrixTelemetry | None = None,
    context: dict[str, Any] | None = None,
    semantic_contract: dict[str, Any] | None = None,
) -> np.ndarray:
    dtype_obj = np.dtype(dtype)
    key = stable_hash(
        {
            "kind": "series_vector",
            "name": str(series.name),
            "rows": int(len(series)),
            "dtype": dtype_obj.name,
            "index_sha256": stable_hash(list(series.index.astype(str))),
            "semantic_contract": semantic_contract or {},
        }
    )
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            if telemetry:
                telemetry.record("cache_hit_series_vector", cached, {**(context or {}), "cache_key": key})
            return cached
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=dtype_obj, copy=False)
    values = np.ascontiguousarray(values)
    if telemetry:
        telemetry.record("extract_series_vector", values, {**(context or {}), "cache_key": key})
    if cache is not None:
        cache.put(key, values)
    return values


def take_rows(array: np.ndarray, indices: np.ndarray, telemetry: MatrixTelemetry | None = None, context: dict[str, Any] | None = None) -> np.ndarray:
    out = np.take(array, indices, axis=0)
    if telemetry:
        telemetry.record("take_rows", out, context)
    return out


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".t_{uuid.uuid4().hex[:8]}"
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".t_{uuid.uuid4().hex[:8]}"
    fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".t_{uuid.uuid4().hex[:8]}"
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def file_identity(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    except FileNotFoundError:
        return {"path": str(path), "exists": False, "size_bytes": 0, "mtime_ns": 0}


def frame_array_manifest(frame: pd.DataFrame, kind: str) -> dict[str, Any]:
    numeric = frame.select_dtypes(include=[np.number])
    values = np.ascontiguousarray(numeric.to_numpy(dtype=np.float64, copy=False)) if len(numeric.columns) else np.empty((len(frame), 0), dtype=np.float64)
    return {
        "kind": kind,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "numeric_columns": list(numeric.columns),
        "numeric_shape": list(values.shape),
        "numeric_dtype": str(values.dtype),
        "numeric_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "column_order_sha256": stable_hash(list(frame.columns)),
    }


def symbol_stage_manifest(symbol: str, data: Any) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "candles": frame_array_manifest(data.candles, "candles"),
        "features": frame_array_manifest(data.features, "features"),
        "targets": frame_array_manifest(data.targets, "targets"),
        "folds": data.folds,
        "folds_sha256": stable_hash(data.folds),
        "available_families": sorted(str(x) for x in data.available_families),
    }


def stage_manifest(by_symbol: dict[str, Any], feature_audit_rows: list[dict[str, Any]], target_manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "symbols": {symbol: symbol_stage_manifest(symbol, data) for symbol, data in sorted(by_symbol.items())},
        "feature_audit_rows": int(len(feature_audit_rows)),
        "target_manifest_rows": int(len(target_manifest_rows)),
        "feature_audit_sha256": stable_hash(feature_audit_rows),
        "target_manifest_sha256": stable_hash(target_manifest_rows),
    }


class StagePreparationCache:
    """Persistent, contract-validated cache for expensive per-stage data prep."""

    schema_version = "rawseq_1m_stage_preparation_cache_v1"

    def __init__(self, root: Path, max_entries: int = 24):
        self.root = Path(root)
        self.max_entries = max(1, int(max_entries))
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.rejected = 0
        self.evictions = 0
        self.events: list[dict[str, Any]] = []

    def key_for_contract(self, contract: dict[str, Any]) -> str:
        return stable_hash({"schema_version": self.schema_version, "contract": contract})

    def manifest_path(self, key: str) -> Path:
        return self.root / f"{key}.manifest.json"

    def payload_path(self, key: str) -> Path:
        return self.root / f"{key}.payload.pkl"

    def record_event(self, event: str, key: str, **fields: Any) -> None:
        self.events.append(
            {
                "event": event,
                "cache_key": key,
                "created_at_monotonic": time.perf_counter(),
                **fields,
                **process_memory_snapshot(),
            }
        )

    def load(self, contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | None:
        key = self.key_for_contract(contract)
        manifest_path = self.manifest_path(key)
        payload_path = self.payload_path(key)
        if not manifest_path.exists() or not payload_path.exists():
            self.misses += 1
            self.record_event("stage_cache_miss", key, reason="missing_manifest_or_payload")
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != self.schema_version:
                raise ValueError("schema_version_mismatch")
            if manifest.get("cache_key") != key:
                raise ValueError("cache_key_mismatch")
            if manifest.get("contract_hash") != stable_hash(contract):
                raise ValueError("contract_hash_mismatch")
            if not manifest.get("cache_complete"):
                raise ValueError("cache_incomplete")
            payload_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
            if payload_hash != manifest.get("payload_sha256"):
                raise ValueError("payload_hash_mismatch")
            payload = pickle.loads(payload_path.read_bytes())
            if stable_hash(payload.get("stage_manifest", {})) != manifest.get("stage_manifest_sha256"):
                raise ValueError("stage_manifest_hash_mismatch")
        except Exception as exc:
            self.rejected += 1
            self.record_event("stage_cache_rejected", key, reason=repr(exc), manifest_path=str(manifest_path), payload_path=str(payload_path))
            return None
        self.hits += 1
        self.record_event("stage_cache_hit", key, manifest_path=str(manifest_path), payload_path=str(payload_path))
        return payload["by_symbol"], payload["feature_audit_rows"], payload["target_manifest_rows"], manifest

    def write(
        self,
        contract: dict[str, Any],
        by_symbol: dict[str, Any],
        feature_audit_rows: list[dict[str, Any]],
        target_manifest_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        key = self.key_for_contract(contract)
        manifest_path = self.manifest_path(key)
        payload_path = self.payload_path(key)
        stage_meta = stage_manifest(by_symbol, feature_audit_rows, target_manifest_rows)
        payload = {
            "schema_version": self.schema_version,
            "contract": contract,
            "by_symbol": by_symbol,
            "feature_audit_rows": feature_audit_rows,
            "target_manifest_rows": target_manifest_rows,
            "stage_manifest": stage_meta,
        }
        payload_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        manifest = {
            "schema_version": self.schema_version,
            "cache_key": key,
            "contract_hash": stable_hash(contract),
            "cache_complete": True,
            "payload_path": str(payload_path),
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "payload_bytes": int(len(payload_bytes)),
            "stage_manifest": stage_meta,
            "stage_manifest_sha256": stable_hash(stage_meta),
            "written_at_monotonic": time.perf_counter(),
        }
        write_bytes_atomic(payload_path, payload_bytes)
        write_json_atomic(manifest_path, manifest)
        self.writes += 1
        self.record_event("stage_cache_write", key, manifest_path=str(manifest_path), payload_path=str(payload_path), payload_bytes=len(payload_bytes))
        self.enforce_limit()
        return manifest

    def enforce_limit(self) -> None:
        manifests = sorted(self.root.glob("*.manifest.json"), key=lambda p: p.stat().st_mtime_ns if p.exists() else 0)
        while len(manifests) > self.max_entries:
            old = manifests.pop(0)
            try:
                payload = json.loads(old.read_text(encoding="utf-8")).get("payload_path", "")
                if payload:
                    Path(payload).unlink(missing_ok=True)
                old.unlink(missing_ok=True)
                self.evictions += 1
            except Exception:
                break

    def stats(self) -> dict[str, Any]:
        return {
            "stage_cache_root": str(self.root),
            "stage_cache_hits": int(self.hits),
            "stage_cache_misses": int(self.misses),
            "stage_cache_writes": int(self.writes),
            "stage_cache_rejected": int(self.rejected),
            "stage_cache_evictions": int(self.evictions),
            "stage_cache_max_entries": int(self.max_entries),
        }

    def write_events_csv(self, path: Path) -> None:
        if self.events:
            write_csv_atomic(path, self.events)


@dataclass
class HeartbeatEmitter:
    enabled: bool = True
    rows: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, phase: str, **fields: Any) -> dict[str, Any]:
        row = {
            "event": "heartbeat",
            "phase": phase,
            "created_at_monotonic": time.perf_counter(),
            **fields,
            **process_memory_snapshot(),
        }
        self.rows.append(row)
        if self.enabled:
            parts = ["HEARTBEAT", f"phase={phase}"]
            for key in ["stage_id", "stage_name", "symbol", "candidate_key", "target_name", "feature_group", "model", "fold", "cache_status"]:
                if key in row:
                    parts.append(f"{key}={row.get(key)}")
            parts.append(f"system_commit_percent={row.get('system_commit_percent')}")
            print(" ".join(parts), flush=True)
        return row

    def write_csv(self, path: Path) -> None:
        if self.rows:
            write_csv_atomic(path, self.rows)


class CheckpointStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.records_dir = self.root / "candidate_records"
        self.stage_dir = self.root / "stage_records"
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_contract_path(self) -> Path:
        return self.root / "run_contract.json"

    def write_or_validate_run_contract(self, contract: dict[str, Any], resume: bool = False) -> str:
        contract_hash = stable_hash(contract)
        payload = {"contract": contract, "contract_hash": contract_hash}
        if self.run_contract_path.exists():
            existing = json.loads(self.run_contract_path.read_text(encoding="utf-8"))
            if existing.get("contract_hash") != contract_hash:
                raise ValueError("checkpoint run contract mismatch")
            return contract_hash
        if resume:
            raise FileNotFoundError(f"missing checkpoint run contract: {self.run_contract_path}")
        write_json_atomic(self.run_contract_path, payload)
        return contract_hash

    def candidate_record_path(self, stage_id: int, candidate_key: str) -> Path:
        return self.records_dir / f"stage_{int(stage_id):03d}_{candidate_key}.json"

    def write_candidate_record(self, stage_id: int, candidate_key: str, payload: dict[str, Any]) -> Path:
        record = {
            **payload,
            "stage_id": int(stage_id),
            "candidate_key": str(candidate_key),
            "checkpoint_complete": True,
            "checkpoint_written_at_monotonic": time.perf_counter(),
        }
        path = self.candidate_record_path(stage_id, candidate_key)
        write_json_atomic(path, record)
        return path

    def fold_record_path(self, stage_id: int, candidate_key: str, fold_id: int | str) -> Path:
        safe_fold = str(fold_id).replace(os.sep, "_")
        return self.records_dir / f"stage_{int(stage_id):03d}_{candidate_key}_fold_{safe_fold}.json"

    def write_fold_record(self, stage_id: int, candidate_key: str, fold_id: int | str, payload: dict[str, Any]) -> Path:
        record = {
            **payload,
            "stage_id": int(stage_id),
            "candidate_key": str(candidate_key),
            "fold_id": fold_id,
            "checkpoint_granularity": "fold",
            "checkpoint_complete": True,
            "checkpoint_written_at_monotonic": time.perf_counter(),
        }
        path = self.fold_record_path(stage_id, candidate_key, fold_id)
        write_json_atomic(path, record)
        return path

    def read_candidate_record(self, stage_id: int, candidate_key: str) -> dict[str, Any] | None:
        path = self.candidate_record_path(stage_id, candidate_key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"corrupt checkpoint record {path}: {exc}") from exc
        if not payload.get("checkpoint_complete"):
            return None
        return payload

    def completed_candidate_keys(self, stage_id: int) -> set[str]:
        out: set[str] = set()
        for path in sorted(self.records_dir.glob(f"stage_{int(stage_id):03d}_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"corrupt checkpoint record {path}: {exc}") from exc
            if payload.get("checkpoint_complete") and payload.get("candidate_key"):
                out.add(str(payload["candidate_key"]))
        return out

    def write_stage_record(self, stage_id: int, payload: dict[str, Any]) -> Path:
        path = self.stage_dir / f"stage_{int(stage_id):03d}_aggregation.json"
        write_json_atomic(path, {**payload, "stage_id": int(stage_id), "checkpoint_complete": True})
        return path

    def load_completed_stage_rows(self, stage_id: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        coverage: list[dict[str, Any]] = []
        for key in sorted(self.completed_candidate_keys(stage_id)):
            payload = self.read_candidate_record(stage_id, key)
            if not payload:
                continue
            rows.extend(payload.get("stage_rows", []))
            coverage.extend(payload.get("coverage_rows", []))
        return rows, coverage


@dataclass
class MemoryGuard:
    policy: str = "fail_closed"
    warn_system_commit_fraction: float = 0.80
    pause_system_commit_fraction: float = 0.90
    fail_system_commit_fraction: float = 0.92
    warn_process_private_bytes: int = 0
    fail_process_private_bytes: int = 0
    snapshot_provider: Any = process_memory_snapshot

    @classmethod
    def from_env(cls, policy: str) -> "MemoryGuard":
        return cls(
            policy=policy,
            warn_system_commit_fraction=float(os.getenv("RAWSEQ_EVOLVE_MEMORY_WARN_SYSTEM_COMMIT_FRACTION", "0.80")),
            pause_system_commit_fraction=float(os.getenv("RAWSEQ_EVOLVE_MEMORY_PAUSE_SYSTEM_COMMIT_FRACTION", "0.90")),
            fail_system_commit_fraction=float(os.getenv("RAWSEQ_EVOLVE_MEMORY_FAIL_SYSTEM_COMMIT_FRACTION", "0.92")),
            warn_process_private_bytes=int(os.getenv("RAWSEQ_EVOLVE_MEMORY_WARN_PRIVATE_BYTES", "0")),
            fail_process_private_bytes=int(os.getenv("RAWSEQ_EVOLVE_MEMORY_FAIL_PRIVATE_BYTES", "0")),
        )

    def check(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        snap = self.snapshot_provider()
        status = "MEMORY_GUARD_OK"
        reason = ""
        commit = snap.get("system_commit_percent")
        private = snap.get("process_private_bytes")
        warn = False
        pause = False
        fail = False
        if isinstance(commit, (int, float)) and math.isfinite(commit):
            warn = warn or commit >= self.warn_system_commit_fraction
            pause = pause or commit >= self.pause_system_commit_fraction
            fail = fail or commit >= self.fail_system_commit_fraction
            if fail:
                reason = f"system_commit_percent>={self.fail_system_commit_fraction}"
            elif pause:
                reason = f"system_commit_percent>={self.pause_system_commit_fraction}"
            elif warn:
                reason = f"system_commit_percent>={self.warn_system_commit_fraction}"
        if isinstance(private, (int, float)) and math.isfinite(private):
            if self.fail_process_private_bytes and private >= self.fail_process_private_bytes:
                fail = True
                reason = f"process_private_bytes>={self.fail_process_private_bytes}"
            elif self.warn_process_private_bytes and private >= self.warn_process_private_bytes:
                warn = True
                reason = reason or f"process_private_bytes>={self.warn_process_private_bytes}"
        should_stop = False
        if fail:
            status = "MEMORY_GUARD_FAILED"
            should_stop = True
        elif pause and self.policy == "checkpoint_and_pause":
            status = "MEMORY_GUARD_PAUSED"
            should_stop = True
        elif warn:
            status = "MEMORY_GUARD_WARNING"
        return {
            **(context or {}),
            **snap,
            "memory_guard_policy": self.policy,
            "memory_guard_status": status,
            "memory_guard_reason": reason,
            "memory_guard_should_stop": should_stop,
        }


def progress_payload(context: dict[str, Any], cache_stats: dict[str, Any] | None = None, checkpoint_path: str = "") -> dict[str, Any]:
    return {
        **context,
        **(cache_stats or {}),
        **process_memory_snapshot(),
        "checkpoint_path": checkpoint_path,
        "progress_created_at_monotonic": time.perf_counter(),
    }


def format_progress_line(payload: dict[str, Any]) -> str:
    fields = [
        "stage_index",
        "stage_total",
        "candidate_index",
        "candidate_total",
        "candidate_key",
        "target_name",
        "feature_group",
        "model",
        "model_seed",
        "fold",
        "elapsed_seconds",
        "recent_candidate_rate_per_sec",
        "estimated_remaining_seconds",
        "cache_hits",
        "cache_misses",
        "process_private_bytes",
        "process_working_set_bytes",
        "system_commit_percent",
        "checkpoint_path",
    ]
    return "PROGRESS " + " ".join(f"{field}={payload.get(field, '')}" for field in fields)
