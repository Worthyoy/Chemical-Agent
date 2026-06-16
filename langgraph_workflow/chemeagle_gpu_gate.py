"""Cross-process GPU gate for ChemEagle local model calls."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


GPU_GATE_ENABLED_ENV = "CHEMEAGLE_GPU_GATE_ENABLED"
GPU_GATE_LOCK_DIR_ENV = "CHEMEAGLE_GPU_GATE_LOCK_DIR"
GPU_GATE_UTIL_THRESHOLD_ENV = "CHEMEAGLE_GPU_GATE_UTIL_THRESHOLD"
GPU_GATE_MEMORY_FREE_MIB_ENV = "CHEMEAGLE_GPU_GATE_MEMORY_FREE_MIB"
GPU_GATE_STABLE_SAMPLES_ENV = "CHEMEAGLE_GPU_GATE_STABLE_SAMPLES"
GPU_GATE_POLL_INTERVAL_ENV = "CHEMEAGLE_GPU_GATE_POLL_INTERVAL"
GPU_GATE_TIMEOUT_ENV = "CHEMEAGLE_GPU_GATE_TIMEOUT"
GPU_GATE_STALE_SECONDS_ENV = "CHEMEAGLE_GPU_GATE_STALE_SECONDS"

DEFAULT_UTIL_THRESHOLD = 20.0
DEFAULT_MEMORY_FREE_MIB = 2000.0
DEFAULT_STABLE_SAMPLES = 2
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_TIMEOUT_SECONDS = 0.0
DEFAULT_STALE_SECONDS = 6 * 60 * 60


def gpu_gate_enabled() -> bool:
    return os.environ.get(GPU_GATE_ENABLED_ENV, "").strip().casefold() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def gpu_gate_config() -> Dict[str, Any]:
    return {
        "enabled": gpu_gate_enabled(),
        "lock_dir": os.environ.get(GPU_GATE_LOCK_DIR_ENV),
        "util_threshold": env_float(GPU_GATE_UTIL_THRESHOLD_ENV, DEFAULT_UTIL_THRESHOLD),
        "memory_free_mib": env_float(GPU_GATE_MEMORY_FREE_MIB_ENV, DEFAULT_MEMORY_FREE_MIB),
        "stable_samples": max(1, env_int(GPU_GATE_STABLE_SAMPLES_ENV, DEFAULT_STABLE_SAMPLES)),
        "poll_interval": max(0.1, env_float(GPU_GATE_POLL_INTERVAL_ENV, DEFAULT_POLL_INTERVAL)),
        "timeout_seconds": max(0.0, env_float(GPU_GATE_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS)),
        "stale_seconds": max(60.0, env_float(GPU_GATE_STALE_SECONDS_ENV, DEFAULT_STALE_SECONDS)),
    }


def query_gpu_snapshot() -> Dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"status": "unavailable", "error": "nvidia-smi not found", "gpus": []}
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    if completed.returncode != 0:
        return {
            "status": "error",
            "error": (completed.stderr or completed.stdout or "").strip(),
            "gpus": [],
        }
    gpus: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(
                {
                    "index": int(float(parts[0])),
                    "util_percent": float(parts[1]),
                    "memory_free_mib": float(parts[2]),
                }
            )
        except ValueError:
            continue
    return {"status": "success" if gpus else "empty", "gpus": gpus}


def gpu_is_available(snapshot: Dict[str, Any], *, util_threshold: float, memory_free_mib: float) -> bool:
    gpus = snapshot.get("gpus") or []
    if not gpus:
        return True
    return any(
        gpu.get("util_percent", 100.0) <= util_threshold
        and gpu.get("memory_free_mib", 0.0) >= memory_free_mib
        for gpu in gpus
    )


@contextlib.contextmanager
def gpu_gate(operation: str) -> Iterator[Dict[str, Any]]:
    info = wait_for_gpu_gate(operation)
    try:
        yield info
    finally:
        release_gpu_gate(info)


def wait_for_gpu_gate(operation: str) -> Dict[str, Any]:
    config = gpu_gate_config()
    started_at = time.perf_counter()
    info: Dict[str, Any] = {
        "enabled": config["enabled"],
        "operation": operation,
        "status": "disabled",
        "wait_seconds": 0.0,
        "samples": 0,
    }
    if not config["enabled"]:
        return info
    lock_root = config.get("lock_dir")
    if not lock_root:
        info.update({"status": "disabled", "error": "missing lock dir"})
        return info

    lock_path = Path(lock_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_acquired = acquire_lock_file(lock_path, operation=operation, config=config)
    except PermissionError:
        fallback_path = fallback_lock_path(lock_path)
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        info["lock_fallback_from"] = str(lock_path)
        lock_path = fallback_path
        lock_acquired = acquire_lock_file(lock_path, operation=operation, config=config)
    if not lock_acquired:
        info.update(
            {
                "status": "timeout_continue",
                "wait_seconds": round(time.perf_counter() - started_at, 3),
                "error": "GPU gate lock timeout reached; continuing without lock",
            }
        )
        return info
    info.update({"status": "acquired", "lock_path": str(lock_path)})

    stable_count = 0
    last_snapshot: Optional[Dict[str, Any]] = None
    while True:
        snapshot = query_gpu_snapshot()
        last_snapshot = snapshot
        info["samples"] += 1
        if gpu_is_available(
            snapshot,
            util_threshold=config["util_threshold"],
            memory_free_mib=config["memory_free_mib"],
        ):
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= config["stable_samples"]:
            info["status"] = "ready"
            break
        elapsed = time.perf_counter() - started_at
        if config["timeout_seconds"] and elapsed >= config["timeout_seconds"]:
            info["status"] = "timeout_continue"
            info["error"] = "GPU gate timeout reached; continuing without availability confirmation"
            break
        time.sleep(config["poll_interval"])

    info["wait_seconds"] = round(time.perf_counter() - started_at, 3)
    info["last_snapshot"] = last_snapshot
    info["config"] = {
        "util_threshold": config["util_threshold"],
        "memory_free_mib": config["memory_free_mib"],
        "stable_samples": config["stable_samples"],
        "poll_interval": config["poll_interval"],
        "timeout_seconds": config["timeout_seconds"],
    }
    return info


def acquire_lock_file(lock_path: Path, *, operation: str, config: Dict[str, Any]) -> bool:
    started_at = time.perf_counter()
    while True:
        try:
            owner = {
                "pid": os.getpid(),
                "operation": operation,
                "created_at": datetime.now().isoformat(),
            }
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(owner, f, ensure_ascii=False)
            return True
        except FileExistsError:
            remove_stale_lock(lock_path, stale_seconds=config["stale_seconds"])
            if config["timeout_seconds"] and time.perf_counter() - started_at >= config["timeout_seconds"]:
                return False
            time.sleep(config["poll_interval"])


def fallback_lock_path(lock_path: Path) -> Path:
    digest = hashlib.sha1(str(lock_path).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"chemeagle_gpu_gate_{digest}.lock"


def remove_stale_lock(lock_path: Path, *, stale_seconds: float) -> None:
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return
    if age < stale_seconds:
        return
    try:
        lock_path.unlink()
    except OSError:
        pass


def release_gpu_gate(info: Dict[str, Any]) -> None:
    lock_path = info.get("lock_path")
    if not lock_path:
        return
    try:
        Path(lock_path).unlink()
    except OSError:
        pass
