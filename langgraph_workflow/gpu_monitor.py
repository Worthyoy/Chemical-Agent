"""GPU utilization sampling for the LangGraph workflow."""

from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


GPU_SAMPLE_FIELDS = [
    "timestamp",
    "elapsed_seconds",
    "gpu_index",
    "memory_used_mib",
    "gpu_util_percent",
    "power_w",
    "sample_error",
]


def _parse_gpu_line(raw: str, fallback_index: int) -> Dict[str, Any]:
    parts = [part.strip() for part in raw.strip().split(",")]
    if len(parts) == 4:
        index_raw, memory_raw, util_raw, power_raw = parts
    elif len(parts) == 3:
        index_raw = str(fallback_index)
        memory_raw, util_raw, power_raw = parts
    else:
        raise ValueError(f"Unexpected nvidia-smi output: {raw!r}")
    return {
        "gpu_index": int(float(index_raw)),
        "memory_used_mib": int(float(memory_raw)),
        "gpu_util_percent": int(float(util_raw)),
        "power_w": float(power_raw),
    }


def sample_gpus() -> List[Dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi returned no GPU rows.")
    return [_parse_gpu_line(line, index) for index, line in enumerate(lines)]


def _valid_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        sample
        for sample in samples
        if sample.get("sample_error") in (None, "")
        and sample.get("gpu_util_percent") not in (None, "")
    ]


def summarize_samples(samples: List[Dict[str, Any]], errors: List[str]) -> Dict[str, Any]:
    valid = _valid_samples(samples)
    if not valid:
        return {
            "status": "no_valid_samples",
            "sample_count": len(samples),
            "valid_sample_count": 0,
            "error_count": len(errors),
            "errors": errors[-20:],
            "gpu_count": 0,
            "duration_seconds": 0,
            "max_gpu_util_percent": None,
            "avg_gpu_util_percent": None,
            "max_memory_used_mib": None,
            "max_power_w": None,
            "by_gpu": {},
        }

    by_gpu: Dict[str, Dict[str, Any]] = {}
    for sample in valid:
        key = str(sample["gpu_index"])
        bucket = by_gpu.setdefault(
            key,
            {
                "sample_count": 0,
                "max_gpu_util_percent": 0,
                "gpu_util_sum": 0,
                "max_memory_used_mib": 0,
                "max_power_w": 0.0,
            },
        )
        util = int(sample["gpu_util_percent"])
        memory = int(sample["memory_used_mib"])
        power = float(sample["power_w"])
        bucket["sample_count"] += 1
        bucket["gpu_util_sum"] += util
        bucket["max_gpu_util_percent"] = max(bucket["max_gpu_util_percent"], util)
        bucket["max_memory_used_mib"] = max(bucket["max_memory_used_mib"], memory)
        bucket["max_power_w"] = max(bucket["max_power_w"], power)

    for bucket in by_gpu.values():
        count = max(1, int(bucket["sample_count"]))
        bucket["avg_gpu_util_percent"] = round(bucket.pop("gpu_util_sum") / count, 2)
        bucket["max_power_w"] = round(bucket["max_power_w"], 2)

    elapsed_values = [float(sample["elapsed_seconds"]) for sample in valid]
    util_values = [int(sample["gpu_util_percent"]) for sample in valid]
    memory_values = [int(sample["memory_used_mib"]) for sample in valid]
    power_values = [float(sample["power_w"]) for sample in valid]
    return {
        "status": "success",
        "sample_count": len(samples),
        "valid_sample_count": len(valid),
        "error_count": len(errors),
        "errors": errors[-20:],
        "gpu_count": len(by_gpu),
        "duration_seconds": round(max(elapsed_values), 3),
        "max_gpu_util_percent": max(util_values),
        "avg_gpu_util_percent": round(sum(util_values) / len(util_values), 2),
        "max_memory_used_mib": max(memory_values),
        "max_power_w": round(max(power_values), 2),
        "by_gpu": by_gpu,
    }


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_placeholder(path: Path, title: str, message: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gpu_timeline(samples: List[Dict[str, Any]], png_path: Path) -> None:
    valid = _valid_samples(samples)
    if not valid:
        _save_placeholder(
            png_path,
            "GPU Timeline",
            "No valid GPU samples were recorded. Check nvidia-smi availability.",
        )
        return

    plt = _setup_matplotlib()
    fig, ax_util = plt.subplots(figsize=(13, 5.8))
    ax_other = ax_util.twinx()
    colors = ["#2563eb", "#7c3aed", "#0891b2", "#dc2626", "#4b5563", "#ca8a04"]

    for line_index, gpu_index in enumerate(sorted({sample["gpu_index"] for sample in valid})):
        gpu_samples = [sample for sample in valid if sample["gpu_index"] == gpu_index]
        color = colors[line_index % len(colors)]
        elapsed = [float(sample["elapsed_seconds"]) for sample in gpu_samples]
        util = [int(sample["gpu_util_percent"]) for sample in gpu_samples]
        memory = [int(sample["memory_used_mib"]) for sample in gpu_samples]
        power = [float(sample["power_w"]) for sample in gpu_samples]
        ax_util.plot(elapsed, util, color=color, label=f"GPU {gpu_index} Util (%)", linewidth=1.9)
        ax_other.plot(elapsed, memory, color=color, linestyle="--", label=f"GPU {gpu_index} Memory (MiB)", linewidth=1.4)
        ax_other.plot(elapsed, power, color=color, linestyle=":", label=f"GPU {gpu_index} Power (W)", linewidth=1.2)

    ax_util.set_xlabel("Elapsed Time (s)")
    ax_util.set_ylabel("GPU Utilization (%)", color="#2563eb")
    ax_util.set_ylim(0, 100)
    ax_util.tick_params(axis="y", labelcolor="#2563eb")
    ax_util.grid(True, axis="y", alpha=0.25)
    ax_other.set_ylabel("Memory (MiB) / Power (W)", color="#f97316")
    ax_other.tick_params(axis="y", labelcolor="#f97316")

    lines_1, labels_1 = ax_util.get_legend_handles_labels()
    lines_2, labels_2 = ax_other.get_legend_handles_labels()
    ax_util.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", fontsize=8)
    fig.suptitle("Pipeline GPU Timeline")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


class GpuMonitor:
    def __init__(
        self,
        *,
        csv_path: Path,
        report_path: Path,
        png_path: Optional[Path],
        interval_seconds: float = 1.0,
    ):
        if interval_seconds <= 0:
            raise ValueError("GPU monitor interval must be greater than 0.")
        self.csv_path = Path(csv_path)
        self.report_path = Path(report_path)
        self.png_path = Path(png_path) if png_path is not None else None
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_perf: Optional[float] = None
        self.samples: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self._csv_lock = threading.Lock()

    def start(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        if self.png_path is not None:
            self.png_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_csv_header()
        self._started_perf = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2))
        return self.write_outputs()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.interval_seconds)

    def _sample_once(self) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        elapsed = 0.0
        if self._started_perf is not None:
            elapsed = time.perf_counter() - self._started_perf
        new_samples: List[Dict[str, Any]] = []
        try:
            rows = sample_gpus()
            for row in rows:
                sample = {
                    "timestamp": timestamp,
                    "elapsed_seconds": round(elapsed, 3),
                    "gpu_index": row["gpu_index"],
                    "memory_used_mib": row["memory_used_mib"],
                    "gpu_util_percent": row["gpu_util_percent"],
                    "power_w": round(float(row["power_w"]), 2),
                    "sample_error": "",
                }
                new_samples.append(sample)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.errors.append(message)
            new_samples.append(
                {
                    "timestamp": timestamp,
                    "elapsed_seconds": round(elapsed, 3),
                    "gpu_index": "",
                    "memory_used_mib": "",
                    "gpu_util_percent": "",
                    "power_w": "",
                    "sample_error": message,
                }
            )
        self.samples.extend(new_samples)
        self._append_csv_rows(new_samples)

    def _write_csv_header(self) -> None:
        with self._csv_lock:
            with self.csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=GPU_SAMPLE_FIELDS)
                writer.writeheader()

    def _append_csv_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with self._csv_lock:
            with self.csv_path.open("a", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=GPU_SAMPLE_FIELDS)
                writer.writerows(rows)

    def write_outputs(self) -> Dict[str, Any]:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._csv_lock:
            with self.csv_path.open("w", encoding="utf-8", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=GPU_SAMPLE_FIELDS)
                writer.writeheader()
                writer.writerows(self.samples)

        summary = summarize_samples(self.samples, self.errors)
        summary["csv_path"] = str(self.csv_path)
        summary["report_path"] = str(self.report_path)
        summary["png_path"] = str(self.png_path) if self.png_path is not None else None
        summary["plot_enabled"] = self.png_path is not None

        if self.png_path is not None:
            try:
                plot_gpu_timeline(self.samples, self.png_path)
            except Exception as exc:
                summary["plot_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

        with self.report_path.open("w", encoding="utf-8") as report_file:
            json.dump(summary, report_file, ensure_ascii=False, indent=2)
        return summary
