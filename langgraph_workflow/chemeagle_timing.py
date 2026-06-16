"""Fine-grained timing for ChemEagle subprocess execution."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    from chemeagle_gpu_gate import gpu_gate, gpu_gate_enabled
except ImportError:
    gpu_gate = None

    def gpu_gate_enabled() -> bool:
        return False


TIMING_EVENTS_ENV = "CHEMEAGLE_TIMING_EVENTS_PATH"
TIMING_REPORT_ENV = "CHEMEAGLE_TIMING_REPORT_PATH"
TIMING_RUN_ID_ENV = "CHEMEAGLE_TIMING_RUN_ID"
TIMING_PDF_NAME_ENV = "CHEMEAGLE_TIMING_PDF_NAME"
TIMING_IMAGE_NAME_ENV = "CHEMEAGLE_TIMING_IMAGE_NAME"
TIMING_SELECTED_AREA_ENV = "CHEMEAGLE_TIMING_SELECTED_AREA"

_image_name_var = contextvars.ContextVar("chemeagle_timing_image_name", default=None)
_selected_area_var = contextvars.ContextVar("chemeagle_timing_selected_area", default=None)
_operation_var = contextvars.ContextVar("chemeagle_timing_operation", default=None)
_lock = threading.Lock()
_openai_patched = False
GPU_GATED_OPERATIONS = {
    "rxnim.predict_image_file",
    "chemietoolkit.extract_molecule_corefs_from_figures",
    "visualheist.page_detection",
}


@contextlib.contextmanager
def timed_event(
    operation: str,
    *,
    stage: str = "chemeagle",
    pdf_name: Optional[str] = None,
    image_name: Optional[str] = None,
    selected_area: Optional[str] = None,
    model: Optional[str] = None,
    event_kind: str = "span",
    **extra: Any,
) -> Iterator[None]:
    status = "success"
    error_type = None
    error = None
    operation_token = _operation_var.set(operation)
    started_at = None
    try:
        with gpu_gate_context(operation):
            started_at = time.perf_counter()
            yield
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        error = str(exc)
        raise
    finally:
        record_event(
            operation=operation,
            stage=stage,
            pdf_name=pdf_name,
            image_name=image_name,
            selected_area=selected_area,
            model=model,
            status=status,
            elapsed_seconds=time.perf_counter() - (started_at or time.perf_counter()),
            event_kind=event_kind,
            error_type=error_type,
            error=error,
            **extra,
        )
        _operation_var.reset(operation_token)


@contextlib.contextmanager
def image_timing_context(image_name: Optional[str]) -> Iterator[None]:
    token = _image_name_var.set(image_name)
    previous = os.environ.get(TIMING_IMAGE_NAME_ENV)
    if image_name:
        os.environ[TIMING_IMAGE_NAME_ENV] = image_name
    try:
        yield
    finally:
        _image_name_var.reset(token)
        if previous is None:
            os.environ.pop(TIMING_IMAGE_NAME_ENV, None)
        else:
            os.environ[TIMING_IMAGE_NAME_ENV] = previous


@contextlib.contextmanager
def selected_area_timing_context(selected_area: Optional[str]) -> Iterator[None]:
    token = _selected_area_var.set(selected_area)
    previous = os.environ.get(TIMING_SELECTED_AREA_ENV)
    if selected_area:
        os.environ[TIMING_SELECTED_AREA_ENV] = selected_area
    try:
        yield
    finally:
        _selected_area_var.reset(token)
        if previous is None:
            os.environ.pop(TIMING_SELECTED_AREA_ENV, None)
        else:
            os.environ[TIMING_SELECTED_AREA_ENV] = previous


def current_image_name() -> Optional[str]:
    return _image_name_var.get() or os.environ.get(TIMING_IMAGE_NAME_ENV)


def current_selected_area() -> Optional[str]:
    return _selected_area_var.get() or os.environ.get(TIMING_SELECTED_AREA_ENV)


def current_operation() -> Optional[str]:
    return _operation_var.get()


def record_event(
    *,
    operation: str,
    elapsed_seconds: float,
    stage: str = "chemeagle",
    pdf_name: Optional[str] = None,
    image_name: Optional[str] = None,
    selected_area: Optional[str] = None,
    model: Optional[str] = None,
    status: str = "success",
    event_kind: Optional[str] = None,
    error_type: Optional[str] = None,
    error: Optional[str] = None,
    **extra: Any,
) -> None:
    events_path = os.environ.get(TIMING_EVENTS_ENV)
    if not events_path:
        return
    record: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "run_id": os.environ.get(TIMING_RUN_ID_ENV),
        "stage": stage,
        "operation": operation,
        "pdf_name": pdf_name or os.environ.get(TIMING_PDF_NAME_ENV),
        "image_name": image_name or current_image_name(),
        "selected_area": selected_area or current_selected_area(),
        "model": model,
        "status": status,
        "event_kind": normalize_event_kind(event_kind, operation),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    parent_operation = current_operation()
    if parent_operation and parent_operation != operation:
        record["parent_operation"] = parent_operation
    if error_type:
        record["error_type"] = error_type
    if error:
        record["error"] = error
    for key, value in extra.items():
        if value is not None:
            record[key] = value
    append_jsonl(Path(events_path), record)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def install_chemeagle_timing(
    events_path: Path | str,
    *,
    report_path: Optional[Path | str] = None,
    run_id: Optional[str] = None,
    pdf_name: Optional[str] = None,
) -> None:
    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ[TIMING_EVENTS_ENV] = str(events_path)
    if report_path:
        os.environ[TIMING_REPORT_ENV] = str(report_path)
    if run_id:
        os.environ[TIMING_RUN_ID_ENV] = run_id
    if pdf_name:
        os.environ[TIMING_PDF_NAME_ENV] = pdf_name
    patch_openai_timing()


def patch_openai_timing() -> None:
    global _openai_patched
    if _openai_patched:
        return
    try:
        from openai.resources.chat.completions.completions import Completions
    except Exception:
        Completions = None
    if Completions is not None:
        Completions.create = _wrap_callable(
            Completions.create,
            operation="openai.chat.completions.create",
            model_from_kwargs=True,
        )
    _openai_patched = True


def patch_known_local_models() -> None:
    patch_specs = [
        ("rxnim.interface", "RxnIM", "predict_image_file", "rxnim.predict_image_file"),
        (
            "chemietoolkit.interface",
            "ChemIEToolkit",
            "extract_molecule_corefs_from_figures",
            "chemietoolkit.extract_molecule_corefs_from_figures",
        ),
        ("pytesseract", None, "image_to_string", "pytesseract.image_to_string"),
        ("chemrxnextractor", "RxnExtractor", "predict_strings", "rxn_extractor.predict_strings"),
        ("chemiener", "ChemNER", "predict_strings", "chemner.predict_strings"),
    ]
    for module_name, class_name, method_name, operation in patch_specs:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [method_name])
            target = getattr(module, class_name) if class_name else module
            original = getattr(target, method_name)
            setattr(target, method_name, _wrap_callable(original, operation=operation))
        except Exception:
            continue


def _wrap_callable(original, *, operation: str, model_from_kwargs: bool = False):
    if getattr(original, "_chemeagle_timing_wrapped", False):
        return original

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        model = kwargs.get("model") if model_from_kwargs else None
        try:
            with gpu_gate_context(operation):
                started_at = time.perf_counter()
                result = original(*args, **kwargs)
        except Exception as exc:
            record_event(
                operation=operation,
                model=str(model) if model is not None else None,
                status="error",
                event_kind="leaf",
                elapsed_seconds=time.perf_counter() - locals().get("started_at", time.perf_counter()),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise
        response_model = model or getattr(result, "model", None)
        record_event(
            operation=operation,
            model=str(response_model) if response_model is not None else None,
            status="success",
            event_kind="leaf",
            elapsed_seconds=time.perf_counter() - started_at,
        )
        return result

    wrapped._chemeagle_timing_wrapped = True
    return wrapped


@contextlib.contextmanager
def gpu_gate_context(operation: str) -> Iterator[None]:
    if operation not in GPU_GATED_OPERATIONS or gpu_gate is None or not gpu_gate_enabled():
        yield
        return
    with gpu_gate(operation) as gate_info:
        record_event(
            operation="chemeagle.gpu_gate.wait",
            status=gate_info.get("status", "unknown"),
            event_kind="leaf",
            elapsed_seconds=gate_info.get("wait_seconds", 0.0),
            gated_operation=operation,
            gate_samples=gate_info.get("samples"),
            gate_error=gate_info.get("error"),
            gate_lock_path=gate_info.get("lock_path"),
            gate_lock_fallback_from=gate_info.get("lock_fallback_from"),
            gate_last_snapshot=gate_info.get("last_snapshot"),
        )
        yield


def summarize_chemeagle_timing(events_path: Path | str, report_path: Path | str) -> Dict[str, Any]:
    events_path = Path(events_path)
    report_path = Path(report_path)
    events = read_events(events_path)
    normalized_events = [normalize_event(event) for event in events]
    summary: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "events_path": str(events_path),
        "event_count": len(normalized_events),
        "success_count": sum(1 for event in normalized_events if event.get("status") == "success"),
        "error_count": sum(1 for event in normalized_events if event.get("status") == "error"),
        "total_elapsed_seconds": round(
            sum(
                numeric_elapsed(event)
                for event in normalized_events
                if event.get("operation") in {"pdf_extraction.run_pdf", "ChemEagle(image_total)"}
            ),
            3,
        ),
        "total_leaf_elapsed_seconds": round(
            sum(numeric_elapsed(event) for event in normalized_events if is_leaf_event(event)),
            3,
        ),
        "total_event_elapsed_seconds": round(sum(numeric_elapsed(event) for event in normalized_events), 3),
        "pdf_count": len({event.get("pdf_name") for event in normalized_events if event.get("pdf_name")}),
        "image_count": len({event.get("image_name") for event in normalized_events if event.get("image_name")}),
        "by_operation": {},
        "by_leaf_operation": {},
        "by_pdf": {},
        "by_image": {},
        "slowest_events": [],
        "slowest_leaf_events": [],
    }
    for event in normalized_events:
        elapsed = numeric_elapsed(event)
        add_bucket(summary["by_operation"], event.get("operation") or "unknown", elapsed, event)
        add_bucket(summary["by_pdf"], event.get("pdf_name") or "unknown", elapsed, event)
        if is_leaf_event(event):
            add_bucket(summary["by_leaf_operation"], event.get("operation") or "unknown", elapsed, event)
        image_name = event.get("image_name")
        if image_name:
            image_bucket = summary["by_image"].setdefault(
                image_name,
                {
                    "count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "total_elapsed_seconds": 0.0,
                    "avg_elapsed_seconds": 0.0,
                    "max_elapsed_seconds": 0.0,
                    "wall_elapsed_seconds": 0.0,
                    "leaf_elapsed_seconds": 0.0,
                    "overhead_seconds": 0.0,
                    "by_phase": {},
                    "by_leaf_operation": {},
                },
            )
            add_bucket(summary["by_image"], image_name, elapsed, event)
            if event.get("operation") == "ChemEagle(image_total)":
                image_bucket["wall_elapsed_seconds"] = round(
                    image_bucket.get("wall_elapsed_seconds", 0.0) + elapsed,
                    3,
                )
            if is_leaf_event(event):
                image_bucket["leaf_elapsed_seconds"] = round(
                    image_bucket.get("leaf_elapsed_seconds", 0.0) + elapsed,
                    3,
                )
                phase = infer_phase(event)
                phase_bucket = image_bucket["by_phase"].setdefault(phase, {})
                add_bucket(phase_bucket, event.get("operation") or "unknown", elapsed, event)
                add_bucket(image_bucket["by_leaf_operation"], event.get("operation") or "unknown", elapsed, event)
            selected_area = event.get("selected_area")
            if selected_area:
                image_bucket["selected_area"] = selected_area

    for image_bucket in summary["by_image"].values():
        image_bucket["overhead_seconds"] = round(
            image_bucket.get("wall_elapsed_seconds", 0.0) - image_bucket.get("leaf_elapsed_seconds", 0.0),
            3,
        )

    summary["slowest_events"] = sorted(
        normalized_events,
        key=numeric_elapsed,
        reverse=True,
    )[:20]
    summary["slowest_leaf_events"] = sorted(
        [event for event in normalized_events if is_leaf_event(event)],
        key=numeric_elapsed,
        reverse=True,
    )[:20]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def add_bucket(target: Dict[str, Any], key: str, elapsed: float, event: Dict[str, Any]) -> None:
    bucket = target.setdefault(
        key,
        {
            "count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_elapsed_seconds": 0.0,
            "avg_elapsed_seconds": 0.0,
            "max_elapsed_seconds": 0.0,
            "event_kind": event.get("event_kind"),
        },
    )
    bucket["count"] += 1
    if event.get("status") == "success":
        bucket["success_count"] += 1
    if event.get("status") == "error":
        bucket["error_count"] += 1
    bucket["total_elapsed_seconds"] = round(bucket["total_elapsed_seconds"] + elapsed, 3)
    bucket["avg_elapsed_seconds"] = round(bucket["total_elapsed_seconds"] / bucket["count"], 3)
    bucket["max_elapsed_seconds"] = round(max(bucket["max_elapsed_seconds"], elapsed), 3)


def numeric_elapsed(event: Dict[str, Any]) -> float:
    value = event.get("elapsed_seconds")
    return float(value) if isinstance(value, (int, float)) else 0.0


def normalize_event(event: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(event)
    normalized["event_kind"] = normalize_event_kind(
        normalized.get("event_kind"),
        normalized.get("operation") or "",
    )
    return normalized


def normalize_event_kind(value: Optional[str], operation: str) -> str:
    if value in {"span", "leaf"}:
        return value
    if operation in {
        "openai.chat.completions.create",
        "rxnim.predict_image_file",
        "chemietoolkit.extract_molecule_corefs_from_figures",
        "pytesseract.image_to_string",
        "rxn_extractor.predict_strings",
        "chemner.predict_strings",
    }:
        return "leaf"
    return "span"


def is_leaf_event(event: Dict[str, Any]) -> bool:
    return normalize_event_kind(event.get("event_kind"), event.get("operation") or "") == "leaf"


def infer_phase(event: Dict[str, Any]) -> str:
    parent = str(event.get("parent_operation") or "")
    operation = str(event.get("operation") or "")
    if parent == "planner_gpt":
        return "planner"
    if parent in {
        "selected_agent_total",
        "process_reaction_image_with_product_variant_R_group",
        "process_reaction_image_with_table_R_group",
        "get_full_reaction_template",
        "get_multi_molecular_full",
    }:
        return "selected_agent"
    if parent == "text_extraction_agent" or operation in {
        "pytesseract.image_to_string",
        "rxn_extractor.predict_strings",
        "chemner.predict_strings",
    }:
        return "text_extraction"
    if parent == "final_json_synthesis_gpt":
        return "final_synthesis"
    if parent == "pdf_extraction.run_pdf" or operation.startswith("visualheist."):
        return "pdf_extraction"
    return "unknown_leaf"


def read_events(events_path: Path) -> list[Dict[str, Any]]:
    if not events_path.exists():
        return []
    events = []
    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events
