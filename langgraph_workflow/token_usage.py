"""OpenAI token usage tracking for the LangGraph workflow."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


USAGE_LOG_ENV = "OPENAI_USAGE_LOG_PATH"
USAGE_RUN_ID_ENV = "OPENAI_USAGE_RUN_ID"
USAGE_STAGE_ENV = "OPENAI_USAGE_STAGE"
USAGE_PDF_NAME_ENV = "OPENAI_USAGE_PDF_NAME"
USAGE_ENABLED_ENV = "OPENAI_USAGE_ENABLED"

_stage_var = contextvars.ContextVar("token_usage_stage", default=None)
_pdf_name_var = contextvars.ContextVar("token_usage_pdf_name", default=None)
_extra_var = contextvars.ContextVar("token_usage_extra", default=None)
_patched = False
_lock = threading.Lock()


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]


def install_token_usage_tracking(
    log_path: Path | str,
    *,
    run_id: Optional[str] = None,
    enabled: bool = True,
    default_stage: Optional[str] = None,
) -> Optional[str]:
    """Install process-wide OpenAI SDK usage tracking."""
    if not enabled:
        os.environ[USAGE_ENABLED_ENV] = "0"
        return None

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = run_id or os.environ.get(USAGE_RUN_ID_ENV) or make_run_id()

    os.environ[USAGE_LOG_ENV] = str(log_path)
    os.environ[USAGE_RUN_ID_ENV] = run_id
    os.environ[USAGE_ENABLED_ENV] = "1"
    if default_stage:
        os.environ[USAGE_STAGE_ENV] = default_stage

    patch_openai_sdk()
    return run_id


@contextlib.contextmanager
def token_usage_context(
    stage: Optional[str] = None,
    pdf_name: Optional[str] = None,
    **extra: Any,
) -> Iterator[None]:
    stage_token = _stage_var.set(stage) if stage is not None else None
    pdf_token = _pdf_name_var.set(pdf_name) if pdf_name is not None else None
    extra_token = _extra_var.set(extra or None) if extra else None
    try:
        yield
    finally:
        if extra_token is not None:
            _extra_var.reset(extra_token)
        if pdf_token is not None:
            _pdf_name_var.reset(pdf_token)
        if stage_token is not None:
            _stage_var.reset(stage_token)


def patch_openai_sdk() -> None:
    global _patched
    if _patched:
        return
    try:
        from openai.resources.chat.completions.completions import Completions
    except Exception:
        Completions = None
    try:
        from openai.resources.responses.responses import Responses
    except Exception:
        Responses = None

    if Completions is not None:
        Completions.create = _wrap_create(Completions.create, operation="chat.completions.create")
    if Responses is not None:
        Responses.create = _wrap_create(Responses.create, operation="responses.create")
    _patched = True


def _wrap_create(original, *, operation: str):
    if getattr(original, "_token_usage_wrapped", False):
        return original

    def wrapped(self, *args, **kwargs):
        started_at = time.perf_counter()
        model = kwargs.get("model")
        try:
            response = original(self, *args, **kwargs)
        except Exception as exc:
            record_token_usage(
                operation=operation,
                model=model,
                response=None,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=time.perf_counter() - started_at,
            )
            raise
        record_token_usage(
            operation=operation,
            model=model or getattr(response, "model", None),
            response=response,
            status="success",
            elapsed_seconds=time.perf_counter() - started_at,
        )
        return response

    wrapped._token_usage_wrapped = True
    return wrapped


def record_token_usage(
    *,
    operation: str,
    model: Any,
    response: Any,
    status: str,
    elapsed_seconds: float,
    error: Optional[str] = None,
) -> None:
    if os.environ.get(USAGE_ENABLED_ENV, "1") == "0":
        return
    log_path = os.environ.get(USAGE_LOG_ENV)
    if not log_path:
        return

    usage = extract_usage(response)
    stage = current_stage()
    pdf_name = current_pdf_name()
    extra = _extra_var.get() or {}
    record = {
        "created_at": datetime.now().isoformat(),
        "run_id": os.environ.get(USAGE_RUN_ID_ENV),
        "stage": stage,
        "pdf_name": pdf_name,
        "model": str(model) if model is not None else None,
        "operation": operation,
        "status": status,
        "usage_present": usage is not None,
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
        "elapsed_seconds": round(elapsed_seconds, 3),
        **extra,
    }
    if error:
        record["error"] = error
    append_jsonl(Path(log_path), record)


def current_stage() -> str:
    stage = _stage_var.get() or os.environ.get(USAGE_STAGE_ENV)
    inferred = infer_stage_from_stack()
    return inferred or stage or "unknown"


def current_pdf_name() -> Optional[str]:
    return _pdf_name_var.get() or os.environ.get(USAGE_PDF_NAME_ENV)


def infer_stage_from_stack() -> Optional[str]:
    mappings = {
        "extract_registry_with_gpt": "text_name_registry",
        "summarize_gp_texts": "text_general_procedure_summary",
        "stage1_screen": "text_stage1_screening",
        "stage2_audit_missing_reactions": "text_stage2_audit",
        "stage2_extract": "text_stage2_extraction",
        "call_role_refinement_llm": "chemeagle_role_refinement",
        "_chat_json": "reaction_type_normalization",
    }
    try:
        for frame in inspect.stack(context=0)[2:16]:
            func = frame.function
            if func == "call_llm_parse":
                callers = {f.function for f in inspect.stack(context=0)[2:24]}
                if "build_scaffold_mapping" in callers:
                    return "text_registry_scaffold_parse"
                return "q1q2_scaffold_parse"
            if func in mappings:
                return mappings[func]
    except Exception:
        return None
    return None


def extract_usage(response: Any) -> Optional[Dict[str, Optional[int]]]:
    if response is None:
        return None
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None

    def get(name: str) -> Optional[int]:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        return int(value) if isinstance(value, int) else None

    prompt_tokens = get("prompt_tokens")
    completion_tokens = get("completion_tokens")
    total_tokens = get("total_tokens")
    if prompt_tokens is None:
        prompt_tokens = get("input_tokens")
    if completion_tokens is None:
        completion_tokens = get("output_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def summarize_token_usage(events_path: Path | str, report_path: Path | str) -> Dict[str, Any]:
    events_path = Path(events_path)
    report_path = Path(report_path)
    events = read_events(events_path)
    summary = {
        "created_at": datetime.now().isoformat(),
        "events_path": str(events_path),
        "request_count": len(events),
        "success_count": sum(1 for event in events if event.get("status") == "success"),
        "error_count": sum(1 for event in events if event.get("status") == "error"),
        "usage_missing_count": sum(1 for event in events if not event.get("usage_present")),
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "by_model": {},
        "by_stage": {},
        "by_pdf": {},
    }
    for event in events:
        prompt = event.get("prompt_tokens")
        completion = event.get("completion_tokens")
        total = event.get("total_tokens")
        add_usage(summary, prompt, completion, total)
        for bucket_name, key in (
            ("by_model", event.get("model") or "unknown"),
            ("by_stage", event.get("stage") or "unknown"),
            ("by_pdf", event.get("pdf_name") or "unknown"),
        ):
            bucket = summary[bucket_name].setdefault(
                key,
                {
                    "request_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "usage_missing_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            bucket["request_count"] += 1
            if event.get("status") == "success":
                bucket["success_count"] += 1
            if event.get("status") == "error":
                bucket["error_count"] += 1
            if not event.get("usage_present"):
                bucket["usage_missing_count"] += 1
            add_usage(bucket, prompt, completion, total)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def add_usage(target: Dict[str, Any], prompt: Any, completion: Any, total: Any) -> None:
    if isinstance(prompt, int):
        target["total_prompt_tokens" if "total_prompt_tokens" in target else "prompt_tokens"] += prompt
    if isinstance(completion, int):
        target["total_completion_tokens" if "total_completion_tokens" in target else "completion_tokens"] += completion
    if isinstance(total, int):
        target["total_tokens"] += total


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
