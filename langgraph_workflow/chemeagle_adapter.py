"""ChemEagle integration helpers for the LangGraph workflow."""

import copy
import contextlib
import json
import os
import re
import socket
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from langgraph_workflow.token_usage import USAGE_PDF_NAME_ENV, USAGE_STAGE_ENV
except ImportError:
    USAGE_PDF_NAME_ENV = "OPENAI_USAGE_PDF_NAME"
    USAGE_STAGE_ENV = "OPENAI_USAGE_STAGE"

try:
    from langgraph_workflow.chemeagle_timing import (
        TIMING_EVENTS_ENV,
        TIMING_PDF_NAME_ENV,
        TIMING_REPORT_ENV,
        TIMING_RUN_ID_ENV,
    )
except ImportError:
    TIMING_EVENTS_ENV = "CHEMEAGLE_TIMING_EVENTS_PATH"
    TIMING_REPORT_ENV = "CHEMEAGLE_TIMING_REPORT_PATH"
    TIMING_RUN_ID_ENV = "CHEMEAGLE_TIMING_RUN_ID"
    TIMING_PDF_NAME_ENV = "CHEMEAGLE_TIMING_PDF_NAME"


BRIDGE_PATH = Path(__file__).resolve().parent / "chemeagle_bridge.py"
INVALID_SMILES_VALUES = {"", "none", "null", "not specified", "<invalid>", "invalid"}
ROLE_REFINEMENT_ROLES = {
    "catalyst",
    "ligand",
    "additive",
    "base",
    "reagent",
    "solvent",
    "temperature",
    "time",
    "concentration",
    "volume",
    "atmosphere",
    "yield",
    "ee",
    "er",
    "dr",
    "unknown",
}
ROLE_REFINEMENT_DIRECT_ROLES = {
    "solvent",
    "temperature",
    "time",
    "concentration",
    "volume",
    "atmosphere",
    "light",
    "light source",
    "wavelength",
    "catalyst",
    "catalysts",
    "ligand",
    "precatalyst",
    "base",
    "additive",
    "additives",
    "yield",
    "ee",
    "er",
    "dr",
}
ROLE_REFINEMENT_SYSTEM_PROMPT = """You refine chemistry condition roles extracted from reaction images.
Return valid JSON only."""
ROLE_REFINEMENT_USER_PROMPT = """Classify only the existing ChemEagle condition rows for one image-extracted reaction.

Rules:
- Use only the provided reactants, products, conditions, and additional_info.
- Do not use text-extraction results or general procedure text.
- Do not create, delete, merge, split, rename, or reorder conditions.
- Do not change SMILES, IUPAC names, labels, or text.
- Choose each refined_role from this exact set:
  catalyst, ligand, additive, base, reagent, solvent, temperature, time, concentration, volume, atmosphere, yield, ee, er, dr, unknown.
- If a metal salt, ligand, or named catalyst is used in mol%, prefer catalyst or ligand when supported.
- If a base or additive is used in equiv, prefer base or additive when supported.
- If evidence is insufficient, use reagent or unknown.

Return this JSON shape:
{{"reaction_id":"...","conditions":[{{"condition_index":0,"original_role":"reagent","text":"CuI (5 mol%)","refined_role":"catalyst","reason":"short reason"}}]}}

Reaction:
{payload}
"""
ROLE_REFINEMENT_BATCH_USER_PROMPT = """Classify only the existing ChemEagle condition rows for multiple reactions from one image.

Rules:
- Use only the provided reactants, products, conditions, and additional_info.
- Do not use text-extraction results or general procedure text.
- Do not create, delete, merge, split, rename, or reorder reactions or conditions.
- Do not change SMILES, IUPAC names, labels, or text.
- Choose each refined_role from this exact set:
  catalyst, ligand, additive, base, reagent, solvent, temperature, time, concentration, volume, atmosphere, yield, ee, er, dr, unknown.
- If a metal salt, ligand, or named catalyst is used in mol%, prefer catalyst or ligand when supported.
- If a base or additive is used in equiv, prefer base or additive when supported.
- If evidence is insufficient, use reagent or unknown.
- Return one condition row for every condition_index listed in conditions_to_classify for every reaction.

Return this JSON shape:
{{"reactions":[{{"reaction_index":0,"reaction_id":"...","conditions":[{{"condition_index":0,"original_role":"reagent","text":"CuI (5 mol%)","refined_role":"catalyst","reason":"short reason"}}]}}]}}

Image payload:
{payload}
"""

try:
    from langgraph_workflow.chemeagle_gpu_gate import (
        GPU_GATE_ENABLED_ENV,
        GPU_GATE_LOCK_DIR_ENV,
        GPU_GATE_MEMORY_FREE_MIB_ENV,
        GPU_GATE_POLL_INTERVAL_ENV,
        GPU_GATE_STABLE_SAMPLES_ENV,
        GPU_GATE_TIMEOUT_ENV,
        GPU_GATE_UTIL_THRESHOLD_ENV,
    )
except ImportError:
    GPU_GATE_ENABLED_ENV = "CHEMEAGLE_GPU_GATE_ENABLED"
    GPU_GATE_LOCK_DIR_ENV = "CHEMEAGLE_GPU_GATE_LOCK_DIR"
    GPU_GATE_UTIL_THRESHOLD_ENV = "CHEMEAGLE_GPU_GATE_UTIL_THRESHOLD"
    GPU_GATE_MEMORY_FREE_MIB_ENV = "CHEMEAGLE_GPU_GATE_MEMORY_FREE_MIB"
    GPU_GATE_STABLE_SAMPLES_ENV = "CHEMEAGLE_GPU_GATE_STABLE_SAMPLES"
    GPU_GATE_POLL_INTERVAL_ENV = "CHEMEAGLE_GPU_GATE_POLL_INTERVAL"
    GPU_GATE_TIMEOUT_ENV = "CHEMEAGLE_GPU_GATE_TIMEOUT"

try:
    from langgraph_workflow.chemeagle_gpu_worker_client import (
        GPU_WORKER_ENABLED_ENV,
        GPU_WORKER_HOST_ENV,
        GPU_WORKER_PORT_ENV,
        GPU_WORKER_TASK_DIR_ENV,
        GPU_WORKER_TIMEOUT_ENV,
    )
except ImportError:
    GPU_WORKER_ENABLED_ENV = "CHEMEAGLE_GPU_WORKER_ENABLED"
    GPU_WORKER_HOST_ENV = "CHEMEAGLE_GPU_WORKER_HOST"
    GPU_WORKER_PORT_ENV = "CHEMEAGLE_GPU_WORKER_PORT"
    GPU_WORKER_TASK_DIR_ENV = "CHEMEAGLE_GPU_WORKER_TASK_DIR"
    GPU_WORKER_TIMEOUT_ENV = "CHEMEAGLE_GPU_WORKER_TIMEOUT"

try:
    import pubchempy as pcp
except ImportError:
    pcp = None

_IUPAC_TIMEOUT_LOCK = threading.Lock()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_chemeagle_pdf(
    *,
    chemeagle_python: str,
    chemeagle_dir: Path,
    pdf_path: Path,
    image_dir: Path,
    raw_result_path: Path,
    pdf_model_size: str,
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    max_images: int,
    use_plan_observer: bool,
    use_action_observer: bool,
    max_parallel_images: int = 1,
    timing_events_path: Optional[Path] = None,
    timing_report_path: Optional[Path] = None,
    timing_run_id: Optional[str] = None,
    enable_gpu_gate: bool = False,
    gpu_gate_lock_dir: Optional[Path] = None,
    gpu_gate_util_threshold: float = 20.0,
    gpu_gate_memory_free_mib: float = 2000.0,
    gpu_gate_stable_samples: int = 2,
    gpu_gate_poll_interval: float = 1.0,
    gpu_gate_timeout: float = 0.0,
    chemeagle_execution_mode: str = "subprocess",
    gpu_worker_host: Optional[str] = None,
    gpu_worker_port: Optional[int] = None,
    gpu_worker_task_dir: Optional[Path] = None,
    gpu_worker_timeout: float = 0.0,
) -> Dict[str, Any]:
    chemeagle_dir = chemeagle_dir.resolve()
    pdf_path = pdf_path.resolve()
    image_dir = image_dir.resolve()
    raw_result_path = raw_result_path.resolve()
    command = [
        chemeagle_python,
        str(BRIDGE_PATH),
        "--chemeagle-dir",
        str(chemeagle_dir),
        "--pdf-path",
        str(pdf_path),
        "--image-dir",
        str(image_dir),
        "--raw-result-path",
        str(raw_result_path),
        "--pdf-model-size",
        pdf_model_size,
    ]
    if max_images and max_images > 0:
        command.extend(["--max-images", str(max_images)])
    if max_parallel_images and max_parallel_images > 1:
        command.extend(["--max-parallel-images", str(max_parallel_images)])
    if not use_plan_observer:
        command.append("--no-plan-observer")
    if not use_action_observer:
        command.append("--no-action-observer")

    log_path = raw_result_path.with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env[USAGE_STAGE_ENV] = "chemeagle_subprocess"
    env[USAGE_PDF_NAME_ENV] = pdf_path.name
    if timing_events_path:
        timing_events_path = Path(timing_events_path).resolve()
        timing_events_path.parent.mkdir(parents=True, exist_ok=True)
        env[TIMING_EVENTS_ENV] = str(timing_events_path)
        env[TIMING_PDF_NAME_ENV] = pdf_path.name
    if timing_report_path:
        timing_report_path = Path(timing_report_path).resolve()
        timing_report_path.parent.mkdir(parents=True, exist_ok=True)
        env[TIMING_REPORT_ENV] = str(timing_report_path)
    if timing_run_id:
        env[TIMING_RUN_ID_ENV] = timing_run_id
    if chemeagle_execution_mode == "task_gpu_worker":
        if not gpu_worker_host or not gpu_worker_port:
            raise ValueError("ChemEagle task_gpu_worker mode requires a running GPU worker host/port.")
        if gpu_worker_task_dir is None:
            gpu_worker_task_dir = raw_result_path.parent / "chemeagle_gpu_worker_tasks"
        gpu_worker_task_dir = Path(gpu_worker_task_dir).resolve()
        gpu_worker_task_dir.mkdir(parents=True, exist_ok=True)
        env[GPU_WORKER_ENABLED_ENV] = "1"
        env[GPU_WORKER_HOST_ENV] = str(gpu_worker_host)
        env[GPU_WORKER_PORT_ENV] = str(gpu_worker_port)
        env[GPU_WORKER_TASK_DIR_ENV] = str(gpu_worker_task_dir)
        env[GPU_WORKER_TIMEOUT_ENV] = str(gpu_worker_timeout)
    if enable_gpu_gate and chemeagle_execution_mode != "task_gpu_worker":
        if gpu_gate_lock_dir is None:
            gpu_gate_lock_dir = raw_result_path.parent / "chemeagle_gpu_gate.lock"
        gpu_gate_lock_dir = Path(gpu_gate_lock_dir).resolve()
        gpu_gate_lock_dir.parent.mkdir(parents=True, exist_ok=True)
        env[GPU_GATE_ENABLED_ENV] = "1"
        env[GPU_GATE_LOCK_DIR_ENV] = str(gpu_gate_lock_dir)
        env[GPU_GATE_UTIL_THRESHOLD_ENV] = str(gpu_gate_util_threshold)
        env[GPU_GATE_MEMORY_FREE_MIB_ENV] = str(gpu_gate_memory_free_mib)
        env[GPU_GATE_STABLE_SAMPLES_ENV] = str(gpu_gate_stable_samples)
        env[GPU_GATE_POLL_INTERVAL_ENV] = str(gpu_gate_poll_interval)
        env[GPU_GATE_TIMEOUT_ENV] = str(gpu_gate_timeout)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        completed = subprocess.run(
            command,
            cwd=str(chemeagle_dir),
            env=env,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = {
        "command": mask_command(command),
        "returncode": completed.returncode,
        "log_path": str(log_path),
    }
    if completed.returncode != 0:
        log_tail = ""
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            pass
        raise RuntimeError(
            "ChemEagle subprocess failed with return code "
            f"{completed.returncode}.\nLog tail:\n{log_tail}"
        )
    return result


def mask_command(command: List[str]) -> List[str]:
    masked = list(command)
    for index, value in enumerate(masked[:-1]):
        if value == "--api-key":
            masked[index + 1] = "***"
    return masked


def enrich_chemeagle_raw_with_iupac(
    raw_payload: Any,
    cache_path: Path,
    *,
    lookup_timeout: float = 15.0,
) -> tuple[Any, Dict[str, Any]]:
    cache = load_iupac_cache(cache_path)
    entries = cache.setdefault("entries", {})
    stats = {
        "status": "success",
        "resolved": 0,
        "not_found": 0,
        "error": 0,
        "skipped": 0,
        "cache_hits": 0,
        "queries": 0,
        "unique_smiles": 0,
        "cache_path": str(cache_path),
    }
    seen_smiles = set()
    enriched_payload = copy.deepcopy(raw_payload)

    def enrich_node(node: Any) -> Any:
        if isinstance(node, list):
            for item in node:
                enrich_node(item)
            return node
        if not isinstance(node, dict):
            return node

        for value in node.values():
            enrich_node(value)

        if "smiles" not in node:
            return node

        smiles = smiles_for_iupac_lookup(node.get("smiles"))
        if not smiles:
            node["iupac_name"] = None
            node["iupac_lookup_status"] = "skipped"
            stats["skipped"] += 1
            return node

        seen_smiles.add(smiles)
        cached = entries.get(smiles)
        if isinstance(cached, dict):
            stats["cache_hits"] += 1
            lookup = cached
        else:
            stats["queries"] += 1
            lookup = lookup_iupac_name(smiles, timeout_seconds=lookup_timeout)
            entries[smiles] = {
                **lookup,
                "updated_at": datetime.now().isoformat(),
            }

        status = lookup.get("status") or "error"
        node["iupac_name"] = lookup.get("iupac_name")
        node["iupac_lookup_status"] = status
        if lookup.get("error"):
            node["iupac_lookup_error"] = lookup.get("error")
        if status in {"resolved", "not_found", "error"}:
            stats[status] += 1
        else:
            stats["error"] += 1
        return node

    enrich_node(enriched_payload)
    stats["unique_smiles"] = len(seen_smiles)
    write_json(cache_path, cache)
    return enriched_payload, stats


def load_iupac_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {"version": 1, "entries": {}}
    try:
        cache = read_json(cache_path)
    except Exception:
        return {"version": 1, "entries": {}}
    if not isinstance(cache, dict):
        return {"version": 1, "entries": {}}
    if not isinstance(cache.get("entries"), dict):
        cache["entries"] = {}
    cache.setdefault("version", 1)
    return cache


def smiles_for_iupac_lookup(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    smiles = value.strip()
    if smiles.casefold() in INVALID_SMILES_VALUES:
        return ""
    return smiles


@contextlib.contextmanager
def temporary_socket_timeout(timeout_seconds: float):
    if timeout_seconds <= 0:
        yield
        return
    with _IUPAC_TIMEOUT_LOCK:
        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout_seconds)
        try:
            yield
        finally:
            socket.setdefaulttimeout(previous_timeout)


def lookup_iupac_name(smiles: str, *, timeout_seconds: float = 15.0) -> Dict[str, Any]:
    if pcp is None:
        return {
            "status": "error",
            "iupac_name": None,
            "error": "pubchempy is not installed",
        }
    try:
        with temporary_socket_timeout(timeout_seconds):
            compounds = pcp.get_compounds(smiles, "smiles")
        if not compounds:
            return {"status": "not_found", "iupac_name": None}
        iupac_name = getattr(compounds[0], "iupac_name", None)
        if not iupac_name:
            return {"status": "not_found", "iupac_name": None}
        return {"status": "resolved", "iupac_name": iupac_name}
    except Exception as exc:
        return {
            "status": "error",
            "iupac_name": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def refine_chemeagle_condition_roles(
    raw_payload: Any,
    *,
    model: str,
    api_key: Optional[str],
    base_url: str,
) -> tuple[Any, Dict[str, Any]]:
    if OpenAI is None:
        raise ImportError("openai package is required for ChemEagle role refinement")

    client = OpenAI(base_url=base_url, api_key=api_key or os.getenv("OPENAI_API_KEY"))
    refined_payload = copy.deepcopy(raw_payload)
    records = refined_payload if isinstance(refined_payload, list) else [refined_payload]
    stats = {
        "status": "success",
        "model": model,
        "total_records": 0,
        "total_reactions": 0,
        "conditions_total": 0,
        "conditions_direct": 0,
        "conditions_llm_requested": 0,
        "conditions_llm_refined": 0,
        "conditions_fallback": 0,
        "role_refinement_llm_batch_requests": 0,
        "role_refinement_llm_fallback_requests": 0,
        "role_refinement_reactions_batched": 0,
        "role_refinement_reactions_fallback": 0,
        "role_refinement_batch_errors": [],
        "failed_reactions": [],
    }

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        stats["total_records"] += 1
        if record.get("status") == "error":
            continue
        reactions = record.get("reactions") or []
        if not isinstance(reactions, list):
            continue
        pending_reactions = []
        for reaction_index, reaction in enumerate(reactions):
            if not isinstance(reaction, dict):
                continue
            stats["total_reactions"] += 1
            pending_indexes = prepare_reaction_condition_roles(reaction, stats)
            if pending_indexes:
                pending_reactions.append(
                    {
                        "reaction": reaction,
                        "reaction_index": reaction_index,
                        "pending_indexes": pending_indexes,
                    }
                )
        refine_record_condition_roles(
            client=client,
            model=model,
            record=record,
            record_index=record_index,
            pending_reactions=pending_reactions,
            stats=stats,
        )
    return refined_payload, stats


def prepare_reaction_condition_roles(reaction: Dict[str, Any], stats: Dict[str, Any]) -> List[int]:
    conditions = reaction.get("conditions") or []
    if not isinstance(conditions, list):
        return []
    condition_dicts = [condition for condition in conditions if isinstance(condition, dict)]
    stats["conditions_total"] += len(condition_dicts)
    for condition in condition_dicts:
        original_role = adapter_clean_text(condition.get("role")).casefold().replace("_", " ")
        condition["original_role"] = original_role
        if role_refinement_can_keep(original_role):
            condition["refined_role"] = role_refinement_normalize_direct_role(original_role)
            condition["role_source"] = "original_chemeagle_role"
            condition["role_reason"] = "Original ChemEagle role is already specific."
            stats["conditions_direct"] += 1

    pending_indexes = [
        index
        for index, condition in enumerate(conditions)
        if isinstance(condition, dict) and not condition.get("refined_role")
    ]
    stats["conditions_llm_requested"] += len(pending_indexes)
    return pending_indexes


def refine_record_condition_roles(
    *,
    client,
    model: str,
    record: Dict[str, Any],
    record_index: int,
    pending_reactions: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> None:
    if not pending_reactions:
        return

    stats["role_refinement_llm_batch_requests"] += 1
    stats["role_refinement_reactions_batched"] += len(pending_reactions)
    fallback_reactions = []
    try:
        batch_rows = call_role_refinement_batch_llm(
            client=client,
            model=model,
            record=record,
            pending_reactions=pending_reactions,
        )
        fallback_reactions = apply_role_refinement_batch_rows(
            pending_reactions,
            batch_rows,
            stats=stats,
        )
    except Exception as exc:
        stats.setdefault("role_refinement_batch_errors", []).append(
            {
                "record_index": record_index,
                "pdf_name": record.get("pdf_name"),
                "image_name": record.get("image_name"),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        fallback_reactions = pending_reactions

    for item in fallback_reactions:
        refine_reaction_condition_roles_with_fallback(
            client=client,
            model=model,
            record=record,
            reaction=item["reaction"],
            record_index=record_index,
            reaction_index=item["reaction_index"],
            pending_indexes=item["pending_indexes"],
            stats=stats,
        )


def refine_reaction_condition_roles_with_fallback(
    *,
    client,
    model: str,
    record: Dict[str, Any],
    reaction: Dict[str, Any],
    record_index: int,
    reaction_index: int,
    pending_indexes: List[int],
    stats: Dict[str, Any],
) -> None:
    unresolved_indexes = [
        index
        for index in pending_indexes
        if index < len(reaction.get("conditions") or [])
        and isinstance((reaction.get("conditions") or [])[index], dict)
        and not (reaction.get("conditions") or [])[index].get("refined_role")
    ]
    if not unresolved_indexes:
        return

    stats["role_refinement_llm_fallback_requests"] += 1
    stats["role_refinement_reactions_fallback"] += 1
    try:
        llm_rows = call_role_refinement_llm(
            client=client,
            model=model,
            record=record,
            reaction=reaction,
            pending_indexes=unresolved_indexes,
        )
        conditions = reaction.get("conditions") or []
        applied = apply_role_refinement_rows(conditions, llm_rows, allowed_indexes=set(unresolved_indexes))
        stats["conditions_llm_refined"] += applied
        fallback_unrefined_conditions(conditions, unresolved_indexes, reason="llm_missing_or_invalid_rows")
        stats["conditions_fallback"] += max(0, len(unresolved_indexes) - applied)
    except Exception as exc:
        conditions = reaction.get("conditions") or []
        fallback_unrefined_conditions(conditions, unresolved_indexes, reason=f"{type(exc).__name__}: {exc}")
        stats["conditions_fallback"] += len(unresolved_indexes)
        record_role_refinement_failure(
            stats,
            record=record,
            record_index=record_index,
            reaction=reaction,
            reaction_index=reaction_index,
            error=f"fallback {type(exc).__name__}: {exc}",
        )


def record_role_refinement_failure(
    stats: Dict[str, Any],
    *,
    record: Dict[str, Any],
    record_index: int,
    reaction: Dict[str, Any],
    reaction_index: int,
    error: str,
) -> None:
    stats["failed_reactions"].append(
        {
            "record_index": record_index,
            "reaction_index": reaction_index,
            "pdf_name": record.get("pdf_name"),
            "image_name": record.get("image_name"),
            "reaction_id": reaction.get("reaction_id") or reaction.get("id"),
            "error": error,
        }
    )


def role_refinement_can_keep(role: str) -> bool:
    return role in ROLE_REFINEMENT_DIRECT_ROLES


def role_refinement_normalize_direct_role(role: str) -> str:
    role = role.casefold().replace("_", " ")
    if role == "light source":
        return "light"
    if role == "precatalyst":
        return "catalyst"
    if role == "catalysts":
        return "catalyst"
    if role == "additives":
        return "additive"
    return role


def fallback_unrefined_conditions(conditions: List[Any], indexes: List[int], *, reason: str) -> None:
    for index in indexes:
        if index >= len(conditions) or not isinstance(conditions[index], dict):
            continue
        condition = conditions[index]
        if condition.get("refined_role"):
            continue
        original_role = adapter_clean_text(condition.get("original_role") or condition.get("role")).casefold().replace("_", " ")
        fallback_role = original_role if original_role in ROLE_REFINEMENT_ROLES else "reagent"
        condition["original_role"] = original_role
        condition["refined_role"] = fallback_role
        condition["role_source"] = "original_chemeagle_role"
        condition["role_reason"] = f"Role refinement fallback: {reason}"


def apply_role_refinement_rows(
    conditions: List[Any],
    rows: List[Dict[str, Any]],
    *,
    allowed_indexes: set,
) -> int:
    applied = 0
    seen_indexes = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("condition_index")
        if not isinstance(index, int) or index < 0 or index >= len(conditions):
            continue
        if index not in allowed_indexes:
            continue
        if index in seen_indexes:
            continue
        condition = conditions[index]
        if not isinstance(condition, dict):
            continue
        refined_role = adapter_clean_text(row.get("refined_role")).casefold().replace("_", " ")
        if refined_role not in ROLE_REFINEMENT_ROLES:
            continue
        condition["original_role"] = adapter_clean_text(condition.get("original_role") or condition.get("role")).casefold().replace("_", " ")
        condition["refined_role"] = refined_role
        condition["role_source"] = "llm_role_refinement"
        condition["role_reason"] = adapter_clean_text(row.get("reason")) or "LLM role refinement"
        applied += 1
        seen_indexes.add(index)
    return applied


def apply_role_refinement_batch_rows(
    pending_reactions: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    *,
    stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows_by_reaction_index: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        reaction_index = row.get("reaction_index")
        if isinstance(reaction_index, int):
            rows_by_reaction_index[reaction_index] = row

    fallback_reactions = []
    for item in pending_reactions:
        reaction = item["reaction"]
        reaction_index = item["reaction_index"]
        pending_indexes = item["pending_indexes"]
        response_row = rows_by_reaction_index.get(reaction_index)
        if not isinstance(response_row, dict):
            fallback_reactions.append(item)
            continue

        conditions = reaction.get("conditions") or []
        condition_rows = response_row.get("conditions")
        if not isinstance(condition_rows, list):
            fallback_reactions.append(item)
            continue

        accepted_rows = validated_role_refinement_rows(
            condition_rows,
            conditions=conditions,
            allowed_indexes=set(pending_indexes),
        )
        accepted_indexes = {row["condition_index"] for row in accepted_rows}
        if accepted_indexes != set(pending_indexes):
            fallback_reactions.append(item)
            continue

        applied = apply_role_refinement_rows(conditions, accepted_rows, allowed_indexes=set(pending_indexes))
        if applied != len(pending_indexes):
            fallback_reactions.append(item)
            continue
        stats["conditions_llm_refined"] += applied
    return fallback_reactions


def validated_role_refinement_rows(
    rows: List[Dict[str, Any]],
    *,
    conditions: List[Any],
    allowed_indexes: set,
) -> List[Dict[str, Any]]:
    accepted = []
    seen_indexes = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        index = row.get("condition_index")
        if not isinstance(index, int) or index < 0 or index >= len(conditions):
            continue
        if index not in allowed_indexes or index in seen_indexes:
            continue
        if not isinstance(conditions[index], dict):
            continue
        refined_role = adapter_clean_text(row.get("refined_role")).casefold().replace("_", " ")
        if refined_role not in ROLE_REFINEMENT_ROLES:
            continue
        normalized = dict(row)
        normalized["condition_index"] = index
        normalized["refined_role"] = refined_role
        accepted.append(normalized)
        seen_indexes.add(index)
    return accepted


def call_role_refinement_batch_llm(
    *,
    client,
    model: str,
    record: Dict[str, Any],
    pending_reactions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    payload = {
        "pdf_name": record.get("pdf_name"),
        "image_name": record.get("image_name"),
        "reactions": [
            compact_reaction_for_role_refinement(
                item["reaction"],
                reaction_index=item["reaction_index"],
                pending_indexes=item["pending_indexes"],
            )
            for item in pending_reactions
        ],
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ROLE_REFINEMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ROLE_REFINEMENT_BATCH_USER_PROMPT.format(
                    payload=json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        temperature=0.0,
    )
    data = load_json_response(response.choices[0].message.content or "")
    rows = data.get("reactions", []) if isinstance(data, dict) else []
    return rows if isinstance(rows, list) else []


def call_role_refinement_llm(
    *,
    client,
    model: str,
    record: Dict[str, Any],
    reaction: Dict[str, Any],
    pending_indexes: List[int],
) -> List[Dict[str, Any]]:
    payload = {
        "pdf_name": record.get("pdf_name"),
        "image_name": record.get("image_name"),
        "reaction_id": reaction.get("reaction_id") or reaction.get("id"),
        "reactants": compact_chemeagle_items(reaction.get("reactants")),
        "products": compact_chemeagle_items(reaction.get("products")),
        "conditions": [
            compact_condition_for_role_refinement(condition, index)
            for index, condition in enumerate(reaction.get("conditions") or [])
            if isinstance(condition, dict)
        ],
        "conditions_to_classify": pending_indexes,
        "additional_info": reaction.get("additional_info") or [],
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ROLE_REFINEMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": ROLE_REFINEMENT_USER_PROMPT.format(
                    payload=json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            },
        ],
        temperature=0.0,
    )
    data = load_json_response(response.choices[0].message.content or "")
    rows = data.get("conditions", []) if isinstance(data, dict) else []
    return rows if isinstance(rows, list) else []


def compact_reaction_for_role_refinement(
    reaction: Dict[str, Any],
    *,
    reaction_index: int,
    pending_indexes: List[int],
) -> Dict[str, Any]:
    return {
        "reaction_index": reaction_index,
        "reaction_id": reaction.get("reaction_id") or reaction.get("id"),
        "reactants": compact_chemeagle_items(reaction.get("reactants")),
        "products": compact_chemeagle_items(reaction.get("products")),
        "conditions": [
            compact_condition_for_role_refinement(condition, index)
            for index, condition in enumerate(reaction.get("conditions") or [])
            if isinstance(condition, dict)
        ],
        "conditions_to_classify": pending_indexes,
        "additional_info": reaction.get("additional_info") or [],
    }


def compact_chemeagle_items(value: Any) -> List[Dict[str, Any]]:
    items = value if isinstance(value, list) else []
    compacted = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compacted.append(
            {
                "label": item.get("label"),
                "text": item.get("text"),
                "smiles": item.get("smiles"),
                "iupac_name": item.get("iupac_name"),
            }
        )
    return compacted


def compact_condition_for_role_refinement(condition: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "condition_index": index,
        "original_role": condition.get("role"),
        "text": condition.get("text"),
        "label": condition.get("label"),
        "smiles": condition.get("smiles"),
        "iupac_name": condition.get("iupac_name"),
        "additional_fields": {
            key: value
            for key, value in condition.items()
            if key not in {"role", "text", "label", "smiles", "iupac_name"}
        },
    }


def load_json_response(raw: str) -> Any:
    raw = (raw or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    elif raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def adapter_clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "none", "null", "not specified"}:
        return ""
    return re.sub(r"\s+", " ", text)


def normalize_chemeagle_payload(
    raw_payload: Any,
    *,
    pdf_path: Path,
    artifact_stem: str,
    source_paper: Optional[str] = None,
) -> Dict[str, Any]:
    records = raw_payload if isinstance(raw_payload, list) else [raw_payload]
    reactions: List[Dict[str, Any]] = []
    image_count = 0
    failed_images = []

    for image_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        image_count += 1
        if record.get("status") == "error":
            failed_images.append(
                {
                    "image_name": record.get("image_name"),
                    "error": record.get("error"),
                }
            )
            continue
        image_name = str(record.get("image_name") or f"image_{image_index}")
        image_path = record.get("image_path")
        condition_contexts = collect_reagents_condition_contexts(record)
        condition_context_labels = collect_condition_context_labels(condition_contexts)
        for reaction_index, reaction in enumerate(record.get("reactions") or [], start=1):
            if not isinstance(reaction, dict):
                continue
            normalized = normalize_chemeagle_reaction(
                reaction,
                pdf_path=pdf_path,
                artifact_stem=artifact_stem,
                source_paper=source_paper,
                image_name=image_name,
                image_path=image_path,
                image_index=image_index,
                reaction_index=reaction_index,
                condition_contexts=condition_contexts,
                condition_context_labels=condition_context_labels,
            )
            reactions.append(normalized)

    return {
        "source": str(pdf_path),
        "paper_key": source_paper,
        "extracted_at": datetime.now().isoformat(),
        "extractor": "ChemEagle",
        "source_modality": "image",
        "total_images": image_count,
        "failed_images": failed_images,
        "total_reactions": len(reactions),
        "reactions": reactions,
    }


def cleanup_reaction_targets(payload: Dict[str, Any]) -> Dict[str, Any]:
    reactions = payload.get("reactions")
    if not isinstance(reactions, list):
        return payload

    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        targets = reaction.get("targets")
        if not isinstance(targets, dict):
            continue
        for metric in ("yield", "ee", "er", "dr"):
            if metric not in targets:
                continue
            cleaned = cleanup_target_value(targets.get(metric))
            if cleaned is None:
                targets[metric] = None
            else:
                targets[metric] = cleaned
    return payload


def cleanup_target_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    while len(text) >= 2 and text.startswith("(") and text.endswith(")"):
        inner = text[1:-1].strip()
        if not inner:
            break
        text = inner
    return text


def normalize_chemeagle_reaction(
    reaction: Dict[str, Any],
    *,
    pdf_path: Path,
    artifact_stem: str,
    source_paper: Optional[str],
    image_name: str,
    image_path: Optional[str],
    image_index: int,
    reaction_index: int,
    condition_contexts: Optional[List[Dict[str, Any]]] = None,
    condition_context_labels: Optional[set] = None,
) -> Dict[str, Any]:
    reaction_id = str(reaction.get("reaction_id") or reaction.get("id") or reaction_index)
    normalized = {
        "id": f"ChemEagle-{image_name}-{reaction_id}",
        "reaction_type": reaction.get("reaction_type") or "unknown reaction",
        "substrates": [
            normalize_compound(item)
            for item in ensure_list(reaction.get("reactants"))
        ],
        "products": [
            normalize_compound(item)
            for item in ensure_list(reaction.get("products"))
        ],
        "catalysts": [],
        "additives": [],
        "reagents": [],
        "conditions": {},
        "targets": {},
        "source_modality": "image",
        "source_image": image_name,
        "source_image_path": image_path,
        "source_paper": source_paper or pdf_path.stem,
        "source_artifact": artifact_stem,
    }

    condition_contexts = condition_contexts or []
    condition_context_labels = condition_context_labels or set()

    for condition in ensure_list(reaction.get("conditions")):
        apply_condition(
            condition,
            normalized,
            context_available=bool(condition_contexts),
            context_labels=condition_context_labels,
        )
    for info in ensure_list(reaction.get("additional_info")):
        apply_additional_info(info, normalized)
    apply_reagents_condition_contexts(condition_contexts, normalized)

    if condition_contexts:
        normalized["condition_context_sources"] = [
            {"path": context["path"], "value": context["value"]}
            for context in condition_contexts
        ]

    if not normalized["targets"]:
        normalized.pop("targets")
    if not normalized["conditions"]:
        normalized["conditions"] = {}

    return normalized


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def text_values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values: List[str] = []
        for item in value:
            values.extend(text_values(item))
        return values
    if isinstance(value, dict):
        return []
    text = str(value).strip()
    return [] if text.casefold() in {"", "none", "null", "not specified"} else [text]


def clean_text(value: Any) -> str:
    return "; ".join(text_values(value))


def effective_condition_role(condition: Dict[str, Any]) -> str:
    refined_role = clean_text(condition.get("refined_role")).casefold().replace("_", " ")
    if refined_role and refined_role != "unknown":
        return refined_role
    return clean_text(condition.get("role")).casefold().replace("_", " ")


def valid_smiles(smiles: Any) -> str:
    text = clean_text(smiles)
    return "" if text.casefold() == "none" else text


def normalize_compound(item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        text = clean_text(item)
        return {"name": text} if text else {"name": ""}

    label = clean_text(item.get("label"))
    smiles = valid_smiles(item.get("smiles"))
    iupac_name = clean_text(item.get("iupac_name"))
    resolved_iupac_name = clean_text(item.get("resolved_iupac_name"))
    resolved_smiles = valid_smiles(item.get("resolved_smiles"))
    resolved_name_normalized = clean_text(item.get("resolved_name_normalized"))
    resolved_name = clean_text(item.get("resolved_name"))
    raw_name = clean_text(item.get("name"))
    text = clean_text(item.get("text"))
    name = (
        smiles
        or iupac_name
        or resolved_iupac_name
        or raw_name
        or resolved_name_normalized
        or resolved_name
        or text
        or label
    )
    normalized = {"name": name}
    if iupac_name:
        normalized["iupac_name"] = iupac_name
    if resolved_iupac_name:
        normalized["resolved_iupac_name"] = resolved_iupac_name
    if resolved_smiles:
        normalized["resolved_smiles"] = resolved_smiles
    if resolved_name_normalized:
        normalized["resolved_name_normalized"] = resolved_name_normalized
    if resolved_name:
        normalized["resolved_name"] = resolved_name
    if label:
        normalized["symbol"] = label
    for key in ("resolution_source", "resolution_method", "resolution_confidence", "resolution_evidence"):
        if item.get(key):
            normalized[key] = item[key]
    amount = clean_text(item.get("amount")) or parse_amount(text) or parse_amount(label)
    if amount:
        normalized["amount"] = amount
    return normalized


def apply_condition(
    condition: Any,
    normalized: Dict[str, Any],
    *,
    context_available: bool = False,
    context_labels: Optional[set] = None,
) -> None:
    if isinstance(condition, (str, list)):
        parse_free_text_condition(clean_text(condition), normalized)
        return
    if not isinstance(condition, dict):
        return

    role = effective_condition_role(condition)
    text = clean_text(condition.get("text")) or clean_text(condition.get("label"))
    if not role:
        parse_free_text_condition(text, normalized)
        return
    if should_skip_reagent_condition(condition, role, context_available, context_labels or set()):
        return

    if role in {"yield", "ee", "er", "dr"}:
        value = extract_metric(text, role) or text
        if value:
            normalized["targets"].setdefault(role, value)
        return

    if role in {"solvent", "temperature", "time", "atmosphere", "wavelength", "light", "light source", "volume", "concentration", "temperature_time"}:
        key = "light source" if role == "light" else role
        if text:
            normalized["conditions"].setdefault(key, text)
        return

    compounds = normalize_condition_compounds(condition)
    if role in {"catalyst", "catalysts", "ligand", "precatalyst"}:
        append_unique_compounds(normalized["catalysts"], compounds)
    elif role in {"additive", "additives", "base"}:
        append_unique_compounds(normalized["additives"], compounds)
    else:
        append_unique_compounds(normalized["reagents"], compounds)


def apply_additional_info(info: Any, normalized: Dict[str, Any]) -> None:
    if isinstance(info, str):
        parse_free_text_condition(info, normalized)
        return
    if not isinstance(info, dict):
        return

    role = clean_text(info.get("role") or info.get("type")).casefold()
    text = clean_text(info.get("text"))
    if role in {"yield", "ee", "er", "dr"}:
        value = extract_metric(text, role) or text
        if value:
            normalized["targets"][role] = value
        return

    for key, value in info.items():
        key_clean = clean_text(key).casefold()
        if key_clean in {"yield", "ee", "er", "dr"}:
            metric = extract_metric(value, key_clean) or clean_text(value)
            if metric:
                normalized["targets"][key_clean] = metric
        elif key_clean and clean_text(value):
            normalized.setdefault("additional_info", []).append({key: value})


def normalize_condition_compound(condition: Dict[str, Any]) -> Dict[str, Any]:
    text = clean_text(condition.get("text"))
    label = clean_text(condition.get("label"))
    notation = clean_text(condition.get("notation_in_image"))
    smiles = valid_smiles(condition.get("smiles"))
    iupac_name = clean_text(condition.get("iupac_name"))
    resolved_iupac_name = clean_text(condition.get("resolved_iupac_name"))
    resolved_smiles = valid_smiles(condition.get("resolved_smiles"))
    resolved_name_normalized = clean_text(condition.get("resolved_name_normalized"))
    resolved_name = clean_text(condition.get("resolved_name"))
    raw_name = clean_text(condition.get("name"))
    name = (
        smiles
        or iupac_name
        or resolved_iupac_name
        or raw_name
        or resolved_name_normalized
        or resolved_name
        or text
        or label
        or notation
    )
    compound = {"name": name}
    if iupac_name:
        compound["iupac_name"] = iupac_name
    if resolved_iupac_name:
        compound["resolved_iupac_name"] = resolved_iupac_name
    if resolved_smiles:
        compound["resolved_smiles"] = resolved_smiles
    if resolved_name_normalized:
        compound["resolved_name_normalized"] = resolved_name_normalized
    if resolved_name:
        compound["resolved_name"] = resolved_name
    if label:
        compound["symbol"] = label
    for key in (
        "original_role",
        "refined_role",
        "role_source",
        "role_reason",
        "resolution_source",
        "resolution_method",
        "resolution_confidence",
        "resolution_evidence",
    ):
        if condition.get(key):
            compound[key] = condition[key]
    amount = clean_text(condition.get("amount")) or parse_amount(text) or parse_amount(notation) or parse_amount(label)
    if amount:
        compound["amount"] = amount
    return compound


def normalize_condition_compounds(condition: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = text_values(condition.get("text"))
    if len(texts) <= 1:
        compound = normalize_condition_compound(condition)
        return [compound] if compound.get("name") else []

    compounds = []
    for text in texts:
        item = dict(condition)
        item["text"] = text
        if clean_text(item.get("label")) == clean_text(condition.get("text")):
            item.pop("label", None)
        compound = normalize_condition_compound(item)
        if compound.get("name"):
            compounds.append(compound)
    return compounds


def append_unique_compounds(target: List[Dict[str, Any]], compounds: List[Dict[str, Any]]) -> None:
    existing = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in target}
    for compound in compounds:
        if not compound.get("name"):
            continue
        key = json.dumps(compound, sort_keys=True, ensure_ascii=False)
        if key not in existing:
            target.append(compound)
            existing.add(key)


def should_skip_reagent_condition(
    condition: Dict[str, Any],
    role: str,
    context_available: bool,
    context_labels: set,
) -> bool:
    if not context_available or role not in {"reagent", "reagents"}:
        return False
    pieces = text_values(condition.get("label")) + text_values(condition.get("text")) + text_values(condition.get("smiles"))
    combined = " ; ".join(pieces)
    if not combined:
        return False
    if ";" in combined or "," in combined:
        return True
    combined_casefold = combined.casefold()
    return any(label and label.casefold() in combined_casefold for label in context_labels)


def collect_reagents_condition_contexts(value: Any, path: str = "") -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            nested_path = f"{path}.{key}" if path else str(key)
            if key == "reagents_conditions":
                contexts.append({"path": nested_path, "value": nested})
                continue
            contexts.extend(collect_reagents_condition_contexts(nested, nested_path))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            contexts.extend(collect_reagents_condition_contexts(nested, f"{path}[{index}]"))
    return contexts


def collect_condition_context_labels(contexts: List[Dict[str, Any]]) -> set:
    labels = set()
    for context in contexts:
        collect_labels_from_value(context.get("value"), labels)
    return labels


def collect_labels_from_value(value: Any, labels: set) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = clean_text(key)
            if key_text and key_text not in {"condition_summary"}:
                labels.add(key_text)
            if key in {"label", "text", "notation_in_image"}:
                for text in text_values(nested):
                    labels.add(text)
            collect_labels_from_value(nested, labels)
    elif isinstance(value, list):
        for nested in value:
            collect_labels_from_value(nested, labels)
    else:
        for text in text_values(value):
            labels.add(text)


def apply_reagents_condition_contexts(contexts: List[Dict[str, Any]], normalized: Dict[str, Any]) -> None:
    for context in contexts:
        apply_reagents_condition_value(context.get("value"), normalized)


def apply_reagents_condition_value(value: Any, normalized: Dict[str, Any], role_hint: str = "") -> None:
    if isinstance(value, dict):
        if "role" in value:
            apply_context_item(value, normalized, role_hint=role_hint)
            return
        for key, nested in value.items():
            if clean_text(key).casefold() == "condition_summary":
                continue
            apply_reagents_condition_value(nested, normalized, role_hint=clean_text(key))
        return
    if isinstance(value, list):
        if not role_hint:
            return
        for item in value:
            apply_reagents_condition_value(item, normalized, role_hint=role_hint)
        return
    if role_hint:
        apply_context_item({"role": role_hint, "text": value}, normalized, role_hint=role_hint)


def apply_context_item(item: Dict[str, Any], normalized: Dict[str, Any], role_hint: str = "") -> None:
    role = clean_text(item.get("role")) or role_hint
    role_lower = role.casefold()
    text = clean_text(item.get("text")) or clean_text(item.get("notation_in_image")) or clean_text(item.get("label"))
    if not role_lower:
        return
    if any(metric == role_lower for metric in ("yield", "ee", "er", "dr")):
        value = extract_metric(text, role_lower) or text
        if value:
            normalized["targets"].setdefault(role_lower, value)
        return
    if "catalyst" in role_lower or "ligand" in role_lower or "precatalyst" in role_lower:
        append_unique_compounds(normalized["catalysts"], normalize_condition_compounds(item))
    elif "base" in role_lower or "additive" in role_lower:
        append_unique_compounds(normalized["additives"], normalize_condition_compounds(item))
    elif "solvent" in role_lower:
        if text:
            normalized["conditions"].setdefault("solvent", text)
    elif "temperature_time" in role_lower:
        if text:
            normalized["conditions"].setdefault("temperature_time", text)
    elif "atmosphere" in role_lower:
        if text:
            normalized["conditions"].setdefault("atmosphere", text)
    elif role_lower in {"temperature", "time", "volume", "concentration", "wavelength", "light", "light source"}:
        key = "light source" if role_lower == "light" else role_lower
        if text:
            normalized["conditions"].setdefault(key, text)
    else:
        append_unique_compounds(normalized["reagents"], normalize_condition_compounds(item))


def parse_free_text_condition(text: str, normalized: Dict[str, Any]) -> None:
    text = clean_text(text)
    if not text:
        return
    for metric in ("yield", "ee", "er", "dr"):
        value = extract_metric(text, metric)
        if value:
            normalized["targets"].setdefault(metric, value)
    if "solvent" not in normalized["conditions"]:
        solvent = extract_labeled_value(text, "solvent")
        if solvent:
            normalized["conditions"]["solvent"] = solvent
    if "temperature" not in normalized["conditions"]:
        temp = extract_temperature(text)
        if temp:
            normalized["conditions"]["temperature"] = temp
    if "time" not in normalized["conditions"]:
        time_val = extract_time(text)
        if time_val:
            normalized["conditions"]["time"] = time_val


def extract_metric(text: Any, metric: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    if metric == "yield":
        if "yield" not in text.casefold() and re.search(r"(?i)\b(?:ee|er|dr)\b", text):
            return ""
        patterns = [
            r"(?i)([<>≥≤~]?\s*\d+(?:\.\d+)?\s*%?)\s*(?:yield|isolated)",
            r"(?i)yield\s*[:=]?\s*([<>≥≤~]?\s*\d+(?:\.\d+)?\s*%?)",
            r"(?i)^([<>≥≤~]?\s*\d+(?:\.\d+)?\s*%)$",
        ]
    elif metric == "ee":
        patterns = [
            r"(?i)([<>≥≤~]?\s*\d+(?:\.\d+)?\s*%?)\s*ee",
            r"(?i)ee\s*[:=]?\s*([<>≥≤~]?\s*\d+(?:\.\d+)?\s*%?)",
        ]
    elif metric == "er":
        patterns = [
            r"(?i)er\s*[:=]?\s*([<>≥≤~]?\s*\d+(?:\.\d+)?\s*[:/]\s*[<>≥≤~]?\s*\d+(?:\.\d+)?)",
            r"(?i)([<>≥≤~]?\s*\d+(?:\.\d+)?\s*[:/]\s*[<>≥≤~]?\s*\d+(?:\.\d+)?)\s*er",
        ]
    else:
        patterns = [
            r"(?i)dr\s*[:=]?\s*([<>≥≤~]?\s*\d+(?:\.\d+)?\s*[:/]\s*[<>≥≤~]?\s*\d+(?:\.\d+)?)",
            r"(?i)([<>≥≤~]?\s*\d+(?:\.\d+)?\s*[:/]\s*[<>≥≤~]?\s*\d+(?:\.\d+)?)\s*dr",
        ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def extract_labeled_value(text: str, label: str) -> str:
    match = re.search(rf"(?i){re.escape(label)}\s*[:=]\s*([^;,]+)", text)
    return match.group(1).strip() if match else ""


def extract_temperature(text: str) -> str:
    match = re.search(r"(?i)(-?\d+\s*(?:°C|掳C|C)\b|room temperature|rt\b)", text)
    return match.group(1).strip() if match else ""


def extract_time(text: str) -> str:
    match = re.search(r"(?i)(\d+(?:\.\d+)?\s*(?:h|hour|hours|min|minutes)\b)", text)
    return match.group(1).strip() if match else ""


def parse_amount(text: str) -> Optional[str]:
    text = clean_text(text)
    if not text:
        return None
    match = re.search(r"(?i)(\d+(?:\.\d+)?\s*(?:mol%|equiv|mmol|mg|g|mL|ml))", text)
    return match.group(1).strip() if match else None
