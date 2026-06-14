"""Run isolated API regression checks for ChemEagle role refinement batching."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


WORKFLOW_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = WORKFLOW_DIR.parent
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from langgraph_workflow.chemeagle_adapter import refine_chemeagle_condition_roles  # noqa: E402
from langgraph_workflow.token_usage import (  # noqa: E402
    install_token_usage_tracking,
    make_run_id,
    summarize_token_usage,
    token_usage_context,
)


PRICE_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def pick_inputs(chemeagle_dir: Path) -> List[Path]:
    raw_iupac_dir = chemeagle_dir / "raw_iupac"
    raw_dir = chemeagle_dir / "raw"
    paths = sorted(raw_iupac_dir.glob("*.json"))
    if paths:
        return paths
    return sorted(raw_dir.glob("*.json"))


def iter_records(payload: Any) -> Iterable[Dict[str, Any]]:
    records = payload if isinstance(payload, list) else [payload]
    for record in records:
        if isinstance(record, dict):
            yield record


def condition_key(record: Dict[str, Any], reaction: Dict[str, Any], condition_index: int) -> Tuple[str, str, int]:
    reaction_id = str(reaction.get("reaction_id") or reaction.get("id") or "")
    image_name = str(record.get("image_name") or "")
    return image_name, reaction_id, condition_index


def condition_role_index(payload: Any) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    indexed: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for record in iter_records(payload):
        for reaction in record.get("reactions") or []:
            if not isinstance(reaction, dict):
                continue
            for index, condition in enumerate(reaction.get("conditions") or []):
                if isinstance(condition, dict):
                    indexed[condition_key(record, reaction, index)] = condition
    return indexed


def compare_to_baseline(baseline: Optional[Any], candidate: Any) -> Dict[str, Any]:
    if baseline is None:
        return {"baseline_available": False, "role_differences": []}
    baseline_index = condition_role_index(baseline)
    candidate_index = condition_role_index(candidate)
    differences = []
    for key in sorted(set(baseline_index) | set(candidate_index)):
        base = baseline_index.get(key)
        cand = candidate_index.get(key)
        if base is None or cand is None:
            differences.append(
                {
                    "image_name": key[0],
                    "reaction_id": key[1],
                    "condition_index": key[2],
                    "baseline_missing": base is None,
                    "candidate_missing": cand is None,
                }
            )
            continue
        base_role = base.get("refined_role") or base.get("role")
        cand_role = cand.get("refined_role") or cand.get("role")
        if base_role != cand_role:
            differences.append(
                {
                    "image_name": key[0],
                    "reaction_id": key[1],
                    "condition_index": key[2],
                    "baseline_role": base_role,
                    "candidate_role": cand_role,
                    "baseline_text": base.get("text") or base.get("label"),
                    "candidate_text": cand.get("text") or cand.get("label"),
                }
            )
    return {
        "baseline_available": True,
        "baseline_conditions": len(baseline_index),
        "candidate_conditions": len(candidate_index),
        "role_difference_count": len(differences),
        "role_differences": differences,
    }


def estimate_cost(usage_summary: Dict[str, Any], model: str) -> Dict[str, Any]:
    by_model = usage_summary.get("by_model", {}).get(model, {})
    prompt = by_model.get("prompt_tokens") or 0
    completion = by_model.get("completion_tokens") or 0
    prices = PRICE_PER_1M.get(model)
    if not prices:
        return {"model": model, "estimated_cost_usd": None}
    input_cost = prompt / 1_000_000 * prices["input"]
    output_cost = completion / 1_000_000 * prices["output"]
    return {
        "model": model,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "estimated_cost_usd": round(input_cost + output_cost, 6),
    }


def run_model(
    *,
    input_paths: List[Path],
    baseline_dir: Path,
    output_root: Path,
    model: str,
    api_key: str,
    base_url: str,
) -> Dict[str, Any]:
    model_dir = output_root / model
    outputs_dir = model_dir / "outputs"
    events_path = model_dir / "token_usage_events.jsonl"
    token_report_path = model_dir / "token_usage_report.json"
    role_report_path = model_dir / "role_refinement_eval_report.json"
    run_id = install_token_usage_tracking(events_path, run_id=make_run_id(), enabled=True, default_stage="chemeagle_role_refinement")
    result = {
        "created_at": datetime.now().isoformat(),
        "model": model,
        "base_url": base_url,
        "run_id": run_id,
        "inputs": [str(path) for path in input_paths],
        "outputs_dir": str(outputs_dir),
        "files": [],
    }
    aggregate_stats: Dict[str, Any] = {
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
    for input_path in input_paths:
        payload = read_json(input_path)
        pdf_name = input_path.name
        with token_usage_context("chemeagle_role_refinement", pdf_name):
            refined_payload, stats = refine_chemeagle_condition_roles(
                payload,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        output_path = outputs_dir / input_path.name
        write_json(output_path, refined_payload)
        baseline_path = baseline_dir / input_path.name
        baseline_payload = read_json(baseline_path) if baseline_path.exists() else None
        comparison = compare_to_baseline(baseline_payload, refined_payload)
        file_report = {
            "input": str(input_path),
            "output": str(output_path),
            "baseline": str(baseline_path) if baseline_path.exists() else None,
            "stats": stats,
            "comparison": comparison,
        }
        result["files"].append(file_report)
        for key, value in stats.items():
            if isinstance(value, int):
                aggregate_stats[key] = aggregate_stats.get(key, 0) + value
            elif key in {"role_refinement_batch_errors", "failed_reactions"} and isinstance(value, list):
                aggregate_stats.setdefault(key, []).extend(value)
    usage_summary = summarize_token_usage(events_path, token_report_path)
    result["aggregate_stats"] = aggregate_stats
    result["token_usage"] = usage_summary
    result["cost"] = estimate_cost(usage_summary, model)
    result["token_report"] = str(token_report_path)
    write_json(role_report_path, result)
    result["report"] = str(role_report_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chemeagle-dir",
        default=str(WORKFLOW_DIR / "output_pdf_full_structure_no_observer" / "chemeagle"),
    )
    parser.add_argument("--models", nargs="+", default=["gpt-4o", "gpt-5-mini"])
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or "https://hk.xty.app/v1")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    env_values = load_env_file(EXTRACT_DIR / ".env")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or env_values.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment or extract/.env")
    base_url = args.base_url or env_values.get("OPENAI_BASE_URL") or "https://hk.xty.app/v1"

    chemeagle_dir = Path(args.chemeagle_dir)
    input_paths = pick_inputs(chemeagle_dir)
    if not input_paths:
        raise SystemExit(f"No raw_iupac or raw json inputs found under {chemeagle_dir}")
    output_root = chemeagle_dir / "role_refined_batch_eval"
    baseline_dir = chemeagle_dir / "role_refined"
    summary = {
        "created_at": datetime.now().isoformat(),
        "chemeagle_dir": str(chemeagle_dir),
        "input_count": len(input_paths),
        "models": {},
    }
    for model in args.models:
        summary["models"][model] = run_model(
            input_paths=input_paths,
            baseline_dir=baseline_dir,
            output_root=output_root,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    printable = {
        "summary": str(summary_path),
        "models": {
            model: {
                "report": data["report"],
                "aggregate_stats": data["aggregate_stats"],
                "request_count": data["token_usage"].get("request_count"),
                "total_tokens": data["token_usage"].get("total_tokens"),
                "cost": data["cost"],
                "role_difference_count": sum(
                    item["comparison"].get("role_difference_count", 0)
                    for item in data["files"]
                ),
            }
            for model, data in summary["models"].items()
        },
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
