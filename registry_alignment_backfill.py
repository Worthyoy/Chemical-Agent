"""Deterministically re-apply same-paper registry alignment to reaction JSON.

This utility never reads PDFs or calls an LLM. It writes corrected copies,
filtered reactions, a text-only KG, a reaction summary CSV, and an audit log
under a separate output root.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from batch_si_extractor import SIExtractor
from cross_modal_kg import (
    TEXT_SUBSTRATE_NAME_POLICIES,
    TEXT_SUBSTRATE_NAME_POLICY_ORIGINAL_IF_AVAILABLE,
    build_cross_modal_kg,
)
from kg_to_reaction_csv import convert_kg_csv
from reaction_filter import filter_reaction_file


BACKFILL_SCHEMA_VERSION = "registry_alignment_backfill_v2"
COMPOUND_FIELDS = (
    "substrates",
    "products",
    "intermediates",
    "catalysts",
    "ligands",
    "other_components",
    "additives",
    "reagents",
)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _target_signature(reactions: Iterable[Dict[str, Any]]) -> List[Tuple[str, str]]:
    return [
        (
            str(reaction.get("id") or ""),
            json.dumps(reaction.get("targets") or {}, ensure_ascii=False, sort_keys=True),
        )
        for reaction in reactions
        if isinstance(reaction, dict)
    ]


def _item_at(items: Any, index: int) -> Any:
    if isinstance(items, list) and 0 <= index < len(items):
        return items[index]
    return None


def _changed_compounds(
    source_file: Path,
    before_reactions: List[Dict[str, Any]],
    after_reactions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    for reaction_index, (before, after) in enumerate(
        zip(before_reactions, after_reactions)
    ):
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        for field in COMPOUND_FIELDS:
            before_items = before.get(field)
            after_items = after.get(field)
            item_count = max(
                len(before_items) if isinstance(before_items, list) else 0,
                len(after_items) if isinstance(after_items, list) else 0,
            )
            for item_index in range(item_count):
                old_item = _item_at(before_items, item_index)
                new_item = _item_at(after_items, item_index)
                if old_item == new_item:
                    continue
                evidence = (
                    new_item.get("resolution_evidence")
                    if isinstance(new_item, dict)
                    else None
                )
                changes.append(
                    {
                        "source_file": source_file.name,
                        "reaction_index": reaction_index,
                        "reaction_id": str(after.get("id") or before.get("id") or ""),
                        "role": field,
                        "item_index": item_index,
                        "reason": (
                            evidence.get("match_type")
                            if isinstance(evidence, dict)
                            else "registry_alignment_update"
                        ),
                        "before": old_item,
                        "after": new_item,
                    }
                )
    return changes


def backfill_payload(
    payload: Dict[str, Any],
    extractor: SIExtractor,
    source_file: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, int]]:
    registry = payload.get("name_registry") or {}
    reactions = payload.get("reactions") or []
    if not isinstance(registry, dict):
        raise ValueError(f"name_registry must be an object: {source_file}")
    if not isinstance(reactions, list):
        raise ValueError(f"reactions must be a list: {source_file}")

    before_reactions = deepcopy(reactions)
    before_targets = _target_signature(before_reactions)
    aligned = extractor.align_names_in_reactions(reactions, registry)
    if len(aligned) != len(before_reactions):
        raise RuntimeError(f"Reaction count changed during registry backfill: {source_file}")
    if _target_signature(aligned) != before_targets:
        raise RuntimeError(f"Reaction targets changed during registry backfill: {source_file}")

    changes = _changed_compounds(source_file, before_reactions, aligned)
    stats = dict(getattr(extractor, "last_registry_resolution_stats", {}) or {})
    result = deepcopy(payload)
    result["reactions"] = aligned
    result["total_reactions"] = len(aligned)
    result.setdefault("postprocessing", {})["registry_alignment"] = {
        "schema_version": BACKFILL_SCHEMA_VERSION,
        "model_calls": 0,
        "changed_compounds": len(changes),
        "stats": stats,
    }
    return result, changes, stats


def run_backfill(
    input_dir: Path | str,
    output_root: Path | str,
    text_substrate_name_policy: str = (
        TEXT_SUBSTRATE_NAME_POLICY_ORIGINAL_IF_AVAILABLE
    ),
) -> Dict[str, Any]:
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    if text_substrate_name_policy not in TEXT_SUBSTRATE_NAME_POLICIES:
        raise ValueError(
            f"Unsupported text substrate name policy: {text_substrate_name_policy!r}"
        )
    input_paths = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not input_paths:
        raise ValueError(f"No JSON files found in input directory: {input_dir}")

    corrected_dir = output_root / "output"
    filtered_dir = output_root / "filtered" / "text"
    kg_dir = output_root / "kg_original"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    extractor = SIExtractor(
        api_key="registry-alignment-backfill-no-network",
        enable_stage2_audit=False,
    )
    audit_changes: List[Dict[str, Any]] = []
    file_reports = []
    filtered_paths = []
    total_reactions = 0

    for input_path in input_paths:
        payload = _read_json(input_path)
        corrected, changes, stats = backfill_payload(payload, extractor, input_path)
        corrected_path = corrected_dir / input_path.name
        filtered_path = filtered_dir / input_path.name
        _write_json(corrected_path, corrected)
        filter_result = filter_reaction_file(
            corrected_path,
            filtered_path,
            overwrite=True,
        )
        filtered_paths.append(filtered_path)
        audit_changes.extend(changes)
        reaction_count = len(corrected.get("reactions") or [])
        total_reactions += reaction_count
        file_reports.append(
            {
                "source_file": input_path.name,
                "reaction_count": reaction_count,
                "changed_compounds": len(changes),
                "registry_stats": stats,
                "filtered_reactions": filter_result.get("reactions"),
            }
        )

    kg_result = build_cross_modal_kg(
        text_reaction_paths=filtered_paths,
        chemeagle_raw_iupac_paths=[],
        output_dir=kg_dir,
        text_substrate_name_policy=text_substrate_name_policy,
    )
    kg_path = Path(kg_result["kg_triples_unified_multimodal"])
    summary_path = kg_dir / "reaction_summary_from_kg.csv"
    csv_result = convert_kg_csv(kg_path, summary_path)

    audit_path = output_root / "registry_alignment_audit.json"
    report = {
        "schema_version": BACKFILL_SCHEMA_VERSION,
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "model_calls": 0,
        "text_substrate_name_policy": text_substrate_name_policy,
        "files_processed": len(input_paths),
        "total_reactions": total_reactions,
        "changed_compounds": len(audit_changes),
        "files": file_reports,
        "changes": audit_changes,
        "kg": kg_result,
        "reaction_summary": csv_result,
    }
    _write_json(audit_path, report)
    report["audit_path"] = str(audit_path)
    report["reaction_summary_path"] = str(summary_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-apply deterministic same-paper registry alignment without PDF or LLM calls."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--text-substrate-name-policy",
        choices=sorted(TEXT_SUBSTRATE_NAME_POLICIES),
        default=TEXT_SUBSTRATE_NAME_POLICY_ORIGINAL_IF_AVAILABLE,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_backfill(
        input_dir=args.input_dir,
        output_root=args.output_root,
        text_substrate_name_policy=args.text_substrate_name_policy,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
