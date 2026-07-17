import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

WORKFLOW_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = WORKFLOW_DIR.parent
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from multimodal_structure_enrichment import apply_enrichment, load_cache
from split_by_substrate import has_valid_condition

from langgraph_workflow.benchmark_utils import (
    REACTION_TYPE_POLICIES,
    SUPPORTED_SOURCE_MODALITIES,
    generate_benchmark_packages_by_modality,
    normalize_source_modality,
)
from langgraph_workflow.paper_question_summary import (
    build_paper_question_option_summary,
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _option_distribution(questions: list[dict]) -> Dict[str, int]:
    counts = sorted({question["option_count"] for question in questions})
    return {
        str(count): sum(1 for question in questions if question["option_count"] == count)
        for count in counts
    }


def _validate_reaction_modalities(reactions: list[dict]) -> Dict[str, list[dict]]:
    partitions = {modality: [] for modality in SUPPORTED_SOURCE_MODALITIES}
    unexpected = set()
    for reaction in reactions:
        modality = normalize_source_modality(reaction.get("source_modality"))
        if modality not in partitions:
            unexpected.add(modality)
            continue
        clean_reaction = deepcopy(reaction)
        clean_reaction["source_modality"] = modality
        partitions[modality].append(clean_reaction)
    if unexpected:
        raise ValueError(
            "Unsupported or missing source_modality values: "
            + ", ".join(sorted(unexpected))
        )
    return partitions


def build_modality_benchmark_bundle(
    merged_payload: dict,
    cache_path: Path,
    reaction_type_policy: str = "required",
) -> dict:
    reactions = merged_payload.get("reactions")
    if not isinstance(reactions, list):
        raise ValueError("Merged payload must contain a reactions list")

    structure_cache = load_cache(Path(cache_path))
    if not structure_cache:
        raise ValueError(f"No structure cache entries found for {cache_path}")

    input_partitions = _validate_reaction_modalities(reactions)
    enriched_partitions = {}
    enrichment_stats = {}
    q1_data = []
    q2_data = []

    for modality in SUPPORTED_SOURCE_MODALITIES:
        enriched_payload, stats = apply_enrichment(
            {"reactions": input_partitions[modality]},
            modality,
            structure_cache,
            roles=("substrates", "products"),
        )
        enriched = enriched_payload["reactions"]
        q1_reactions = [reaction for reaction in enriched if has_valid_condition(reaction)]
        q2_reactions = [
            reaction
            for reaction in q1_reactions
            if any(
                substrate.get("parseable") and substrate.get("scaffold")
                for substrate in reaction.get("substrates", []) or []
            )
        ]
        enriched_partitions[modality] = enriched
        enrichment_stats[modality] = stats
        q1_data.extend(q1_reactions)
        q2_data.extend(q2_reactions)

    generated = generate_benchmark_packages_by_modality(
        q1_data,
        q2_data,
        reaction_type_policy=reaction_type_policy,
    )

    by_modality_summary = {}
    for modality in SUPPORTED_SOURCE_MODALITIES:
        modality_bundle = generated["by_modality"][modality]
        q1_questions = modality_bundle["q1"]["benchmark"]
        q2_questions = modality_bundle["q2"]["benchmark"]
        by_modality_summary[modality] = {
            "input_reactions": len(input_partitions[modality]),
            "structure_enrichment": enrichment_stats[modality],
            "q1_reactions": len(modality_bundle["q1_reactions"]),
            "q2_reactions": len(modality_bundle["q2_reactions"]),
            "q1_questions": len(q1_questions),
            "q1_review_questions": len(modality_bundle["q1"]["review_set"]),
            "q1_reaction_type_conflict_questions": modality_bundle["q1"]["report"].get(
                "reaction_type_conflict_questions", 0
            ),
            "q1_option_count_distribution": _option_distribution(q1_questions),
            "q2_questions": len(q2_questions),
            "q2_review_questions": len(modality_bundle["q2"]["review_set"]),
            "q2_reaction_type_conflict_questions": modality_bundle["q2"]["report"].get(
                "reaction_type_conflict_questions", 0
            ),
            "q2_option_count_distribution": _option_distribution(q2_questions),
        }

    generated["split_report"] = {
        "reaction_type_policy": reaction_type_policy,
        "total_reactions": len(reactions),
        "q1_reactions": len(q1_data),
        "q2_reactions": len(q2_data),
        "by_modality": {
            modality: {
                "input_reactions": len(input_partitions[modality]),
                "q1_reactions": len(generated["by_modality"][modality]["q1_reactions"]),
                "q2_reactions": len(generated["by_modality"][modality]["q2_reactions"]),
                "structure_enrichment": enrichment_stats[modality],
            }
            for modality in SUPPORTED_SOURCE_MODALITIES
        },
    }
    generated["summary"] = {
        "reaction_type_policy": reaction_type_policy,
        "input_reactions": len(reactions),
        "q1_questions": len(generated["q1"]["benchmark"]),
        "q2_questions": len(generated["q2"]["benchmark"]),
        "q1_reaction_type_conflict_questions": generated["q1"]["report"].get(
            "reaction_type_conflict_questions", 0
        ),
        "q2_reaction_type_conflict_questions": generated["q2"]["report"].get(
            "reaction_type_conflict_questions", 0
        ),
        "by_modality": by_modality_summary,
        "cross_modal_overlap": generated["cross_modal_overlap"],
    }
    generated["paper_question_option_summary"] = build_paper_question_option_summary(
        reactions,
        generated["question_option_counts"],
    )
    return generated


def validate_benchmark_bundle(bundle: dict) -> None:
    reaction_type_policy = bundle.get("reaction_type_policy", "required")
    seen_question_ids = set()
    for task in ("q1", "q2"):
        combined_questions = bundle[task]["benchmark"]
        expected_count = 0
        for modality in SUPPORTED_SOURCE_MODALITIES:
            questions = bundle["by_modality"][modality][task]["benchmark"]
            expected_count += len(questions)
            for question in questions:
                if question.get("source_modality") != modality:
                    raise ValueError(f"Mixed modality question: {question.get('id')}")
        if len(combined_questions) != expected_count:
            raise ValueError(f"Combined {task.upper()} question count is inconsistent")

        for question in combined_questions:
            question_id = question.get("id")
            if not question_id or question_id in seen_question_ids:
                raise ValueError(f"Duplicate or missing question id: {question_id}")
            seen_question_ids.add(question_id)
            options = question.get("options", [])
            option_ids = [option.get("option_id") for option in options]
            if question.get("option_count") != len(options):
                raise ValueError(f"option_count mismatch for {question_id}")
            if len(option_ids) != len(set(option_ids)) or None in option_ids:
                raise ValueError(f"Invalid option ids for {question_id}")
            option_id_set = set(option_ids)
            if not set(question.get("gold_option_ids", [])) <= option_id_set:
                raise ValueError(f"Invalid gold option ids for {question_id}")
            if set(question.get("gold_ranked_option_ids", [])) != option_id_set:
                raise ValueError(f"Invalid ranked option ids for {question_id}")
            if reaction_type_policy == "ignored":
                if "reaction_type" in question:
                    raise ValueError(
                        f"Ignored policy leaked reaction_type into {question_id}"
                    )
                if "Reaction type:" in str(question.get("question_en", "")):
                    raise ValueError(
                        f"Ignored policy leaked reaction type text into {question_id}"
                    )

        report_distribution = bundle[task]["report"]["option_count_distribution"]
        if report_distribution != _option_distribution(combined_questions):
            raise ValueError(f"{task.upper()} report option distribution is inconsistent")

    detail_ids = [row["question_id"] for row in bundle["question_option_counts"]]
    if set(detail_ids) != seen_question_ids or len(detail_ids) != len(seen_question_ids):
        raise ValueError("question_option_counts does not cover every question exactly once")
    paper_summary = bundle.get("paper_question_option_summary")
    if paper_summary is not None:
        if paper_summary["total_questions"] != len(seen_question_ids):
            raise ValueError("Paper summary question count is inconsistent")
        if sum(
            paper["total_questions"] for paper in paper_summary["papers"]
        ) != len(seen_question_ids):
            raise ValueError("Paper-level question totals are inconsistent")


def _artifact_payloads(bundle: dict) -> Dict[Path, Any]:
    payloads = {
        Path("Q1_substrate_to_condition.json"): [
            reaction
            for modality in SUPPORTED_SOURCE_MODALITIES
            for reaction in bundle["by_modality"][modality]["q1_reactions"]
        ],
        Path("Q2_condition_to_substrate.json"): [
            reaction
            for modality in SUPPORTED_SOURCE_MODALITIES
            for reaction in bundle["by_modality"][modality]["q2_reactions"]
        ],
        Path("q1q2_split_report.json"): bundle["split_report"],
        Path("Q1_benchmark.json"): bundle["q1"]["benchmark"],
        Path("Q1_benchmark_review.json"): bundle["q1"]["review_set"],
        Path("Q1_benchmark_report.json"): bundle["q1"]["report"],
        Path("Q2_benchmark.json"): bundle["q2"]["benchmark"],
        Path("Q2_benchmark_review.json"): bundle["q2"]["review_set"],
        Path("Q2_benchmark_report.json"): bundle["q2"]["report"],
        Path("benchmark_summary.json"): bundle["summary"],
        Path("question_option_counts.json"): bundle["question_option_counts"],
        Path("paper_question_option_summary.json"): bundle[
            "paper_question_option_summary"
        ],
    }
    for modality in SUPPORTED_SOURCE_MODALITIES:
        modality_bundle = bundle["by_modality"][modality]
        base = Path(modality)
        payloads.update(
            {
                base / "Q1_substrate_to_condition.json": modality_bundle["q1_reactions"],
                base / "Q2_condition_to_substrate.json": modality_bundle["q2_reactions"],
                base / "Q1_benchmark.json": modality_bundle["q1"]["benchmark"],
                base / "Q1_benchmark_review.json": modality_bundle["q1"]["review_set"],
                base / "Q1_benchmark_report.json": modality_bundle["q1"]["report"],
                base / "Q2_benchmark.json": modality_bundle["q2"]["benchmark"],
                base / "Q2_benchmark_review.json": modality_bundle["q2"]["review_set"],
                base / "Q2_benchmark_report.json": modality_bundle["q2"]["report"],
            }
        )
    return payloads


def write_benchmark_bundle(
    bundle: dict,
    benchmark_dir: Path,
    merged_path: Path,
    cache_path: Path,
    backup_existing: bool = True,
) -> dict:
    validate_benchmark_bundle(bundle)
    benchmark_dir = Path(benchmark_dir)
    merged_path = Path(merged_path)
    cache_path = Path(cache_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    backup_dir = None
    if backup_existing and benchmark_dir.exists() and any(benchmark_dir.iterdir()):
        backup_dir = benchmark_dir.parent / f"{benchmark_dir.name}_backup_pre_modality_{run_id}"
        suffix = 1
        while backup_dir.exists():
            backup_dir = benchmark_dir.parent / (
                f"{benchmark_dir.name}_backup_pre_modality_{run_id}_{suffix}"
            )
            suffix += 1
        shutil.copytree(benchmark_dir, backup_dir)

    payloads = _artifact_payloads(bundle)
    for relative_path, payload in payloads.items():
        atomic_write_json(benchmark_dir / relative_path, payload)

    artifact_hashes = {
        str(relative_path).replace("\\", "/"): file_sha256(benchmark_dir / relative_path)
        for relative_path in sorted(payloads, key=lambda path: str(path))
    }
    cache_sources = [
        cache_path,
        cache_path.parent / "multimodal_structure_parse_cache.json",
        cache_path.parent / "substrate_parse_cache.json",
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "mode": "offline_modality_regeneration",
        "reaction_type_policy": bundle.get("reaction_type_policy", "required"),
        "input": str(merged_path),
        "input_sha256": file_sha256(merged_path),
        "structure_cache_sources": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in cache_sources
            if path.exists()
        ],
        "backup_dir": str(backup_dir) if backup_dir else None,
        "historical_workflow_report_modified": False,
        "summary": bundle["summary"],
        "artifact_sha256": artifact_hashes,
    }
    atomic_write_json(benchmark_dir / "regeneration_manifest.json", manifest)
    return manifest


def regenerate_benchmark(
    merged_path: Path,
    benchmark_dir: Path,
    cache_path: Path,
    backup_existing: bool = True,
    reaction_type_policy: str = "required",
) -> dict:
    merged_payload = read_json(Path(merged_path))
    bundle = build_modality_benchmark_bundle(
        merged_payload,
        Path(cache_path),
        reaction_type_policy=reaction_type_policy,
    )
    return write_benchmark_bundle(
        bundle,
        benchmark_dir=Path(benchmark_dir),
        merged_path=Path(merged_path),
        cache_path=Path(cache_path),
        backup_existing=backup_existing,
    )


def regenerate_paper_question_summary(
    merged_path: Path,
    benchmark_dir: Path,
) -> dict:
    merged_path = Path(merged_path)
    benchmark_dir = Path(benchmark_dir)
    merged_payload = read_json(merged_path)
    question_option_counts = read_json(
        benchmark_dir / "question_option_counts.json"
    )
    summary = build_paper_question_option_summary(
        merged_payload.get("reactions", []),
        question_option_counts,
    )

    actual_option_counts = {}
    for task in ("Q1", "Q2"):
        for question in read_json(benchmark_dir / f"{task}_benchmark.json"):
            actual_option_counts[question["id"]] = len(question.get("options", []))
    for row in question_option_counts:
        question_id = row["question_id"]
        if actual_option_counts.get(question_id) != row["option_count"]:
            raise ValueError(f"Option count mismatch for {question_id}")

    output_path = benchmark_dir / "paper_question_option_summary.json"
    atomic_write_json(output_path, summary)

    manifest_path = benchmark_dir / "regeneration_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        manifest.setdefault("artifact_sha256", {})[
            "paper_question_option_summary.json"
        ] = file_sha256(output_path)
        manifest["paper_question_option_summary"] = {
            "total_papers": summary["total_papers"],
            "total_questions": summary["total_questions"],
            "questions_by_modality": summary["questions_by_modality"],
        }
        atomic_write_json(manifest_path, manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate Q1/Q2 benchmark artifacts independently by source modality."
    )
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--benchmark-dir", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--paper-summary-only", action="store_true")
    parser.add_argument(
        "--reaction-type-policy",
        choices=REACTION_TYPE_POLICIES,
        default="required",
        help="Use reaction type for filtering/grouping (required) or ignore it (ignored).",
    )
    args = parser.parse_args()
    if args.paper_summary_only:
        summary = regenerate_paper_question_summary(
            merged_path=args.merged,
            benchmark_dir=args.benchmark_dir,
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2))
        return
    if args.cache is None:
        parser.error("--cache is required unless --paper-summary-only is used")
    manifest = regenerate_benchmark(
        merged_path=args.merged,
        benchmark_dir=args.benchmark_dir,
        cache_path=args.cache,
        backup_existing=not args.no_backup,
        reaction_type_policy=args.reaction_type_policy,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
