import csv
import hashlib
import json
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from batch_si_extractor import SIExtractor
from merge_filtered import merge_filtered_reactions_from_files
from normalize_reaction_types_llm import normalize_reaction_types_file
from reaction_filter import filter_reaction_file
from split_by_substrate import batch_llm_parse, split_reactions
from cross_modal_kg import build_cross_modal_kg
from multimodal_structure_enrichment import enrich_multimodal_structure
from symbol_resolution import canonical_paper_key, resolve_cross_modal_symbols, write_resolution_outputs
from langgraph_workflow.chemeagle_adapter import (
    enrich_chemeagle_raw_with_iupac,
    normalize_chemeagle_payload,
    refine_chemeagle_condition_roles,
    run_chemeagle_pdf,
)
from langgraph_workflow.benchmark_utils import (
    generate_q1_benchmark_package,
    generate_q2_benchmark,
)
from langgraph_workflow.token_usage import summarize_token_usage, token_usage_context
from langgraph_workflow.chemeagle_timing import summarize_chemeagle_timing

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class PipelineConfig:
    si_folder: Path
    output_dir: Path
    filtered_dir: Path
    intermediate_dir: Path
    api_key: str
    pdf_folder: Optional[Path] = None
    paper_map: Optional[Path] = None
    pages_per_chunk: int = 5
    screen_model: str = "gpt-5-mini"
    extract_model: str = "gpt-5-mini"
    split_model: str = "gpt-5-mini"
    chemeagle_role_refinement_model: str = "gpt-5-mini"
    overwrite: bool = False
    resume: bool = True
    limit: Optional[int] = None
    paper_name: Optional[str] = None
    base_url: str = "https://oneapi.xty.app/v1"
    max_parallel_pdfs: int = 2
    pipeline_version: str = "parallel_pdf_v1"
    skip_reaction_type_normalization: bool = False
    skip_chemeagle_normalization: bool = False
    skip_downstream_build: bool = False
    enable_cross_modal_symbol_resolution: bool = False
    enable_chemeagle_iupac_enrichment: bool = False
    enable_chemeagle_role_refinement: bool = False
    enable_multimodal_kg: bool = False
    enable_multimodal_structure_enrichment: bool = False
    enable_stage2_audit: bool = True
    input_mode: str = "supporting_information"
    enable_chemeagle: str = "auto"
    use_chemeagle: bool = False
    chemeagle_dir: Optional[Path] = None
    chemeagle_python: str = "python"
    chemeagle_pdf_model_size: str = "large"
    chemeagle_model_name: str = "/models/Qwen3-VL-32B-Instruct-AWQ"
    chemeagle_base_url: Optional[str] = None
    chemeagle_api_key: Optional[str] = None
    chemeagle_max_images: int = 0
    chemeagle_use_plan_observer: bool = False
    chemeagle_use_action_observer: bool = False
    token_usage_tracking: bool = True
    token_usage_run_id: Optional[str] = None

    @property
    def page_cache_dir(self) -> Path:
        return self.intermediate_dir / "page_cache"

    @property
    def entity_context_dir(self) -> Path:
        return self.intermediate_dir / "entity_context"

    @property
    def section_chunks_dir(self) -> Path:
        return self.intermediate_dir / "section_chunks"

    @property
    def registry_debug_dir(self) -> Path:
        return self.intermediate_dir / "registry_debug"

    @property
    def pipeline_cache_dir(self) -> Path:
        return self.intermediate_dir / "cache"

    @property
    def text_filtered_dir(self) -> Path:
        return self.filtered_dir / "text"

    @property
    def merged_dir(self) -> Path:
        return self.filtered_dir / "merged"

    @property
    def kg_dir(self) -> Path:
        return self.filtered_dir / "kg"

    @property
    def alignments_dir(self) -> Path:
        return self.filtered_dir / "alignments"

    @property
    def reports_dir(self) -> Path:
        return self.filtered_dir / "reports"

    @property
    def benchmark_dir(self) -> Path:
        return self.filtered_dir / "benchmark"

    @property
    def chemeagle_output_root(self) -> Path:
        return self.filtered_dir.parent / "chemeagle"

    @property
    def symbol_resolved_root(self) -> Path:
        return self.filtered_dir.parent / "symbol_resolved"

    @property
    def text_symbol_resolved_dir(self) -> Path:
        return self.symbol_resolved_root / "text"

    @property
    def symbol_resolution_report_path(self) -> Path:
        return self.symbol_resolved_root / "symbol_resolution_report.json"

    @property
    def symbol_resolution_candidates_path(self) -> Path:
        return self.symbol_resolved_root / "symbol_resolution_candidates.json"

    @property
    def symbol_resolution_validation_report_path(self) -> Path:
        return self.symbol_resolved_root / "symbol_resolution_validation_report.json"

    @property
    def chemeagle_image_root(self) -> Path:
        return self.chemeagle_output_root / "images"

    @property
    def chemeagle_raw_dir(self) -> Path:
        return self.chemeagle_output_root / "raw"

    @property
    def chemeagle_symbol_resolved_dir(self) -> Path:
        return self.chemeagle_output_root / "symbol_resolved"

    @property
    def chemeagle_raw_iupac_dir(self) -> Path:
        return self.chemeagle_output_root / "raw_iupac"

    @property
    def chemeagle_role_refined_dir(self) -> Path:
        return self.chemeagle_output_root / "role_refined"

    @property
    def chemeagle_role_refinement_report_path(self) -> Path:
        return self.chemeagle_output_root / "role_refinement_report.json"

    @property
    def chemeagle_iupac_cache_path(self) -> Path:
        return self.chemeagle_output_root / "iupac_cache.json"

    @property
    def chemeagle_normalized_dir(self) -> Path:
        return self.chemeagle_output_root / "normalized"

    @property
    def chemeagle_filtered_dir(self) -> Path:
        return self.chemeagle_output_root / "filtered"

    @property
    def token_usage_events_path(self) -> Path:
        return self.intermediate_dir / "token_usage_events.jsonl"

    @property
    def token_usage_report_path(self) -> Path:
        return self.reports_dir / "token_usage_report.json"

    @property
    def chemeagle_timing_events_path(self) -> Path:
        return self.intermediate_dir / "chemeagle_timing_events.jsonl"

    @property
    def chemeagle_timing_report_path(self) -> Path:
        return self.reports_dir / "chemeagle_timing_report.json"

    @property
    def multimodal_structure_enriched_dir(self) -> Path:
        return self.filtered_dir / "structure_enriched"

    @property
    def multimodal_structure_parse_cache_path(self) -> Path:
        return self.pipeline_cache_dir / "multimodal_structure_parse_cache.json"

    @property
    def multimodal_structure_enrichment_report_path(self) -> Path:
        return self.reports_dir / "multimodal_structure_enrichment_report.json"


def make_extractor(config: PipelineConfig) -> SIExtractor:
    return SIExtractor(
        api_key=config.api_key,
        pages_per_chunk=config.pages_per_chunk,
        screen_model=config.screen_model,
        extract_model=config.extract_model,
        base_url=config.base_url,
        enable_stage2_audit=config.enable_stage2_audit,
    )


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+" if path.exists() else "w"
    with open(path, mode, encoding="utf-8") as f:
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()


def safe_artifact_stem(raw_stem: str, max_prefix_len: int = 48) -> str:
    normalized = re.sub(r"[^\w.-]+", "_", raw_stem).strip("._")
    if not normalized:
        normalized = "file"
    digest = hashlib.md5(raw_stem.encode("utf-8")).hexdigest()[:8]
    prefix = normalized[:max_prefix_len].rstrip("._")
    if not prefix:
        prefix = "file"
    return f"{prefix}_{digest}"


def list_folder_pdf_files(folder: Optional[Path]) -> List[Path]:
    if folder is None or not folder.exists():
        return []
    return sorted(folder.glob("*.pdf"))


def list_pdf_files(config: PipelineConfig) -> List[Path]:
    pdfs = list_folder_pdf_files(config.si_folder)
    if config.paper_name:
        needle = config.paper_name.lower()
        pdfs = [p for p in pdfs if needle in p.name.lower()]
    if config.limit is not None:
        pdfs = pdfs[: config.limit]
    return pdfs


def file_lookup(files: List[Path]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in files:
        for key in (path.name, path.stem, canonical_paper_key(path.name)):
            key = (key or "").casefold()
            if key and key not in lookup:
                lookup[key] = path
    return lookup


def find_mapped_file(value: str, lookup: Dict[str, Path]) -> Optional[Path]:
    token = (value or "").strip()
    if not token:
        return None
    for key in (token, Path(token).name, Path(token).stem, canonical_paper_key(token)):
        key = (key or "").casefold()
        if key in lookup:
            return lookup[key]
    return None


def job_matches_filter(job: Dict, paper_name: Optional[str]) -> bool:
    if not paper_name:
        return True
    needle = paper_name.casefold()
    haystack = " ".join(
        str(job.get(key) or "")
        for key in ("paper_key", "pdf_name", "text_pdf_name", "image_pdf_name")
    ).casefold()
    return needle in haystack


def modality_artifact_stem(paper_key: str, modality: str, path: Optional[Path]) -> str:
    base = paper_key or (path.stem if path else "paper")
    path_hint = path.stem if path else "missing"
    return safe_artifact_stem(f"{base}__{modality}__{path_hint}")


def iter_with_progress(items, desc: str, unit: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True)


def source_metadata(pdf_path: Path, config: PipelineConfig) -> Dict:
    stat = pdf_path.stat()
    return {
        "pipeline_version": config.pipeline_version,
        "source_path": str(pdf_path),
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
        "screen_model": config.screen_model,
        "extract_model": config.extract_model,
        "split_model": config.split_model,
        "chemeagle_role_refinement_model": config.chemeagle_role_refinement_model,
        "pages_per_chunk": config.pages_per_chunk,
        "enable_stage2_audit": config.enable_stage2_audit,
    }


def metadata_matches(path: Path, expected: Dict) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    actual = payload.get("metadata")
    return isinstance(actual, dict) and all(actual.get(k) == v for k, v in expected.items())


def add_metadata(path: Path, metadata: Dict) -> None:
    payload = read_json(path)
    if isinstance(payload, dict):
        payload["metadata"] = metadata
        write_json(path, payload)


def build_symbol_index(registry: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    return {
        symbol: {
            "symbol": symbol,
            "name": name,
            "source": "name_registry",
        }
        for symbol, name in registry.items()
    }


def build_scaffold_mapping(registry: Dict[str, str], config: PipelineConfig) -> Dict[str, Dict]:
    names = sorted({name for name in registry.values() if name})
    if not names or OpenAI is None:
        return {}

    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
    )
    parsed = batch_llm_parse(names, client, model=config.split_model, batch_size=30)
    mapping = {}
    name_to_symbol = {name: symbol for symbol, name in registry.items()}
    for name, info in parsed.items():
        symbol = name_to_symbol.get(name)
        key = symbol or name
        mapping[key] = {
            "symbol": symbol,
            "name": name,
            "parseable": info.get("parseable", False),
            "scaffold": info.get("scaffold"),
            "substituents": info.get("substituents", []),
        }
    return mapping


def summarize_chemeagle_raw_payload(raw_payload) -> Dict:
    records = raw_payload if isinstance(raw_payload, list) else [raw_payload]
    total_images = 0
    failed_images = []
    raw_reactions = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        total_images += 1
        status = record.get("status")
        if status == "error":
            failed_images.append(
                {
                    "index": index,
                    "image": record.get("image") or record.get("image_path"),
                    "error": record.get("error"),
                }
            )
            continue
        reactions = record.get("reactions") or []
        reaction_count = record.get("reaction_count")
        raw_reactions += reaction_count if isinstance(reaction_count, int) else len(reactions)
    return {
        "total_images": total_images,
        "failed_images": failed_images,
        "raw_reactions": raw_reactions,
    }


def build_pdf_job(pdf_path: Path, config: PipelineConfig) -> Dict:
    artifact_stem = safe_artifact_stem(pdf_path.stem)
    return {
        "pdf_path": str(pdf_path),
        "pdf_name": pdf_path.name,
        "paper_key": canonical_paper_key(pdf_path.name) or pdf_path.stem,
        "mode": "legacy_joint",
        "run_text": True,
        "run_chemeagle": config.use_chemeagle,
        "text_pdf_path": str(pdf_path),
        "text_pdf_name": pdf_path.name,
        "image_pdf_path": str(pdf_path),
        "image_pdf_name": pdf_path.name,
        "artifact_stem": artifact_stem,
        "text_artifact_stem": artifact_stem,
        "image_artifact_stem": artifact_stem,
        "page_cache_path": str(config.page_cache_dir / f"{artifact_stem}.json"),
        "context_path": str(config.entity_context_dir / f"{artifact_stem}.json"),
        "section_debug_path": str(config.section_chunks_dir / f"{artifact_stem}.json"),
        "registry_debug_path": str(config.registry_debug_dir / f"{artifact_stem}.json"),
        "reaction_output_path": str(config.output_dir / f"{artifact_stem}.json"),
        "text_symbol_resolved_output_path": str(config.text_symbol_resolved_dir / f"{artifact_stem}.json"),
        "filtered_output_path": str(config.text_filtered_dir / f"{artifact_stem}.json"),
        "chemeagle_image_dir": str(config.chemeagle_image_root / artifact_stem),
        "chemeagle_raw_result_path": str(config.chemeagle_raw_dir / f"{artifact_stem}.json"),
        "chemeagle_symbol_resolved_output_path": str(config.chemeagle_symbol_resolved_dir / f"{artifact_stem}.json"),
        "chemeagle_raw_iupac_output_path": str(config.chemeagle_raw_iupac_dir / f"{artifact_stem}.json"),
        "chemeagle_role_refined_output_path": str(config.chemeagle_role_refined_dir / f"{artifact_stem}.json"),
        "chemeagle_normalized_output_path": str(config.chemeagle_normalized_dir / f"{artifact_stem}.json"),
        "chemeagle_filtered_output_path": str(config.chemeagle_filtered_dir / f"{artifact_stem}.json"),
    }


def build_paper_job(
    *,
    paper_key: str,
    text_pdf_path: Optional[Path],
    image_pdf_path: Optional[Path],
    config: PipelineConfig,
    mode: Optional[str] = None,
) -> Dict:
    resolved_key = paper_key or canonical_paper_key(
        (text_pdf_path or image_pdf_path).name if (text_pdf_path or image_pdf_path) else "paper"
    )
    text_stem = modality_artifact_stem(resolved_key, "text", text_pdf_path)
    image_stem = modality_artifact_stem(resolved_key, "image", image_pdf_path)
    primary_path = text_pdf_path or image_pdf_path
    if primary_path is None:
        raise ValueError("paper job requires at least one text or image PDF path")
    run_text = text_pdf_path is not None
    run_chemeagle = image_pdf_path is not None and config.use_chemeagle
    if mode is None:
        if run_text and run_chemeagle:
            mode = "dual_source"
        elif run_text:
            mode = "text_only"
        else:
            mode = "image_only"
    return {
        "pdf_path": str(primary_path),
        "pdf_name": resolved_key or primary_path.name,
        "paper_key": resolved_key,
        "mode": mode,
        "run_text": run_text,
        "run_chemeagle": run_chemeagle,
        "text_pdf_path": str(text_pdf_path) if text_pdf_path else None,
        "text_pdf_name": text_pdf_path.name if text_pdf_path else None,
        "image_pdf_path": str(image_pdf_path) if image_pdf_path else None,
        "image_pdf_name": image_pdf_path.name if image_pdf_path else None,
        "artifact_stem": safe_artifact_stem(resolved_key or primary_path.stem),
        "text_artifact_stem": text_stem,
        "image_artifact_stem": image_stem,
        "page_cache_path": str(config.page_cache_dir / f"{text_stem}.json"),
        "context_path": str(config.entity_context_dir / f"{text_stem}.json"),
        "section_debug_path": str(config.section_chunks_dir / f"{text_stem}.json"),
        "registry_debug_path": str(config.registry_debug_dir / f"{text_stem}.json"),
        "reaction_output_path": str(config.output_dir / f"{text_stem}.json"),
        "text_symbol_resolved_output_path": str(config.text_symbol_resolved_dir / f"{text_stem}.json"),
        "filtered_output_path": str(config.text_filtered_dir / f"{text_stem}.json"),
        "chemeagle_image_dir": str(config.chemeagle_image_root / image_stem),
        "chemeagle_raw_result_path": str(config.chemeagle_raw_dir / f"{image_stem}.json"),
        "chemeagle_symbol_resolved_output_path": str(config.chemeagle_symbol_resolved_dir / f"{image_stem}.json"),
        "chemeagle_raw_iupac_output_path": str(config.chemeagle_raw_iupac_dir / f"{image_stem}.json"),
        "chemeagle_role_refined_output_path": str(config.chemeagle_role_refined_dir / f"{image_stem}.json"),
        "chemeagle_normalized_output_path": str(config.chemeagle_normalized_dir / f"{image_stem}.json"),
        "chemeagle_filtered_output_path": str(config.chemeagle_filtered_dir / f"{image_stem}.json"),
    }


def build_dual_folder_jobs(config: PipelineConfig) -> tuple[List[Dict], Dict]:
    text_files = list_folder_pdf_files(config.si_folder)
    image_files = list_folder_pdf_files(config.pdf_folder)
    report = {
        "mode": "dual_folder",
        "si_folder": str(config.si_folder),
        "pdf_folder": str(config.pdf_folder) if config.pdf_folder else None,
        "paper_map": str(config.paper_map) if config.paper_map else None,
        "text_pdf_count": len(text_files),
        "image_pdf_count": len(image_files),
        "matched_papers": [],
        "text_only_papers": [],
        "image_only_papers": [],
        "unmatched_si_files": [],
        "unmatched_pdf_files": [],
        "duplicate_paper_keys": [],
    }
    jobs: List[Dict] = []
    used_text: set[Path] = set()
    used_image: set[Path] = set()

    if config.paper_map and config.paper_map.exists():
        text_lookup = file_lookup(text_files)
        image_lookup = file_lookup(image_files)
        with open(config.paper_map, "r", encoding="utf-8-sig", newline="") as f:
            for row_index, row in enumerate(csv.DictReader(f), start=1):
                paper_key = (row.get("paper_key") or "").strip()
                text_path = find_mapped_file(row.get("si_file") or row.get("text_file") or "", text_lookup)
                image_path = find_mapped_file(row.get("pdf_file") or row.get("image_file") or "", image_lookup)
                if not paper_key:
                    paper_key = canonical_paper_key(
                        (row.get("si_file") or row.get("pdf_file") or f"paper_{row_index}")
                    )
                if text_path:
                    used_text.add(text_path)
                if image_path:
                    used_image.add(image_path)
                if text_path or image_path:
                    jobs.append(
                        build_paper_job(
                            paper_key=paper_key,
                            text_pdf_path=text_path,
                            image_pdf_path=image_path,
                            config=config,
                        )
                    )

    if not jobs:
        text_by_key: Dict[str, List[Path]] = {}
        image_by_key: Dict[str, List[Path]] = {}
        for path in text_files:
            text_by_key.setdefault(canonical_paper_key(path.name) or path.stem.casefold(), []).append(path)
        for path in image_files:
            image_by_key.setdefault(canonical_paper_key(path.name) or path.stem.casefold(), []).append(path)
        for key in sorted(set(text_by_key) | set(image_by_key)):
            texts = text_by_key.get(key, [])
            images = image_by_key.get(key, [])
            if len(texts) > 1 or len(images) > 1:
                report["duplicate_paper_keys"].append(
                    {
                        "paper_key": key,
                        "text_files": [p.name for p in texts],
                        "image_files": [p.name for p in images],
                    }
                )
            for index in range(max(len(texts), len(images), 1)):
                text_path = texts[index] if index < len(texts) else None
                image_path = images[index] if index < len(images) else None
                job_key = key if max(len(texts), len(images), 1) == 1 else f"{key}_{index + 1}"
                jobs.append(
                    build_paper_job(
                        paper_key=job_key,
                        text_pdf_path=text_path,
                        image_pdf_path=image_path,
                        config=config,
                    )
                )
                if text_path:
                    used_text.add(text_path)
                if image_path:
                    used_image.add(image_path)

    for path in text_files:
        if path not in used_text:
            jobs.append(build_paper_job(paper_key=canonical_paper_key(path.name), text_pdf_path=path, image_pdf_path=None, config=config))
    for path in image_files:
        if path not in used_image:
            jobs.append(build_paper_job(paper_key=canonical_paper_key(path.name), text_pdf_path=None, image_pdf_path=path, config=config))

    jobs = [job for job in jobs if job_matches_filter(job, config.paper_name)]
    jobs.sort(key=lambda item: (item.get("paper_key") or "", item.get("mode") or ""))
    if config.limit is not None:
        jobs = jobs[: config.limit]

    for job in jobs:
        if job["mode"] == "dual_source":
            report["matched_papers"].append(job["paper_key"])
        elif job["mode"] == "text_only":
            report["text_only_papers"].append(job["paper_key"])
            if job.get("text_pdf_name"):
                report["unmatched_si_files"].append(job["text_pdf_name"])
        elif job["mode"] == "image_only":
            report["image_only_papers"].append(job["paper_key"])
            if job.get("image_pdf_name"):
                report["unmatched_pdf_files"].append(job["image_pdf_name"])
    return jobs, report


class PrepareJobsAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        for path in (
            self.config.page_cache_dir,
            self.config.entity_context_dir,
            self.config.section_chunks_dir,
            self.config.registry_debug_dir,
            self.config.pipeline_cache_dir,
            self.config.output_dir,
            self.config.filtered_dir,
            self.config.text_filtered_dir,
            self.config.merged_dir,
            self.config.kg_dir,
            self.config.alignments_dir,
            self.config.reports_dir,
            self.config.benchmark_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if self.config.enable_cross_modal_symbol_resolution:
            self.config.text_symbol_resolved_dir.mkdir(parents=True, exist_ok=True)
            self.config.symbol_resolved_root.mkdir(parents=True, exist_ok=True)
        if self.config.use_chemeagle:
            paths = [self.config.chemeagle_image_root, self.config.chemeagle_raw_dir]
            if self.config.enable_cross_modal_symbol_resolution:
                paths.append(self.config.chemeagle_symbol_resolved_dir)
            if self.config.enable_chemeagle_iupac_enrichment:
                paths.append(self.config.chemeagle_raw_iupac_dir)
            if self.config.enable_chemeagle_role_refinement:
                paths.append(self.config.chemeagle_role_refined_dir)
            if not self.config.skip_chemeagle_normalization:
                paths.extend([self.config.chemeagle_normalized_dir, self.config.chemeagle_filtered_dir])
            for path in paths:
                path.mkdir(parents=True, exist_ok=True)

        dual_input_report = None
        if self.config.input_mode == "dual_folder":
            jobs, dual_input_report = build_dual_folder_jobs(self.config)
            pdf_files = [
                Path(path)
                for job in jobs
                for path in (job.get("text_pdf_path"), job.get("image_pdf_path"))
                if path
            ]
        else:
            pdf_files = list_pdf_files(self.config)
            jobs = [build_pdf_job(pdf_path, self.config) for pdf_path in pdf_files]
        state["pdf_files"] = [str(p) for p in pdf_files]
        state["jobs"] = jobs
        state.setdefault("steps", {})["prepare_jobs"] = {
            "pdf_count": len(jobs),
            "si_folder": str(self.config.si_folder),
            "pdf_folder": str(self.config.pdf_folder) if self.config.pdf_folder else None,
            "paper_map": str(self.config.paper_map) if self.config.paper_map else None,
            "paper_name": self.config.paper_name,
            "limit": self.config.limit,
            "max_parallel_pdfs": self.config.max_parallel_pdfs,
            "input_mode": self.config.input_mode,
            "use_chemeagle": self.config.use_chemeagle,
            "skip_chemeagle_normalization": self.config.skip_chemeagle_normalization,
            "skip_downstream_build": self.config.skip_downstream_build,
            "enable_cross_modal_symbol_resolution": self.config.enable_cross_modal_symbol_resolution,
            "enable_chemeagle_iupac_enrichment": self.config.enable_chemeagle_iupac_enrichment,
            "enable_chemeagle_role_refinement": self.config.enable_chemeagle_role_refinement,
            "enable_multimodal_kg": self.config.enable_multimodal_kg,
            "enable_multimodal_structure_enrichment": self.config.enable_multimodal_structure_enrichment,
            "multimodal_structure_enrichment_effective": (
                "required_by_multimodal_kg" if self.config.enable_multimodal_kg else "disabled"
            ),
            "dual_input_report": dual_input_report,
            "jobs": jobs,
        }
        return {
            "pdf_files": state["pdf_files"],
            "jobs": jobs,
            "steps": state["steps"],
        }


class ProcessPDFAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        job = state["job"]
        started_at = time.perf_counter()
        run_text = bool(job.get("run_text", True))
        run_chemeagle = bool(job.get("run_chemeagle", self.config.use_chemeagle))
        can_symbol_resolve = run_text and run_chemeagle and self.config.enable_cross_modal_symbol_resolution
        result = {
            "pdf": job["pdf_path"],
            "pdf_name": job["pdf_name"],
            "paper_key": job.get("paper_key"),
            "mode": job.get("mode", "legacy_joint"),
            "text_pdf": job.get("text_pdf_path"),
            "image_pdf": job.get("image_pdf_path"),
            "artifact_stem": job["artifact_stem"],
            "status": "failed",
            "text_status": "pending" if run_text else "skipped",
            "chemeagle_status": "pending" if run_chemeagle else "skipped",
            "chemeagle_raw_status": "pending" if run_chemeagle else "skipped",
            "symbol_resolution_status": (
                "pending" if can_symbol_resolve else "skipped"
            ),
            "chemeagle_iupac_status": (
                "skipped"
                if not run_chemeagle or not self.config.enable_chemeagle_iupac_enrichment
                else "pending"
            ),
            "chemeagle_role_refinement_status": (
                "skipped"
                if not run_chemeagle or not self.config.enable_chemeagle_role_refinement
                else "pending"
            ),
            "chemeagle_normalization_status": (
                "skipped"
                if not run_chemeagle or self.config.skip_chemeagle_normalization
                else "pending"
            ),
            "paths": {
                "page_cache": job["page_cache_path"],
                "context": job["context_path"],
                "section_debug": job["section_debug_path"],
                "registry_debug": job["registry_debug_path"],
                "reaction_output": job["reaction_output_path"],
                "filtered_output": job["filtered_output_path"],
                "text_reaction_output": job["reaction_output_path"],
                "text_symbol_resolved_output": job["text_symbol_resolved_output_path"],
                "text_filtered_output": job["filtered_output_path"],
                "chemeagle_image_dir": job["chemeagle_image_dir"],
                "chemeagle_raw_output": job["chemeagle_raw_result_path"],
                "chemeagle_symbol_resolved_output": job["chemeagle_symbol_resolved_output_path"],
                "chemeagle_raw_iupac_output": job["chemeagle_raw_iupac_output_path"],
                "chemeagle_role_refined_output": job["chemeagle_role_refined_output_path"],
                "chemeagle_normalized_output": job["chemeagle_normalized_output_path"],
                "chemeagle_filtered_output": job["chemeagle_filtered_output_path"],
            },
            "cache": {},
            "counts": {},
            "chemeagle": None,
            "error": None,
        }

        errors = []
        reaction_output_path = Path(job["reaction_output_path"])
        if run_text:
            try:
                reaction_output_path = self._run_text_branch(job, result)
            except Exception as exc:
                result["text_status"] = "failed"
                errors.append(
                    {
                        "stage": "text",
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
        if run_chemeagle:
            self._run_chemeagle_branch(job, result)
        if run_text and result.get("text_status") == "success":
            filter_input_path = (
                Path(job["text_symbol_resolved_output_path"])
                if can_symbol_resolve and Path(job["text_symbol_resolved_output_path"]).exists()
                else reaction_output_path
            )
            try:
                self._run_text_filter(job, result, filter_input_path)
            except Exception as exc:
                result["text_status"] = "failed"
                errors.append(
                    {
                        "stage": "text_filter",
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                )
        if errors:
            result["error"] = {"branches": errors}
        if (
            result.get("text_status") == "success"
            or result.get("chemeagle_raw_status") == "success"
            or result.get("chemeagle_normalization_status") == "success"
        ):
            result["status"] = "success"

        result["elapsed_seconds"] = round(time.perf_counter() - started_at, 3)
        return {"pdf_results": [result]}

    def _write_section_debug(
        self,
        *,
        job: Dict,
        pdf_path: Path,
        pages: List[Dict],
        extractor: SIExtractor,
        metadata: Dict,
    ) -> Optional[Path]:
        section_debug_path = Path(job["section_debug_path"])
        debug = getattr(extractor, "last_section_debug", {}) or {}
        if not debug and pages:
            extractor.get_runtime_sections(pages)
            debug = getattr(extractor, "last_section_debug", {}) or {}
        if not debug:
            return None
        payload = {
            "source": str(pdf_path),
            "pdf_name": pdf_path.name,
            "paper_key": job.get("paper_key"),
            "source_modality": "text",
            "created_at": datetime.now().isoformat(),
            "section_debug_schema": "toc_section_chunks_v1",
            "metadata": metadata,
            "source_method": debug.get("source"),
            "toc_page_nums": debug.get("toc_page_nums") or [],
            "toc_page_offset": debug.get("toc_page_offset"),
            "toc_sections": debug.get("toc_sections") or [],
            "toc_error": debug.get("toc_error"),
            "raw_sections": debug.get("raw_sections") or [],
            "section_chunks": debug.get("section_chunks") or [],
            "registry_section_selection": debug.get("registry_section_selection") or {},
            "selected_registry_sections": debug.get("selected_registry_sections") or [],
            "selected_registry_section_chunks": debug.get("selected_registry_section_chunks") or [],
            "registry_selector_raw_preview": debug.get("registry_selector_raw_preview"),
            "registry_verifier_raw_preview": debug.get("registry_verifier_raw_preview"),
            "registry_selector_error": debug.get("registry_selector_error"),
            "registry_verifier_error": debug.get("registry_verifier_error"),
            "stats": debug.get("stats") or {},
        }
        write_json(section_debug_path, payload)
        return section_debug_path

    def _write_registry_debug(
        self,
        *,
        job: Dict,
        pdf_path: Path,
        extractor: SIExtractor,
        metadata: Dict,
    ) -> Optional[Path]:
        registry_debug_path = Path(job["registry_debug_path"])
        debug = getattr(extractor, "last_registry_debug", {}) or {}
        if not debug:
            return None
        summary = debug.get("summary") or {}
        payload = {
            "source": str(pdf_path),
            "pdf_name": pdf_path.name,
            "paper_key": job.get("paper_key"),
            "source_modality": "text",
            "created_at": datetime.now().isoformat(),
            "registry_debug_schema": "registry_raw_only_v1",
            "metadata": metadata,
            "strategy": debug.get("strategy"),
            "status": debug.get("status"),
            "section_selection": debug.get("section_selection") or {},
            "selection_stats": debug.get("selection_stats") or {},
            "section_debug": debug.get("section_debug") or {},
            "chunks": debug.get("chunks") or [],
            "summary": {
                "raw_registry_merged": summary.get("raw_registry_merged") or {},
                "final_registry": summary.get("final_registry") or {},
                "registry_raw_count": summary.get("registry_raw_count", 0),
                "registry_final_count": summary.get("registry_final_count", 0),
            },
        }
        write_json(registry_debug_path, payload)
        return registry_debug_path

    def _run_text_branch(self, job: Dict, result: Dict) -> Path:
        pdf_path = Path(job.get("text_pdf_path") or job["pdf_path"])
        metadata = source_metadata(pdf_path, self.config)
        metadata["paper_key"] = job.get("paper_key")
        metadata["source_modality"] = "text"
        metadata["source_pdf"] = str(pdf_path)
        extractor = make_extractor(self.config)

        page_cache_path = Path(job["page_cache_path"])
        if self.config.resume and not self.config.overwrite and metadata_matches(page_cache_path, metadata):
            page_payload = read_json(page_cache_path)
            pages = page_payload.get("pages", [])
            result["cache"]["page_cache"] = "hit"
        else:
            pages = extractor.extract_text_by_pages(str(pdf_path))
            write_json(
                page_cache_path,
                {
                    "source": str(pdf_path),
                    "cached_at": datetime.now().isoformat(),
                    "total_pages": len(pages),
                    "pages": pages,
                    "metadata": metadata,
                },
            )
            result["cache"]["page_cache"] = "miss"
        result["counts"]["pages"] = len(pages)

        context_path = Path(job["context_path"])
        if self.config.resume and not self.config.overwrite and metadata_matches(context_path, metadata):
            entity_context = read_json(context_path)
            result["cache"]["entity_context"] = "hit"
            if not Path(job["section_debug_path"]).exists():
                with token_usage_context("text_section_chunking_debug", pdf_path.name):
                    section_debug_path = self._write_section_debug(
                        job=job,
                        pdf_path=pdf_path,
                        pages=pages,
                        extractor=extractor,
                        metadata=metadata,
                    )
                result["cache"]["section_debug"] = "rebuilt" if section_debug_path else "skipped"
            else:
                result["cache"]["section_debug"] = "hit"
            result["cache"]["registry_debug"] = (
                "hit" if Path(job["registry_debug_path"]).exists() else "missing"
            )
        else:
            with token_usage_context("text_name_registry", pdf_path.name):
                registry = extractor.extract_name_registry(
                    pages,
                    max_scan_pages=20,
                    pages_per_chunk=5,
                )
            gp_texts = extractor.extract_general_procedure_texts(pages)
            symbol_index = build_symbol_index(registry)
            with token_usage_context("text_registry_scaffold_parse", pdf_path.name):
                scaffold_mapping = build_scaffold_mapping(registry, self.config)
            registry_validation_stats = getattr(extractor, "last_registry_validation_stats", {}) or {}
            entity_context = {
                "source": str(pdf_path),
                "paper_key": job.get("paper_key"),
                "source_modality": "text",
                "source_pdf": str(pdf_path),
                "created_at": datetime.now().isoformat(),
                "name_registry": registry,
                "general_procedures": gp_texts,
                "substrate_index": symbol_index,
                "product_index": symbol_index,
                "symbol_name_mapping": registry,
                "scaffold_substituent_mapping": scaffold_mapping,
                "section_debug_path": job["section_debug_path"],
                "registry_debug_path": job["registry_debug_path"],
                "stats": {
                    "total_pages": len(pages),
                    "registry_size": len(registry),
                    "registry_raw_count": registry_validation_stats.get("registry_raw_count", len(registry)),
                    "registry_final_count": registry_validation_stats.get("registry_final_count", len(registry)),
                    "section_chunking_enabled": registry_validation_stats.get("section_chunking_enabled", False),
                    "section_chunking_source": registry_validation_stats.get("section_chunking_source", "fixed_fallback"),
                    "section_scope": registry_validation_stats.get("section_scope", "full_document"),
                    "raw_section_count": registry_validation_stats.get("raw_section_count", 0),
                    "section_chunk_count": registry_validation_stats.get("section_chunk_count", 0),
                    "section_chunk_max_pages": registry_validation_stats.get("section_chunk_max_pages", 5),
                    "section_chunk_max_chars": registry_validation_stats.get("section_chunk_max_chars", 8000),
                    "section_count": registry_validation_stats.get("section_count", 0),
                    "registry_chunk_strategy": registry_validation_stats.get("registry_chunk_strategy", "fixed_pages"),
                    "registry_chunks_total": registry_validation_stats.get("registry_chunks_total", 0),
                    "registry_selection_strategy": registry_validation_stats.get("registry_selection_strategy", "skipped"),
                    "registry_selection_status": registry_validation_stats.get("registry_selection_status", "not_started"),
                    "registry_sections_selected": registry_validation_stats.get("registry_sections_selected", 0),
                    "registry_sections_rejected": registry_validation_stats.get("registry_sections_rejected", 0),
                    "registry_fallback_scan_pages": registry_validation_stats.get("registry_fallback_scan_pages", 20),
                    "registry_selector_error": registry_validation_stats.get("registry_selector_error"),
                    "registry_verifier_error": registry_validation_stats.get("registry_verifier_error"),
                    "gp_templates": len(gp_texts),
                    "scaffold_mappings": len(scaffold_mapping),
                },
                "metadata": metadata,
            }
            write_json(context_path, entity_context)
            section_debug_path = self._write_section_debug(
                job=job,
                pdf_path=pdf_path,
                pages=pages,
                extractor=extractor,
                metadata=metadata,
            )
            registry_debug_path = self._write_registry_debug(
                job=job,
                pdf_path=pdf_path,
                extractor=extractor,
                metadata=metadata,
            )
            result["cache"]["entity_context"] = "miss"
            result["cache"]["section_debug"] = "miss" if section_debug_path else "skipped"
            result["cache"]["registry_debug"] = "miss" if registry_debug_path else "skipped"
        result["counts"]["registry_size"] = len(entity_context.get("name_registry", {}))
        result["counts"]["gp_templates"] = len(entity_context.get("general_procedures", {}))

        reaction_output_path = Path(job["reaction_output_path"])
        reaction_payload = {}
        if self.config.resume and not self.config.overwrite and metadata_matches(reaction_output_path, metadata):
            reaction_payload = read_json(reaction_output_path)
            reactions = reaction_payload.get("reactions", [])
            result["cache"]["reaction_output"] = "hit"
        else:
            entity_context = dict(entity_context)
            entity_context["_context_path"] = str(context_path)
            with token_usage_context("text_reaction_extraction", pdf_path.name):
                reactions = extractor.process_pages_with_context(
                    str(pdf_path),
                    pages,
                    entity_context,
                    self.config.output_dir,
                    output_path=reaction_output_path,
                )
            if reactions is None:
                reactions = []
                if not reaction_output_path.exists():
                    write_json(
                        reaction_output_path,
                        {
                            "source": str(pdf_path),
                            "paper_key": job.get("paper_key"),
                            "source_modality": "text",
                            "source_pdf": str(pdf_path),
                            "extracted_at": datetime.now().isoformat(),
                            "total_reactions": 0,
                            "name_registry": entity_context.get("name_registry", {}),
                            "general_procedures": entity_context.get("general_procedures", {}),
                            "entity_context_path": str(context_path),
                            "stats": {"no_reaction_chunks": True},
                            "reactions": [],
                        },
                    )
            add_metadata(reaction_output_path, metadata)
            result["cache"]["reaction_output"] = "miss"
        result["counts"]["reactions"] = len(reactions)
        if not reaction_payload and reaction_output_path.exists():
            try:
                reaction_payload = read_json(reaction_output_path)
            except Exception:
                reaction_payload = {}
        result["counts"]["stage2_audit_recovered"] = (
            reaction_payload.get("stats", {}).get("stage2_audit_recovered", 0)
            if isinstance(reaction_payload, dict)
            else 0
        )
        result["text_status"] = "success"
        return reaction_output_path

    def _run_text_filter(self, job: Dict, result: Dict, input_path: Path) -> None:
        filter_result = filter_reaction_file(
            input_path,
            Path(job["filtered_output_path"]),
            overwrite=self.config.overwrite,
        )
        filter_result["input"] = str(input_path)
        filter_result["filter_input_stage"] = (
            "symbol_resolved"
            if Path(input_path) == Path(job["text_symbol_resolved_output_path"])
            else "raw"
        )
        filter_result["symbol_resolution_enabled"] = self.config.enable_cross_modal_symbol_resolution
        result["filter"] = filter_result
        if filter_result.get("reactions") is None and Path(job["filtered_output_path"]).exists():
            filtered_payload = read_json(Path(job["filtered_output_path"]))
            result["counts"]["filtered_reactions"] = len(filtered_payload.get("reactions", []))
        else:
            result["counts"]["filtered_reactions"] = filter_result.get("reactions", 0)

    def _run_chemeagle_branch(self, job: Dict, result: Dict) -> None:
        chemeagle_result = {
            "status": "failed",
            "raw_status": "pending",
            "iupac_status": "skipped" if not self.config.enable_chemeagle_iupac_enrichment else "pending",
            "role_refinement_status": "skipped" if not self.config.enable_chemeagle_role_refinement else "pending",
            "symbol_resolution_status": "skipped" if not self.config.enable_cross_modal_symbol_resolution else "pending",
            "normalization_status": "skipped" if self.config.skip_chemeagle_normalization else "pending",
            "raw_output": job["chemeagle_raw_result_path"],
            "symbol_resolved_output": job["chemeagle_symbol_resolved_output_path"],
            "raw_iupac_output": job["chemeagle_raw_iupac_output_path"],
            "role_refined_output": job["chemeagle_role_refined_output_path"],
            "normalized_output": job["chemeagle_normalized_output_path"],
            "filtered_output": job["chemeagle_filtered_output_path"],
            "image_dir": job["chemeagle_image_dir"],
            "error": None,
        }
        try:
            if not self.config.chemeagle_dir:
                raise ValueError("ChemEagle directory is not configured.")
            pdf_path = Path(job.get("image_pdf_path") or job["pdf_path"])
            metadata = source_metadata(pdf_path, self.config)
            metadata["paper_key"] = job.get("paper_key")
            metadata["source_modality"] = "image"
            metadata["source_pdf"] = str(pdf_path)
            text_raw_path = Path(job["reaction_output_path"])
            text_symbol_resolved_path = Path(job["text_symbol_resolved_output_path"])
            raw_path = Path(job["chemeagle_raw_result_path"])
            symbol_resolved_path = Path(job["chemeagle_symbol_resolved_output_path"])
            raw_iupac_path = Path(job["chemeagle_raw_iupac_output_path"])
            role_refined_path = Path(job["chemeagle_role_refined_output_path"])
            normalized_path = Path(job["chemeagle_normalized_output_path"])
            filtered_path = Path(job["chemeagle_filtered_output_path"])
            image_dir = Path(job["chemeagle_image_dir"])

            if self.config.resume and not self.config.overwrite and raw_path.exists():
                chemeagle_result["cache"] = "hit"
            else:
                with token_usage_context("chemeagle_subprocess", pdf_path.name):
                    subprocess_result = run_chemeagle_pdf(
                        chemeagle_python=self.config.chemeagle_python,
                        chemeagle_dir=Path(self.config.chemeagle_dir),
                        pdf_path=pdf_path,
                        image_dir=image_dir,
                        raw_result_path=raw_path,
                        pdf_model_size=self.config.chemeagle_pdf_model_size,
                        model_name=self.config.chemeagle_model_name,
                        base_url=self.config.chemeagle_base_url,
                        api_key=self.config.chemeagle_api_key,
                        max_images=self.config.chemeagle_max_images,
                        use_plan_observer=self.config.chemeagle_use_plan_observer,
                        use_action_observer=self.config.chemeagle_use_action_observer,
                        timing_events_path=self.config.chemeagle_timing_events_path,
                        timing_report_path=self.config.chemeagle_timing_report_path,
                        timing_run_id=self.config.token_usage_run_id,
                    )
                chemeagle_result["cache"] = "miss"
                chemeagle_result["subprocess"] = subprocess_result

            raw_payload = read_json(raw_path)
            raw_summary = summarize_chemeagle_raw_payload(raw_payload)
            chemeagle_result.update(raw_summary)
            chemeagle_result["raw_status"] = "success"
            result["chemeagle_raw_status"] = "success"
            result["counts"]["chemeagle_images"] = raw_summary["total_images"]
            result["counts"]["chemeagle_raw_reactions"] = raw_summary["raw_reactions"]

            chemeagle_iupac_input_payload = raw_payload
            chemeagle_iupac_input_path = raw_path
            can_symbol_resolve = (
                self.config.enable_cross_modal_symbol_resolution
                and bool(job.get("run_text", True))
                and text_raw_path.exists()
            )
            if can_symbol_resolve:
                try:
                    text_payload_for_symbol = read_json(text_raw_path)
                    if isinstance(text_payload_for_symbol, dict):
                        text_payload_for_symbol.setdefault("paper_key", job.get("paper_key"))
                    image_payload_for_symbol = raw_payload
                    image_records = (
                        image_payload_for_symbol
                        if isinstance(image_payload_for_symbol, list)
                        else [image_payload_for_symbol]
                    )
                    for record in image_records:
                        if isinstance(record, dict):
                            record.setdefault("paper_key", job.get("paper_key"))
                    resolved_text, resolved_image, symbol_report = resolve_cross_modal_symbols(
                        text_payload_for_symbol,
                        image_payload_for_symbol,
                    )
                    symbol_report["cache"] = "refreshed"
                    write_resolution_outputs(
                        text_output_path=text_symbol_resolved_path,
                        image_output_path=symbol_resolved_path,
                        report_path=self.config.symbol_resolution_report_path,
                        candidates_path=self.config.symbol_resolution_candidates_path,
                        validation_path=self.config.symbol_resolution_validation_report_path,
                        resolved_text=resolved_text,
                        resolved_image=resolved_image,
                        report=symbol_report,
                    )
                    chemeagle_iupac_input_payload = resolved_image
                    chemeagle_iupac_input_path = symbol_resolved_path
                    chemeagle_result["symbol_resolution_status"] = "success"
                    chemeagle_result["symbol_resolution"] = {
                        **{k: v for k, v in symbol_report.items() if k != "candidates"},
                        "candidates": symbol_report.get("candidates", []),
                        "text_output": str(text_symbol_resolved_path),
                        "image_output": str(symbol_resolved_path),
                        "report_output": str(self.config.symbol_resolution_report_path),
                        "candidates_output": str(self.config.symbol_resolution_candidates_path),
                        "validation_output": str(self.config.symbol_resolution_validation_report_path),
                    }
                    result["symbol_resolution_status"] = "success"
                    result["counts"]["symbol_text_entities_resolved"] = symbol_report.get("text_entities_resolved", 0)
                    result["counts"]["symbol_image_entities_resolved"] = symbol_report.get("image_entities_resolved", 0)
                    result["counts"]["symbol_resolution_candidates"] = symbol_report.get("candidate_count", 0)
                    validation_report = symbol_report.get("validation", {})
                    result["counts"]["symbol_resolution_invalid"] = validation_report.get("invalid_resolution_count", 0)
                except Exception as exc:
                    chemeagle_result["symbol_resolution_status"] = "failed"
                    chemeagle_result["symbol_resolution"] = {
                        "status": "failed",
                        "text_output": str(text_symbol_resolved_path),
                        "image_output": str(symbol_resolved_path),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        },
                    }
                    result["symbol_resolution_status"] = "failed"
            elif self.config.enable_cross_modal_symbol_resolution:
                chemeagle_result["symbol_resolution_status"] = "skipped"
                chemeagle_result["symbol_resolution"] = {
                    "status": "skipped",
                    "reason": "no_text_pair",
                    "text_output": str(text_symbol_resolved_path),
                    "image_output": str(symbol_resolved_path),
                }
                result["symbol_resolution_status"] = "skipped"

            if self.config.enable_chemeagle_iupac_enrichment:
                try:
                    enriched_payload, iupac_stats = enrich_chemeagle_raw_with_iupac(
                        chemeagle_iupac_input_payload,
                        cache_path=self.config.chemeagle_iupac_cache_path,
                    )
                    write_json(raw_iupac_path, enriched_payload)
                    chemeagle_result["iupac_status"] = "success"
                    chemeagle_result["iupac"] = {
                        **iupac_stats,
                        "input": str(chemeagle_iupac_input_path),
                        "output": str(raw_iupac_path),
                    }
                    result["chemeagle_iupac_status"] = "success"
                    result["counts"]["iupac_resolved_count"] = iupac_stats.get("resolved", 0)
                    result["counts"]["iupac_not_found_count"] = iupac_stats.get("not_found", 0)
                    result["counts"]["iupac_error_count"] = iupac_stats.get("error", 0)
                except Exception as exc:
                    chemeagle_result["iupac_status"] = "failed"
                    chemeagle_result["iupac"] = {
                        "status": "failed",
                        "output": str(raw_iupac_path),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        },
                    }
                    result["chemeagle_iupac_status"] = "failed"

            role_refinement_input = None
            role_refinement_input_path = chemeagle_iupac_input_path
            if self.config.enable_chemeagle_iupac_enrichment and raw_iupac_path.exists():
                try:
                    role_refinement_input = read_json(raw_iupac_path)
                    role_refinement_input_path = raw_iupac_path
                except Exception:
                    role_refinement_input = None
            if role_refinement_input is None:
                role_refinement_input = chemeagle_iupac_input_payload

            chemeagle_normalization_input = role_refinement_input
            chemeagle_normalization_input_path = role_refinement_input_path
            if self.config.enable_chemeagle_role_refinement:
                try:
                    if self.config.resume and not self.config.overwrite and role_refined_path.exists():
                        refined_payload = read_json(role_refined_path)
                        chemeagle_normalization_input = refined_payload
                        chemeagle_normalization_input_path = role_refined_path
                        role_stats = {
                            "status": "success",
                            "cache": "hit",
                            "model": self.config.chemeagle_role_refinement_model,
                            "conditions_total": 0,
                            "conditions_llm_refined": 0,
                            "conditions_fallback": 0,
                        }
                    else:
                        with token_usage_context("chemeagle_role_refinement", pdf_path.name):
                            refined_payload, role_stats = refine_chemeagle_condition_roles(
                                role_refinement_input,
                                model=self.config.chemeagle_role_refinement_model,
                                api_key=self.config.api_key,
                                base_url=self.config.base_url,
                            )
                        write_json(role_refined_path, refined_payload)
                        chemeagle_normalization_input = refined_payload
                        chemeagle_normalization_input_path = role_refined_path
                        role_stats["cache"] = "miss"
                    chemeagle_result["role_refinement_status"] = "success"
                    chemeagle_result["role_refinement"] = {
                        **role_stats,
                        "input": str(role_refinement_input_path),
                        "output": str(role_refined_path),
                    }
                    result["chemeagle_role_refinement_status"] = "success"
                    result["counts"]["role_refinement_conditions_total"] = role_stats.get("conditions_total", 0)
                    result["counts"]["role_refinement_conditions_llm_refined"] = role_stats.get("conditions_llm_refined", 0)
                    result["counts"]["role_refinement_conditions_fallback"] = role_stats.get("conditions_fallback", 0)
                except Exception as exc:
                    chemeagle_result["role_refinement_status"] = "failed"
                    chemeagle_result["role_refinement"] = {
                        "status": "failed",
                        "input": str(role_refinement_input_path),
                        "output": str(role_refined_path),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(limit=8),
                        },
                    }
                    result["chemeagle_role_refinement_status"] = "failed"

            if self.config.skip_chemeagle_normalization:
                chemeagle_result["status"] = "raw_only"
                chemeagle_result["normalization_status"] = "skipped"
                chemeagle_result["filter"] = {
                    "status": "skipped",
                    "reason": "skip_chemeagle_normalization",
                    "input": str(raw_path),
                    "output": None,
                }
                result["chemeagle_status"] = "raw_only"
                result["chemeagle_normalization_status"] = "skipped"
                result["chemeagle"] = chemeagle_result
                return

            normalized_payload = normalize_chemeagle_payload(
                chemeagle_normalization_input,
                pdf_path=pdf_path,
                artifact_stem=job["artifact_stem"],
                source_paper=job.get("paper_key"),
            )
            normalized_payload["metadata"] = {
                **metadata,
                "extractor": "ChemEagle",
                "chemeagle_pdf_model_size": self.config.chemeagle_pdf_model_size,
                "chemeagle_model_name": self.config.chemeagle_model_name,
                "role_refinement_input": str(chemeagle_normalization_input_path),
            }
            write_json(normalized_path, normalized_payload)

            filter_result = filter_reaction_file(
                normalized_path,
                filtered_path,
                overwrite=True,
            )
            chemeagle_result["filter"] = filter_result
            if filter_result.get("reactions") is None and filtered_path.exists():
                filtered_payload = read_json(filtered_path)
                filtered_count = len(filtered_payload.get("reactions", []))
            else:
                filtered_count = filter_result.get("reactions", 0)

            chemeagle_result["status"] = "success"
            chemeagle_result["normalization_status"] = "success"
            chemeagle_result["total_images"] = normalized_payload.get("total_images", 0)
            chemeagle_result["failed_images"] = normalized_payload.get("failed_images", [])
            chemeagle_result["normalized_reactions"] = normalized_payload.get("total_reactions", 0)
            chemeagle_result["filtered_reactions"] = filtered_count
            result["counts"]["chemeagle_images"] = chemeagle_result["total_images"]
            result["counts"]["chemeagle_reactions"] = chemeagle_result["normalized_reactions"]
            result["counts"]["chemeagle_filtered_reactions"] = filtered_count
            result["chemeagle_status"] = "success"
            result["chemeagle_normalization_status"] = "success"
        except Exception as exc:
            chemeagle_result["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            if chemeagle_result.get("raw_status") != "success":
                chemeagle_result["raw_status"] = "failed"
                result["chemeagle_raw_status"] = "failed"
            if chemeagle_result.get("normalization_status") == "pending":
                chemeagle_result["normalization_status"] = "failed"
                result["chemeagle_normalization_status"] = "failed"
            if chemeagle_result.get("iupac_status") == "pending":
                chemeagle_result["iupac_status"] = "failed"
                result["chemeagle_iupac_status"] = "failed"
            if chemeagle_result.get("symbol_resolution_status") == "pending":
                chemeagle_result["symbol_resolution_status"] = "failed"
                result["symbol_resolution_status"] = "failed"
            if chemeagle_result.get("role_refinement_status") == "pending":
                chemeagle_result["role_refinement_status"] = "failed"
                result["chemeagle_role_refinement_status"] = "failed"
            result["chemeagle_status"] = "failed"
        result["chemeagle"] = chemeagle_result


class CollectResultsAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        results = state.get("pdf_results", [])
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") != "success"]
        text_filtered_paths = [
            r["paths"]["text_filtered_output"]
            for r in results
            if r.get("text_status") == "success"
        ]
        text_symbol_resolved_paths = [
            r["paths"]["text_symbol_resolved_output"]
            for r in results
            if r.get("symbol_resolution_status") == "success"
        ]
        chemeagle_raw_paths = [
            r["paths"]["chemeagle_raw_output"]
            for r in results
            if r.get("chemeagle_raw_status") == "success"
        ]
        chemeagle_symbol_resolved_paths = [
            r["paths"]["chemeagle_symbol_resolved_output"]
            for r in results
            if r.get("symbol_resolution_status") == "success"
        ]
        chemeagle_raw_iupac_paths = [
            r["paths"]["chemeagle_raw_iupac_output"]
            for r in results
            if r.get("chemeagle_iupac_status") == "success"
        ]
        chemeagle_role_refined_paths = [
            r["paths"]["chemeagle_role_refined_output"]
            for r in results
            if r.get("chemeagle_role_refinement_status") == "success"
        ]
        chemeagle_filtered_paths = [
            r["paths"]["chemeagle_filtered_output"]
            for r in results
            if (
                not self.config.skip_chemeagle_normalization
                and r.get("chemeagle_normalization_status") == "success"
            )
        ]
        chemeagle_kg_input_paths = chemeagle_filtered_paths
        filtered_paths = text_filtered_paths + chemeagle_filtered_paths
        state["successful_filtered_paths"] = filtered_paths
        state["successful_text_filtered_paths"] = text_filtered_paths
        state["successful_text_symbol_resolved_paths"] = text_symbol_resolved_paths
        state["successful_chemeagle_raw_paths"] = chemeagle_raw_paths
        state["successful_chemeagle_symbol_resolved_paths"] = chemeagle_symbol_resolved_paths
        state["successful_chemeagle_raw_iupac_paths"] = chemeagle_raw_iupac_paths
        state["successful_chemeagle_role_refined_paths"] = chemeagle_role_refined_paths
        state["successful_chemeagle_kg_input_paths"] = chemeagle_kg_input_paths
        state["successful_chemeagle_filtered_paths"] = chemeagle_filtered_paths
        state["failed_pdfs"] = failed
        state["has_successful_pdfs"] = bool(filtered_paths)
        role_refinement_report = {
            "status": "success" if self.config.enable_chemeagle_role_refinement else "skipped",
            "created_at": datetime.now().isoformat(),
            "enabled": self.config.enable_chemeagle_role_refinement,
            "model": self.config.chemeagle_role_refinement_model,
            "output_dir": str(self.config.chemeagle_role_refined_dir),
            "successful_outputs": chemeagle_role_refined_paths,
            "failed": [
                {
                    "pdf_name": r.get("pdf_name"),
                    "error": r.get("chemeagle", {}).get("role_refinement", {}).get("error"),
                }
                for r in results
                if r.get("chemeagle_role_refinement_status") == "failed"
            ],
            "conditions_total": sum(
                r.get("counts", {}).get("role_refinement_conditions_total", 0)
                for r in results
            ),
            "conditions_llm_refined": sum(
                r.get("counts", {}).get("role_refinement_conditions_llm_refined", 0)
                for r in results
            ),
            "conditions_fallback": sum(
                r.get("counts", {}).get("role_refinement_conditions_fallback", 0)
                for r in results
            ),
        }
        if self.config.enable_chemeagle_role_refinement:
            write_json(self.config.chemeagle_role_refinement_report_path, role_refinement_report)
        symbol_candidates = []
        symbol_validations = []
        for r in results:
            candidates = r.get("chemeagle", {}).get("symbol_resolution", {}).get("candidates", [])
            if isinstance(candidates, list):
                for candidate in candidates:
                    if isinstance(candidate, dict):
                        symbol_candidates.append(
                            {
                                "pdf_name": r.get("pdf_name"),
                                "artifact_stem": r.get("artifact_stem"),
                                **candidate,
                            }
                        )
            validation = r.get("chemeagle", {}).get("symbol_resolution", {}).get("validation")
            if isinstance(validation, dict):
                symbol_validations.append(
                    {
                        "pdf_name": r.get("pdf_name"),
                        "artifact_stem": r.get("artifact_stem"),
                        **validation,
                    }
                )
        symbol_resolution_report = {
            "status": "success" if self.config.enable_cross_modal_symbol_resolution else "skipped",
            "created_at": datetime.now().isoformat(),
            "enabled": self.config.enable_cross_modal_symbol_resolution,
            "text_outputs": text_symbol_resolved_paths,
            "chemeagle_outputs": chemeagle_symbol_resolved_paths,
            "text_entities_resolved": sum(
                r.get("counts", {}).get("symbol_text_entities_resolved", 0)
                for r in results
            ),
            "image_entities_resolved": sum(
                r.get("counts", {}).get("symbol_image_entities_resolved", 0)
                for r in results
            ),
            "candidate_count": len(symbol_candidates),
            "invalid_resolution_count": sum(
                validation.get("invalid_resolution_count", 0)
                for validation in symbol_validations
            ),
            "validation_report": (
                str(self.config.symbol_resolution_validation_report_path)
                if self.config.enable_cross_modal_symbol_resolution
                else None
            ),
            "failed": [
                {
                    "pdf_name": r.get("pdf_name"),
                    "error": r.get("chemeagle", {}).get("symbol_resolution", {}).get("error"),
                }
                for r in results
                if r.get("symbol_resolution_status") == "failed"
            ],
        }
        if self.config.enable_cross_modal_symbol_resolution:
            write_json(self.config.symbol_resolution_report_path, symbol_resolution_report)
            write_json(self.config.symbol_resolution_candidates_path, symbol_candidates)
            write_json(
                self.config.symbol_resolution_validation_report_path,
                {
                    "status": (
                        "success"
                        if not any(
                            validation.get("invalid_resolution_count", 0) > 0
                            for validation in symbol_validations
                        )
                        else "failed"
                    ),
                    "created_at": datetime.now().isoformat(),
                    "enabled": True,
                    "total_reports": len(symbol_validations),
                    "resolved_count": sum(
                        validation.get("resolved_count", 0)
                        for validation in symbol_validations
                    ),
                    "text_resolved_count": sum(
                        validation.get("text_resolved_count", 0)
                        for validation in symbol_validations
                    ),
                    "image_resolved_count": sum(
                        validation.get("image_resolved_count", 0)
                        for validation in symbol_validations
                    ),
                    "candidate_count": len(symbol_candidates),
                    "invalid_resolution_count": sum(
                        validation.get("invalid_resolution_count", 0)
                        for validation in symbol_validations
                    ),
                    "reports": symbol_validations,
                },
            )
        state.setdefault("steps", {})["collect_results"] = {
            "successful": len(successful),
            "failed": len(failed),
            "text_successful": len(text_filtered_paths),
            "text_symbol_resolved_successful": len(text_symbol_resolved_paths),
            "chemeagle_raw_successful": len(chemeagle_raw_paths),
            "chemeagle_symbol_resolved_successful": len(chemeagle_symbol_resolved_paths),
            "chemeagle_raw_iupac_successful": len(chemeagle_raw_iupac_paths),
            "chemeagle_role_refined_successful": len(chemeagle_role_refined_paths),
            "chemeagle_successful": len(chemeagle_filtered_paths),
            "chemeagle_normalized_successful": len(chemeagle_filtered_paths),
            "chemeagle_role_refinement_report": (
                str(self.config.chemeagle_role_refinement_report_path)
                if self.config.enable_chemeagle_role_refinement
                else None
            ),
            "symbol_resolution_report": (
                str(self.config.symbol_resolution_report_path)
                if self.config.enable_cross_modal_symbol_resolution
                else None
            ),
            "symbol_text_entities_resolved": sum(
                r.get("counts", {}).get("symbol_text_entities_resolved", 0)
                for r in results
            ),
            "symbol_image_entities_resolved": sum(
                r.get("counts", {}).get("symbol_image_entities_resolved", 0)
                for r in results
            ),
            "symbol_resolution_candidates": sum(
                r.get("counts", {}).get("symbol_resolution_candidates", 0)
                for r in results
            ),
            "symbol_resolution_invalid": sum(
                r.get("counts", {}).get("symbol_resolution_invalid", 0)
                for r in results
            ),
            "iupac_resolved_count": sum(
                r.get("counts", {}).get("iupac_resolved_count", 0)
                for r in results
            ),
            "iupac_not_found_count": sum(
                r.get("counts", {}).get("iupac_not_found_count", 0)
                for r in results
            ),
            "iupac_error_count": sum(
                r.get("counts", {}).get("iupac_error_count", 0)
                for r in results
            ),
            "successful_pdfs": [r["pdf_name"] for r in successful],
            "failed_pdfs": failed,
        }
        return {
            "successful_filtered_paths": filtered_paths,
            "successful_text_filtered_paths": text_filtered_paths,
            "successful_text_symbol_resolved_paths": text_symbol_resolved_paths,
            "successful_chemeagle_raw_paths": chemeagle_raw_paths,
            "successful_chemeagle_symbol_resolved_paths": chemeagle_symbol_resolved_paths,
            "successful_chemeagle_raw_iupac_paths": chemeagle_raw_iupac_paths,
            "successful_chemeagle_role_refined_paths": chemeagle_role_refined_paths,
            "successful_chemeagle_kg_input_paths": chemeagle_kg_input_paths,
            "successful_chemeagle_filtered_paths": chemeagle_filtered_paths,
            "failed_pdfs": failed,
            "has_successful_pdfs": bool(filtered_paths),
            "steps": state["steps"],
        }


class ReactionReorganizationAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        merged_path = self.config.merged_dir / "all_filtered_reactions.json"
        input_paths = state.get("successful_filtered_paths", [])
        merge_result = merge_filtered_reactions_from_files(
            input_paths,
            merged_path,
        )
        state.setdefault("steps", {})["reaction_reorganization"] = {
            "merge": {
                "output": str(merged_path),
                "total_reactions": merge_result.get("total_reactions", 0),
                "total_files": merge_result.get("total_files", 0),
                "input_files": merge_result.get("input_files", []),
                "text_input_files": state.get("successful_text_filtered_paths", []),
                "chemeagle_input_files": state.get("successful_chemeagle_filtered_paths", []),
            },
        }
        state["merged_reactions_path"] = str(merged_path)
        return {
            "merged_reactions_path": str(merged_path),
            "steps": state["steps"],
        }


class CrossModalKGAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        if not self.config.enable_multimodal_kg:
            result = {
                "status": "skipped",
                "reason": "enable_multimodal_kg_false",
            }
        else:
            text_paths = state.get("successful_text_filtered_paths", [])
            chemeagle_paths = state.get("successful_chemeagle_kg_input_paths", [])
            if not text_paths and not chemeagle_paths:
                result = {
                    "status": "skipped",
                    "reason": "missing_filtered_inputs",
                    "text_inputs": text_paths,
                    "chemeagle_inputs": chemeagle_paths,
                    "input_policy": "multimodal KG uses filtered text and filtered ChemEagle reactions only",
                }
            else:
                if OpenAI is None:
                    raise ImportError("openai is required for multimodal structure enrichment")
                client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
                with token_usage_context("multimodal_structure_enrichment"):
                    structure_result = enrich_multimodal_structure(
                        text_reaction_paths=text_paths,
                        chemeagle_reaction_paths=chemeagle_paths,
                        output_dir=self.config.multimodal_structure_enriched_dir,
                        cache_path=self.config.multimodal_structure_parse_cache_path,
                        report_path=self.config.multimodal_structure_enrichment_report_path,
                        client=client,
                        model=self.config.split_model,
                        batch_size=30,
                    )
                kg_text_paths = structure_result.get("text_outputs") or text_paths
                kg_chemeagle_paths = structure_result.get("chemeagle_outputs") or chemeagle_paths
                result = build_cross_modal_kg(
                    text_reaction_paths=kg_text_paths,
                    chemeagle_raw_iupac_paths=kg_chemeagle_paths,
                    output_dir=self.config.filtered_dir,
                    kg_output_dir=self.config.kg_dir,
                    alignments_output_dir=self.config.alignments_dir,
                )
                result["structure_enrichment"] = structure_result

        state.setdefault("steps", {})["cross_modal_kg"] = result
        result["text_filtered_kg_inputs"] = state.get("successful_text_filtered_paths", [])
        result["chemeagle_filtered_kg_inputs"] = state.get("successful_chemeagle_filtered_paths", [])
        result["text_structure_enriched_kg_inputs"] = result.get("structure_enrichment", {}).get("text_outputs")
        result["chemeagle_structure_enriched_kg_inputs"] = result.get("structure_enrichment", {}).get("chemeagle_outputs")
        state["cross_modal_alignments_path"] = result.get("cross_modal_alignments")
        state["cross_modal_candidates_path"] = result.get("cross_modal_alignment_candidates")
        state["reaction_alignment_candidates_path"] = result.get("reaction_alignment_candidates")
        state["kg_unified_multimodal_path"] = result.get("kg_triples_unified_multimodal")
        state["kg_multimodal_path"] = result.get("kg_triples_multimodal") or state["kg_unified_multimodal_path"]
        state["text_structure_enriched_kg_inputs"] = result.get("text_structure_enriched_kg_inputs")
        state["chemeagle_structure_enriched_kg_inputs"] = result.get("chemeagle_structure_enriched_kg_inputs")
        return {
            "cross_modal_alignments_path": state["cross_modal_alignments_path"],
            "cross_modal_candidates_path": state["cross_modal_candidates_path"],
            "reaction_alignment_candidates_path": state["reaction_alignment_candidates_path"],
            "kg_unified_multimodal_path": state["kg_unified_multimodal_path"],
            "kg_multimodal_path": state["kg_multimodal_path"],
            "text_structure_enriched_kg_inputs": state["text_structure_enriched_kg_inputs"],
            "chemeagle_structure_enriched_kg_inputs": state["chemeagle_structure_enriched_kg_inputs"],
            "steps": state["steps"],
        }


class ReactionTypeNormalizationAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        merged_path = Path(state["merged_reactions_path"])
        report_path = self.config.reports_dir / "reaction_type_normalization_report.json"

        if self.config.skip_reaction_type_normalization:
            result = {
                "status": "skipped",
                "reason": "skip_reaction_type_normalization",
                "input": str(merged_path),
                "output": str(merged_path),
                "report_output": str(report_path),
            }
            if not report_path.exists() or self.config.overwrite:
                write_json(report_path, result)
        else:
            with token_usage_context("reaction_type_normalization"):
                result = normalize_reaction_types_file(
                    input_path=merged_path,
                    output_path=merged_path,
                    report_path=report_path,
                    model=self.config.extract_model,
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    batch_size=20,
                )

        state.setdefault("steps", {})["reaction_type_normalization"] = result
        state["reaction_type_report"] = str(report_path)
        return {
            "merged_reactions_path": str(merged_path),
            "reaction_type_report": str(report_path),
            "steps": state["steps"],
        }


class Q1Q2SplitAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        with token_usage_context("q1q2_scaffold_parse"):
            split_result = split_reactions(
                input_path=state["merged_reactions_path"],
                output_dir=self.config.benchmark_dir,
                cache_path=self.config.pipeline_cache_dir / "substrate_parse_cache.json",
                model=self.config.split_model,
                batch_size=30,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        if not split_result:
            raise RuntimeError("Q1/Q2 split failed; no split result was returned.")
        state.setdefault("steps", {})["q1q2_split"] = split_result
        state["q1_path"] = split_result["q1_output"]
        state["q2_path"] = split_result["q2_output"]
        return {
            "q1_path": state["q1_path"],
            "q2_path": state["q2_path"],
            "steps": state["steps"],
        }


class BenchmarkAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        q1_path = Path(state["q1_path"])
        q2_path = Path(state["q2_path"])
        q1_data = read_json(q1_path)
        q2_data = read_json(q2_path)

        q1_package = generate_q1_benchmark_package(q1_data)
        q1_benchmark = q1_package["benchmark"]
        q1_review = q1_package["review_set"]
        q1_report = q1_package["report"]
        q2_benchmark = generate_q2_benchmark(q2_data)

        q1_output = self.config.benchmark_dir / "Q1_benchmark.json"
        q1_review_output = self.config.benchmark_dir / "Q1_benchmark_review.json"
        q1_report_output = self.config.benchmark_dir / "Q1_benchmark_report.json"
        q2_output = self.config.benchmark_dir / "Q2_benchmark.json"
        write_json(q1_output, q1_benchmark)
        write_json(q1_review_output, q1_review)
        write_json(q1_report_output, q1_report)
        write_json(q2_output, q2_benchmark)

        state.setdefault("steps", {})["benchmark"] = {
            "q1": {
                "output": str(q1_output),
                "questions": len(q1_benchmark),
                "review_output": str(q1_review_output),
                "review_questions": len(q1_review),
                "report_output": str(q1_report_output),
            },
            "q2": {"output": str(q2_output), "questions": len(q2_benchmark)},
        }
        state["q1_benchmark"] = str(q1_output)
        state["q2_benchmark"] = str(q2_output)
        state["q1_benchmark_review"] = str(q1_review_output)
        state["q1_benchmark_report"] = str(q1_report_output)
        return {
            "q1_benchmark": state["q1_benchmark"],
            "q2_benchmark": state["q2_benchmark"],
            "q1_benchmark_review": state["q1_benchmark_review"],
            "q1_benchmark_report": state["q1_benchmark_report"],
            "steps": state["steps"],
        }


class SkipDownstreamAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        skipped = {
            "status": "skipped",
            "reason": "disabled_by_config",
            "config": "skip_downstream_build",
            "input": state.get("merged_reactions_path"),
        }
        state.setdefault("steps", {})["q1q2_split"] = dict(skipped)
        state["steps"]["benchmark"] = dict(skipped)
        for key in (
            "q1_path",
            "q2_path",
            "q1_benchmark",
            "q2_benchmark",
            "q1_benchmark_review",
            "q1_benchmark_report",
        ):
            state[key] = None
        return {
            "q1_path": None,
            "q2_path": None,
            "q1_benchmark": None,
            "q2_benchmark": None,
            "q1_benchmark_review": None,
            "q1_benchmark_report": None,
            "steps": state["steps"],
        }


class ReportAgent:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, state: Dict) -> Dict:
        report_path = self.config.reports_dir / "workflow_report.json"
        pdf_results = state.get("pdf_results", [])
        successful = [r for r in pdf_results if r.get("status") == "success"]
        failed = [r for r in pdf_results if r.get("status") != "success"]
        token_usage_summary = None
        if self.config.token_usage_tracking:
            token_usage_summary = summarize_token_usage(
                self.config.token_usage_events_path,
                self.config.token_usage_report_path,
            )
            state.setdefault("steps", {})["token_usage"] = {
                "status": "processed",
                "events": str(self.config.token_usage_events_path),
                "report": str(self.config.token_usage_report_path),
                "request_count": token_usage_summary.get("request_count", 0),
                "total_tokens": token_usage_summary.get("total_tokens", 0),
                "usage_missing_count": token_usage_summary.get("usage_missing_count", 0),
            }
        else:
            state.setdefault("steps", {})["token_usage"] = {
                "status": "skipped",
                "reason": "disable_token_usage_tracking",
            }
        chemeagle_timing_summary = summarize_chemeagle_timing(
            self.config.chemeagle_timing_events_path,
            self.config.chemeagle_timing_report_path,
        )
        state.setdefault("steps", {})["chemeagle_timing"] = {
            "status": "processed",
            "events": str(self.config.chemeagle_timing_events_path),
            "report": str(self.config.chemeagle_timing_report_path),
            "event_count": chemeagle_timing_summary.get("event_count", 0),
            "total_elapsed_seconds": chemeagle_timing_summary.get("total_elapsed_seconds", 0),
        }
        report = {
            "created_at": datetime.now().isoformat(),
            "pipeline_version": self.config.pipeline_version,
            "si_folder": str(self.config.si_folder),
            "pdf_folder": str(self.config.pdf_folder) if self.config.pdf_folder else None,
            "paper_map": str(self.config.paper_map) if self.config.paper_map else None,
            "output_dir": str(self.config.output_dir),
            "filtered_dir": str(self.config.filtered_dir),
            "filtered_subdirs": {
                "text": str(self.config.text_filtered_dir),
                "merged": str(self.config.merged_dir),
                "kg": str(self.config.kg_dir),
                "alignments": str(self.config.alignments_dir),
                "reports": str(self.config.reports_dir),
                "benchmark": str(self.config.benchmark_dir),
                "structure_enriched": str(self.config.multimodal_structure_enriched_dir),
            },
            "intermediate_dir": str(self.config.intermediate_dir),
            "intermediate_subdirs": {
                "cache": str(self.config.pipeline_cache_dir),
                "page_cache": str(self.config.page_cache_dir),
                "entity_context": str(self.config.entity_context_dir),
                "section_chunks": str(self.config.section_chunks_dir),
                "registry_debug": str(self.config.registry_debug_dir),
            },
            "config": {
                "pages_per_chunk": self.config.pages_per_chunk,
                "screen_model": self.config.screen_model,
                "extract_model": self.config.extract_model,
                "split_model": self.config.split_model,
                "chemeagle_role_refinement_model": self.config.chemeagle_role_refinement_model,
                "base_url": self.config.base_url,
                "max_parallel_pdfs": self.config.max_parallel_pdfs,
                "resume": self.config.resume,
                "overwrite": self.config.overwrite,
                "skip_reaction_type_normalization": self.config.skip_reaction_type_normalization,
                "skip_chemeagle_normalization": self.config.skip_chemeagle_normalization,
                "skip_downstream_build": self.config.skip_downstream_build,
                "enable_cross_modal_symbol_resolution": self.config.enable_cross_modal_symbol_resolution,
                "enable_chemeagle_iupac_enrichment": self.config.enable_chemeagle_iupac_enrichment,
                "enable_chemeagle_role_refinement": self.config.enable_chemeagle_role_refinement,
                "enable_multimodal_kg": self.config.enable_multimodal_kg,
                "enable_multimodal_structure_enrichment": self.config.enable_multimodal_structure_enrichment,
                "multimodal_structure_enrichment_effective": (
                    "required_by_multimodal_kg" if self.config.enable_multimodal_kg else "disabled"
                ),
                "enable_stage2_audit": self.config.enable_stage2_audit,
                "paper_name": self.config.paper_name,
                "limit": self.config.limit,
                "input_mode": self.config.input_mode,
                "pdf_folder": str(self.config.pdf_folder) if self.config.pdf_folder else None,
                "paper_map": str(self.config.paper_map) if self.config.paper_map else None,
                "enable_chemeagle": self.config.enable_chemeagle,
                "use_chemeagle": self.config.use_chemeagle,
                "chemeagle_dir": str(self.config.chemeagle_dir) if self.config.chemeagle_dir else None,
                "chemeagle_python": self.config.chemeagle_python,
                "chemeagle_pdf_model_size": self.config.chemeagle_pdf_model_size,
                "chemeagle_model_name": self.config.chemeagle_model_name,
                "chemeagle_base_url": self.config.chemeagle_base_url,
                "chemeagle_max_images": self.config.chemeagle_max_images,
                "chemeagle_use_plan_observer": self.config.chemeagle_use_plan_observer,
                "chemeagle_use_action_observer": self.config.chemeagle_use_action_observer,
                "token_usage_tracking": self.config.token_usage_tracking,
                "token_usage_run_id": self.config.token_usage_run_id,
            },
            "pdf_count": len(state.get("pdf_files", [])),
            "successful_pdf_count": len(successful),
            "failed_pdf_count": len(failed),
            "pdf_results": pdf_results,
            "artifacts": {
                "merged_reactions": state.get("merged_reactions_path"),
                "combined_merged_output": state.get("merged_reactions_path"),
                "text_reaction_outputs": [
                    r["paths"]["text_reaction_output"]
                    for r in pdf_results
                    if r.get("text_status") == "success"
                ],
                "section_debug_outputs": [
                    r["paths"]["section_debug"]
                    for r in pdf_results
                    if r.get("text_status") == "success" and r.get("paths", {}).get("section_debug")
                ],
                "registry_debug_outputs": [
                    r["paths"]["registry_debug"]
                    for r in pdf_results
                    if r.get("text_status") == "success" and r.get("paths", {}).get("registry_debug")
                ],
                "text_filtered_outputs": state.get("successful_text_filtered_paths", []),
                "text_symbol_resolved_outputs": state.get("successful_text_symbol_resolved_paths", []),
                "chemeagle_raw_outputs": state.get("successful_chemeagle_raw_paths", []),
                "chemeagle_symbol_resolved_outputs": state.get("successful_chemeagle_symbol_resolved_paths", []),
                "symbol_resolution_report": (
                    str(self.config.symbol_resolution_report_path)
                    if self.config.enable_cross_modal_symbol_resolution
                    else None
                ),
                "symbol_resolution_candidates": (
                    str(self.config.symbol_resolution_candidates_path)
                    if self.config.enable_cross_modal_symbol_resolution
                    else None
                ),
                "symbol_resolution_validation_report": (
                    str(self.config.symbol_resolution_validation_report_path)
                    if self.config.enable_cross_modal_symbol_resolution
                    else None
                ),
                "chemeagle_raw_iupac_outputs": state.get("successful_chemeagle_raw_iupac_paths", []),
                "chemeagle_role_refined_outputs": state.get("successful_chemeagle_role_refined_paths", []),
                "chemeagle_role_refinement_report": (
                    str(self.config.chemeagle_role_refinement_report_path)
                    if self.config.enable_chemeagle_role_refinement
                    else None
                ),
                "chemeagle_filtered_outputs": state.get("successful_chemeagle_filtered_paths", []),
                "text_filtered_kg_inputs": state.get("steps", {}).get("cross_modal_kg", {}).get("text_filtered_kg_inputs"),
                "chemeagle_filtered_kg_inputs": state.get("steps", {}).get("cross_modal_kg", {}).get("chemeagle_filtered_kg_inputs"),
                "text_structure_enriched_kg_inputs": state.get("steps", {}).get("cross_modal_kg", {}).get("text_structure_enriched_kg_inputs"),
                "chemeagle_structure_enriched_kg_inputs": state.get("steps", {}).get("cross_modal_kg", {}).get("chemeagle_structure_enriched_kg_inputs"),
                "multimodal_structure_parse_cache": (
                    str(self.config.multimodal_structure_parse_cache_path)
                    if self.config.enable_multimodal_kg
                    else None
                ),
                "multimodal_structure_enrichment_report": (
                    str(self.config.multimodal_structure_enrichment_report_path)
                    if self.config.enable_multimodal_kg
                    else None
                ),
                "cross_modal_alignments": state.get("cross_modal_alignments_path"),
                "cross_modal_alignment_candidates": state.get("cross_modal_candidates_path"),
                "reaction_alignment_candidates": state.get("reaction_alignment_candidates_path"),
                "kg_triples_unified_multimodal": state.get("kg_unified_multimodal_path"),
                "kg_triples_multimodal": state.get("kg_multimodal_path"),
                "reaction_type_normalization_report": state.get("reaction_type_report"),
                "q1": state.get("q1_path"),
                "q2": state.get("q2_path"),
                "q1_benchmark": state.get("q1_benchmark"),
                "q1_benchmark_review": state.get("q1_benchmark_review"),
                "q1_benchmark_report": state.get("q1_benchmark_report"),
                "q2_benchmark": state.get("q2_benchmark"),
                "token_usage_events": (
                    str(self.config.token_usage_events_path)
                    if self.config.token_usage_tracking
                    else None
                ),
                "token_usage_report": (
                    str(self.config.token_usage_report_path)
                    if self.config.token_usage_tracking
                    else None
                ),
                "chemeagle_timing_events": str(self.config.chemeagle_timing_events_path),
                "chemeagle_timing_report": str(self.config.chemeagle_timing_report_path),
            },
            "token_usage": token_usage_summary,
            "chemeagle_timing": chemeagle_timing_summary,
            "steps": state.get("steps", {}),
        }
        write_json(report_path, report)
        state["workflow_report"] = str(report_path)
        return {"workflow_report": str(report_path)}
