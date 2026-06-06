import argparse
import operator
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Annotated, Dict, List, Optional, TypedDict

WORKFLOW_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = WORKFLOW_DIR.parent
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

try:
    from dotenv import load_dotenv, dotenv_values
    DOTENV_PATH = EXTRACT_DIR / ".env"
    load_dotenv(DOTENV_PATH)
    load_dotenv()
except ImportError:
    dotenv_values = None
    DOTENV_PATH = EXTRACT_DIR / ".env"

from langgraph_workflow.pipeline_agents import (  # noqa: E402
    BenchmarkAgent,
    CollectResultsAgent,
    CrossModalKGAgent,
    PipelineConfig,
    PrepareJobsAgent,
    ProcessPDFAgent,
    Q1Q2SplitAgent,
    ReactionReorganizationAgent,
    ReactionTypeNormalizationAgent,
    ReportAgent,
    SkipDownstreamAgent,
)
from langgraph_workflow.token_usage import (  # noqa: E402
    install_token_usage_tracking,
    make_run_id,
    summarize_token_usage,
    token_usage_context,
)


class PipelineState(TypedDict, total=False):
    pdf_files: List[str]
    jobs: List[Dict]
    job: Dict
    pdf_results: Annotated[List[Dict], operator.add]
    successful_filtered_paths: List[str]
    successful_text_filtered_paths: List[str]
    successful_text_symbol_resolved_paths: List[str]
    successful_chemeagle_raw_paths: List[str]
    successful_chemeagle_symbol_resolved_paths: List[str]
    successful_chemeagle_raw_iupac_paths: List[str]
    successful_chemeagle_role_refined_paths: List[str]
    successful_chemeagle_kg_input_paths: List[str]
    successful_chemeagle_filtered_paths: List[str]
    failed_pdfs: List[Dict]
    merged_reactions_path: str
    reaction_type_report: str
    cross_modal_alignments_path: Optional[str]
    cross_modal_candidates_path: Optional[str]
    kg_unified_multimodal_path: Optional[str]
    kg_multimodal_path: Optional[str]
    text_structure_enriched_kg_inputs: List[str]
    chemeagle_structure_enriched_kg_inputs: List[str]
    q1_path: Optional[str]
    q2_path: Optional[str]
    q1_benchmark: Optional[str]
    q2_benchmark: Optional[str]
    q1_benchmark_review: Optional[str]
    q1_benchmark_report: Optional[str]
    workflow_report: str
    has_successful_pdfs: bool
    steps: Dict


def with_agent_log(agent_name: str, run_fn):
    def _wrapped(state: Dict) -> Dict:
        print(f"\n[AGENT START] {agent_name}", flush=True)
        started_at = time.perf_counter()
        job = state.get("job") if isinstance(state, dict) else None
        pdf_name = job.get("pdf_name") if isinstance(job, dict) else None
        try:
            with token_usage_context(agent_name, pdf_name):
                next_state = run_fn(state)
            elapsed = time.perf_counter() - started_at
            print(f"[AGENT END]   {agent_name} | {elapsed:.2f}s", flush=True)
            return next_state
        except Exception as exc:
            elapsed = time.perf_counter() - started_at
            print(
                f"[AGENT FAIL]  {agent_name} | {elapsed:.2f}s | {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            raise

    return _wrapped


def build_graph(config: PipelineConfig):
    try:
        from langgraph.graph import END, StateGraph
        from langgraph.types import Send
    except ImportError as exc:
        raise ImportError(
            "LangGraph is required for this pipeline. Install it with: pip install langgraph"
        ) from exc

    def fan_out_pdf_jobs(state: Dict):
        jobs = state.get("jobs", [])
        if not jobs:
            return "collect_results"
        return [Send("process_pdf", {"job": job}) for job in jobs]

    def after_collect(state: Dict):
        if config.enable_multimodal_kg:
            return "cross_modal_kg"
        if state.get("has_successful_pdfs"):
            return "reaction_reorganization"
        return "report"

    def after_cross_modal_kg(state: Dict):
        if state.get("has_successful_pdfs"):
            return "reaction_reorganization"
        return "report"

    def after_reaction_type_normalization(state: Dict):
        if config.skip_downstream_build:
            return "skip_downstream"
        return "q1q2_split"

    graph = StateGraph(PipelineState)
    graph.add_node("prepare_jobs", with_agent_log("prepare_jobs", PrepareJobsAgent(config).run))
    graph.add_node("process_pdf", with_agent_log("process_pdf", ProcessPDFAgent(config).run))
    graph.add_node("collect_results", with_agent_log("collect_results", CollectResultsAgent(config).run))
    graph.add_node("cross_modal_kg", with_agent_log("cross_modal_kg", CrossModalKGAgent(config).run))
    graph.add_node(
        "reaction_reorganization",
        with_agent_log("reaction_reorganization", ReactionReorganizationAgent(config).run),
    )
    graph.add_node(
        "reaction_type_normalization",
        with_agent_log("reaction_type_normalization", ReactionTypeNormalizationAgent(config).run),
    )
    graph.add_node("q1q2_split", with_agent_log("q1q2_split", Q1Q2SplitAgent(config).run))
    graph.add_node("benchmark", with_agent_log("benchmark", BenchmarkAgent(config).run))
    graph.add_node("skip_downstream", with_agent_log("skip_downstream", SkipDownstreamAgent(config).run))
    graph.add_node("report", with_agent_log("report", ReportAgent(config).run))

    graph.set_entry_point("prepare_jobs")
    graph.add_conditional_edges("prepare_jobs", fan_out_pdf_jobs, ["process_pdf", "collect_results"])
    graph.add_edge("process_pdf", "collect_results")
    graph.add_conditional_edges("collect_results", after_collect, ["cross_modal_kg", "reaction_reorganization", "report"])
    graph.add_conditional_edges("cross_modal_kg", after_cross_modal_kg, ["reaction_reorganization", "report"])
    graph.add_edge("reaction_reorganization", "reaction_type_normalization")
    graph.add_conditional_edges(
        "reaction_type_normalization",
        after_reaction_type_normalization,
        ["q1q2_split", "skip_downstream"],
    )
    graph.add_edge("q1q2_split", "benchmark")
    graph.add_edge("benchmark", "report")
    graph.add_edge("skip_downstream", "report")
    graph.add_edge("report", END)
    return graph.compile()


def parse_args():
    parser = argparse.ArgumentParser(
        description="LangGraph pipeline for SI reaction extraction."
    )
    parser.add_argument("--si_folder", default=None)
    parser.add_argument("--pdf_folder", default=None)
    parser.add_argument("--paper_map", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--filtered", default=None)
    parser.add_argument("--intermediate", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--pages_per_chunk", type=int, default=5)
    parser.add_argument("--screen_model", default="gpt-5-mini")
    parser.add_argument("--extract_model", default="gpt-5-mini")
    parser.add_argument("--split_model", default="gpt-5-mini")
    parser.add_argument("--chemeagle_role_refinement_model", default="gpt-5-mini")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--paper_name", default=None)
    parser.add_argument("--base_url", default="https://oneapi.xty.app/v1")
    parser.add_argument("--max_parallel_pdfs", type=int, default=2)
    parser.add_argument("--skip_reaction_type_normalization", action="store_true")
    parser.add_argument("--skip_stage2_audit", action="store_true")
    parser.add_argument(
        "--skip_chemeagle_normalization",
        action="store_true",
        help="Run ChemEagle raw extraction only; do not normalize/filter ChemEagle output.",
    )
    parser.add_argument(
        "--skip_downstream_build",
        action="store_true",
        help="Skip Q1/Q2 split and benchmark generation. Multimodal KG still runs when --enable_multimodal_kg is enabled.",
    )
    parser.add_argument(
        "--enable_chemeagle_iupac_enrichment",
        action="store_true",
        help="Add a raw_iupac ChemEagle JSON by resolving SMILES to PubChem IUPAC names.",
    )
    parser.add_argument(
        "--enable_cross_modal_symbol_resolution",
        action="store_true",
        help="Resolve symbol-only entities between text raw JSON and ChemEagle raw JSON before filtering/IUPAC enrichment.",
    )
    parser.add_argument(
        "--enable_chemeagle_role_refinement",
        action="store_true",
        help="Use an LLM to refine ChemEagle condition roles before multimodal KG construction.",
    )
    parser.add_argument(
        "--enable_multimodal_kg",
        action="store_true",
        help="Build unified text/ChemEagle reaction KG and cross-modal alignments.",
    )
    parser.add_argument(
        "--enable_multimodal_structure_enrichment",
        action="store_true",
        help="Compatibility flag. Structure enrichment runs automatically whenever --enable_multimodal_kg is enabled.",
    )
    parser.add_argument(
        "--input_mode",
        choices=("auto", "supporting_information", "pdf_folder", "dual_folder"),
        default="auto",
        help="auto keeps supporting_information text-only and other PDF folders text+ChemEagle; dual_folder uses SI for text and PDF folder for ChemEagle.",
    )
    parser.add_argument(
        "--enable_chemeagle",
        choices=("auto", "always", "never"),
        default="auto",
        help="Controls whether the ChemEagle image workflow runs.",
    )
    parser.add_argument("--chemeagle_dir", default=None)
    parser.add_argument("--chemeagle_python", default=sys.executable)
    parser.add_argument("--chemeagle_pdf_model_size", choices=("base", "large"), default="large")
    parser.add_argument("--chemeagle_model_name", default="/models/Qwen3-VL-32B-Instruct-AWQ")
    parser.add_argument("--chemeagle_base_url", default=None)
    parser.add_argument("--chemeagle_api_key", default=None)
    parser.add_argument(
        "--chemeagle_max_images",
        type=int,
        default=0,
        help="For validation only: limit ChemEagle to the first N extracted images. 0 means all images.",
    )
    parser.add_argument(
        "--disable_token_usage_tracking",
        action="store_true",
        help="Disable local OpenAI token usage tracking.",
    )
    return parser.parse_args()


def default_chemeagle_dir() -> Path:
    candidates = [
        EXTRACT_DIR.parents[2] / "ChemEagle",
        Path.cwd() / "ChemEagle",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_input_mode(si_folder: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if si_folder.name.casefold() == "supporting_information":
        return "supporting_information"
    return "pdf_folder"


def resolve_use_chemeagle(input_mode: str, enable_chemeagle: str) -> bool:
    if enable_chemeagle == "always":
        return True
    if enable_chemeagle == "never":
        return False
    return input_mode in {"pdf_folder", "dual_folder"}


def config_from_args(args) -> PipelineConfig:
    if dotenv_values is None:
        raise ValueError("python-dotenv is required. Install with: pip install python-dotenv")
    env_values = dotenv_values(DOTENV_PATH)
    api_key = args.api_key or env_values.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in .env or pass --api_key.")

    si_folder = Path(args.si_folder) if args.si_folder else EXTRACT_DIR.parent / "supporting_information"
    pdf_folder = Path(args.pdf_folder) if args.pdf_folder else None
    paper_map = Path(args.paper_map) if args.paper_map else None
    output_dir = Path(args.output) if args.output else WORKFLOW_DIR / "output" / "output"
    filtered_dir = Path(args.filtered) if args.filtered else WORKFLOW_DIR / "output" / "filtered"
    intermediate_dir = Path(args.intermediate) if args.intermediate else WORKFLOW_DIR / "output" / "intermediate"
    input_mode = resolve_input_mode(si_folder, args.input_mode)
    if not si_folder.exists():
        raise ValueError(f"--si_folder does not exist: {si_folder}")
    if input_mode == "dual_folder" and pdf_folder is None:
        raise ValueError("--pdf_folder is required when --input_mode dual_folder is used.")
    if pdf_folder is not None and not pdf_folder.exists():
        raise ValueError(f"--pdf_folder does not exist: {pdf_folder}")
    if paper_map is not None and not paper_map.exists():
        raise ValueError(f"--paper_map does not exist: {paper_map}")
    use_chemeagle = resolve_use_chemeagle(input_mode, args.enable_chemeagle)
    chemeagle_dir = Path(args.chemeagle_dir) if args.chemeagle_dir else default_chemeagle_dir()

    token_usage_run_id = make_run_id()
    return PipelineConfig(
        si_folder=si_folder,
        pdf_folder=pdf_folder,
        paper_map=paper_map,
        output_dir=output_dir,
        filtered_dir=filtered_dir,
        intermediate_dir=intermediate_dir,
        api_key=api_key,
        pages_per_chunk=args.pages_per_chunk,
        screen_model=args.screen_model,
        extract_model=args.extract_model,
        split_model=args.split_model,
        chemeagle_role_refinement_model=args.chemeagle_role_refinement_model,
        overwrite=args.overwrite,
        resume=not args.no_resume,
        limit=args.limit,
        paper_name=args.paper_name,
        base_url=args.base_url,
        max_parallel_pdfs=max(1, args.max_parallel_pdfs),
        skip_reaction_type_normalization=args.skip_reaction_type_normalization,
        skip_chemeagle_normalization=args.skip_chemeagle_normalization,
        skip_downstream_build=args.skip_downstream_build,
        enable_cross_modal_symbol_resolution=args.enable_cross_modal_symbol_resolution,
        enable_chemeagle_iupac_enrichment=args.enable_chemeagle_iupac_enrichment,
        enable_chemeagle_role_refinement=args.enable_chemeagle_role_refinement,
        enable_multimodal_kg=args.enable_multimodal_kg,
        enable_multimodal_structure_enrichment=args.enable_multimodal_structure_enrichment,
        enable_stage2_audit=not args.skip_stage2_audit,
        input_mode=input_mode,
        enable_chemeagle=args.enable_chemeagle,
        use_chemeagle=use_chemeagle,
        chemeagle_dir=chemeagle_dir,
        chemeagle_python=args.chemeagle_python,
        chemeagle_pdf_model_size=args.chemeagle_pdf_model_size,
        chemeagle_model_name=args.chemeagle_model_name,
        chemeagle_base_url=args.chemeagle_base_url,
        chemeagle_api_key=args.chemeagle_api_key,
        chemeagle_max_images=max(0, args.chemeagle_max_images),
        chemeagle_use_plan_observer=False,
        chemeagle_use_action_observer=False,
        token_usage_tracking=not args.disable_token_usage_tracking,
        token_usage_run_id=token_usage_run_id,
    )


def main():
    config = config_from_args(parse_args())
    if config.token_usage_tracking:
        install_token_usage_tracking(
            config.token_usage_events_path,
            run_id=config.token_usage_run_id,
            enabled=True,
            default_stage="workflow",
        )
    app = build_graph(config)
    final_state = None
    try:
        final_state = app.invoke({"steps": {}, "pdf_results": []}, config={"max_concurrency": config.max_parallel_pdfs})
    finally:
        if config.token_usage_tracking:
            summarize_token_usage(config.token_usage_events_path, config.token_usage_report_path)
    print("\nPipeline complete.")
    print(f"  Report: {final_state.get('workflow_report')}")
    print(f"  Multimodal KG CSV: {final_state.get('kg_unified_multimodal_path')}")


if __name__ == "__main__":
    main()
