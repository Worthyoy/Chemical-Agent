import argparse
import json
import operator
import os
import socket
import subprocess
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
from langgraph_workflow.gpu_monitor import GpuMonitor  # noqa: E402


class PipelineState(TypedDict, total=False):
    pdf_files: List[str]
    jobs: List[Dict]
    job: Dict
    pdf_results: Annotated[List[Dict], operator.add]
    successful_filtered_paths: List[str]
    successful_text_filtered_paths: List[str]
    successful_text_structure_enriched_paths: List[str]
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
    q1q2_split_report: Optional[str]
    q1_benchmark: Optional[Dict[str, str]]
    q2_benchmark: Optional[Dict[str, str]]
    q1_benchmark_review: Optional[str]
    q1_benchmark_report: Optional[str]
    q2_benchmark_review: Optional[str]
    q2_benchmark_report: Optional[str]
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
    parser.add_argument("--base_url", default="https://hk.xty.app/v1")
    parser.add_argument("--max_parallel_pdfs", type=int, default=2)
    parser.add_argument(
        "--pdf_text_layout",
        choices=("single", "two_column", "auto"),
        default="single",
        help="PDF text extraction layout: single keeps current whole-page extraction; two_column reads left/right columns; auto detects two-column pages.",
    )
    parser.add_argument(
        "--pdf_text_x_tolerance",
        type=float,
        default=3.0,
        help="pdfplumber horizontal character tolerance (default: 3.0).",
    )
    parser.add_argument(
        "--pdf_text_y_tolerance",
        type=float,
        default=5.0,
        help="pdfplumber vertical line tolerance (default: 5.0).",
    )
    parser.add_argument(
        "--max_parallel_text_chunks",
        type=int,
        default=1,
        help="Maximum chunk-level concurrency inside each text extraction PDF. 1 preserves serial behavior.",
    )
    parser.add_argument(
        "--generic_resolution_batch_size",
        type=int,
        default=10,
        help="Reaction count per combined generic-substrate classification/resolution LLM call (default: 10).",
    )
    parser.add_argument(
        "--enable_stage1_page_trimming",
        action="store_true",
        help="Enable legacy Stage1 relevant_pages trimming. Disabled by default to preserve cross-page target evidence.",
    )
    parser.add_argument("--skip_reaction_type_normalization", action="store_true")
    parser.add_argument(
        "--benchmark_reaction_type_policy",
        choices=("required", "ignored"),
        default="required",
        help=(
            "Use reaction type for benchmark filtering/grouping (required) or "
            "ignore it and skip reaction-type normalization (ignored)."
        ),
    )
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
        "--chemeagle_iupac_lookup_timeout",
        type=float,
        default=15.0,
        help="Maximum seconds for each PubChem IUPAC lookup. 0 disables the timeout.",
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
        "--text-substrate-name-policy",
        choices=(
            "resolved",
            "original_if_available",
            "registry_if_resolved_else_original",
        ),
        default="resolved",
        help=(
            "Choose resolved substrate names for the KG (default), prefer "
            "source-reported names/symbols, or use registry-resolved names "
            "while retaining original names for other resolution sources."
        ),
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
        "--chemeagle_execution_mode",
        choices=("subprocess", "task_gpu_worker"),
        default="subprocess",
        help="Run ChemEagle normally or route GPU-heavy tasks through one long-lived GPU worker.",
    )
    parser.add_argument(
        "--chemeagle_gpu_worker_granularity",
        choices=("coarse", "model_forward"),
        default="coarse",
        help="In task_gpu_worker mode, choose coarse whole-method tasks or finer model-forward worker tasks.",
    )
    parser.add_argument(
        "--chemeagle_max_images",
        type=int,
        default=0,
        help="For validation only: limit ChemEagle to the first N extracted images. 0 means all images.",
    )
    parser.add_argument(
        "--chemeagle_max_parallel_images",
        type=int,
        default=1,
        help="Maximum extracted images to process concurrently inside each ChemEagle PDF subprocess. 1 preserves serial behavior.",
    )
    parser.add_argument(
        "--disable_token_usage_tracking",
        action="store_true",
        help="Disable local OpenAI token usage tracking.",
    )
    parser.add_argument(
        "--enable_gpu_monitor",
        action="store_true",
        help="Record GPU utilization during the pipeline and write CSV/JSON/PNG reports.",
    )
    parser.add_argument(
        "--gpu_monitor_interval",
        type=float,
        default=1.0,
        help="GPU sampling interval in seconds when --enable_gpu_monitor is set.",
    )
    parser.add_argument(
        "--disable_gpu_monitor_plot",
        action="store_true",
        help="Only write GPU monitor CSV/JSON; skip PNG generation.",
    )
    parser.add_argument(
        "--enable_chemeagle_gpu_gate",
        action="store_true",
        help="Serialize ChemEagle GPU-heavy local model calls across parallel PDF subprocesses.",
    )
    parser.add_argument(
        "--chemeagle_gpu_gate_util_threshold",
        type=float,
        default=20.0,
        help="GPU gate waits until utilization is at or below this percent.",
    )
    parser.add_argument(
        "--chemeagle_gpu_gate_memory_free_mib",
        type=float,
        default=2000.0,
        help="GPU gate waits until at least this much GPU memory is free.",
    )
    parser.add_argument(
        "--chemeagle_gpu_gate_stable_samples",
        type=int,
        default=2,
        help="Number of consecutive available GPU samples required before entering a gated operation.",
    )
    parser.add_argument(
        "--chemeagle_gpu_gate_poll_interval",
        type=float,
        default=1.0,
        help="Seconds between GPU gate availability checks.",
    )
    parser.add_argument(
        "--chemeagle_gpu_gate_timeout",
        type=float,
        default=0.0,
        help="Maximum seconds to wait for the GPU gate. 0 means wait indefinitely.",
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
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or env_values.get("OPENAI_API_KEY")
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
        base_url=os.getenv("LANGGRAPH_BASE_URL") or args.base_url,
        max_parallel_pdfs=max(1, args.max_parallel_pdfs),
        max_parallel_text_chunks=max(1, args.max_parallel_text_chunks),
        pdf_text_layout=args.pdf_text_layout,
        pdf_text_x_tolerance=args.pdf_text_x_tolerance,
        pdf_text_y_tolerance=args.pdf_text_y_tolerance,
        generic_resolution_batch_size=max(1, args.generic_resolution_batch_size),
        enable_stage1_page_trimming=args.enable_stage1_page_trimming,
        skip_reaction_type_normalization=args.skip_reaction_type_normalization,
        benchmark_reaction_type_policy=args.benchmark_reaction_type_policy,
        skip_chemeagle_normalization=args.skip_chemeagle_normalization,
        skip_downstream_build=args.skip_downstream_build,
        enable_cross_modal_symbol_resolution=args.enable_cross_modal_symbol_resolution,
        enable_chemeagle_iupac_enrichment=args.enable_chemeagle_iupac_enrichment,
        chemeagle_iupac_lookup_timeout=max(0.0, args.chemeagle_iupac_lookup_timeout),
        enable_chemeagle_role_refinement=args.enable_chemeagle_role_refinement,
        enable_multimodal_kg=args.enable_multimodal_kg,
        text_substrate_name_policy=args.text_substrate_name_policy,
        enable_multimodal_structure_enrichment=args.enable_multimodal_structure_enrichment,
        enable_stage2_audit=not args.skip_stage2_audit,
        input_mode=input_mode,
        enable_chemeagle=args.enable_chemeagle,
        use_chemeagle=use_chemeagle,
        chemeagle_dir=Path(os.getenv("CHEMEAGLE_DIR")) if os.getenv("CHEMEAGLE_DIR") else chemeagle_dir,
        chemeagle_python=os.getenv("CHEMEAGLE_PYTHON") or args.chemeagle_python,
        chemeagle_pdf_model_size=args.chemeagle_pdf_model_size,
        chemeagle_model_name=os.getenv("CHEMEAGLE_MODEL_NAME") or args.chemeagle_model_name,
        chemeagle_base_url=os.getenv("CHEMEAGLE_BASE_URL") or args.chemeagle_base_url,
        chemeagle_api_key=os.getenv("CHEMEAGLE_API_KEY") or args.chemeagle_api_key,
        chemeagle_max_images=max(0, args.chemeagle_max_images),
        chemeagle_max_parallel_images=max(1, args.chemeagle_max_parallel_images),
        chemeagle_use_plan_observer=False,
        chemeagle_use_action_observer=False,
        chemeagle_execution_mode=args.chemeagle_execution_mode,
        chemeagle_gpu_worker_granularity=args.chemeagle_gpu_worker_granularity,
        token_usage_tracking=not args.disable_token_usage_tracking,
        token_usage_run_id=token_usage_run_id,
        enable_gpu_monitor=args.enable_gpu_monitor,
        gpu_monitor_interval=max(0.1, args.gpu_monitor_interval),
        gpu_monitor_plot=not args.disable_gpu_monitor_plot,
        enable_chemeagle_gpu_gate=args.enable_chemeagle_gpu_gate,
        chemeagle_gpu_gate_util_threshold=max(0.0, args.chemeagle_gpu_gate_util_threshold),
        chemeagle_gpu_gate_memory_free_mib=max(0.0, args.chemeagle_gpu_gate_memory_free_mib),
        chemeagle_gpu_gate_stable_samples=max(1, args.chemeagle_gpu_gate_stable_samples),
        chemeagle_gpu_gate_poll_interval=max(0.1, args.chemeagle_gpu_gate_poll_interval),
        chemeagle_gpu_gate_timeout=max(0.0, args.chemeagle_gpu_gate_timeout),
    )


def choose_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_chemeagle_gpu_worker(config: PipelineConfig):
    if not config.chemeagle_dir:
        raise ValueError("ChemEagle directory is required for task_gpu_worker mode.")
    port = choose_free_port()
    config.chemeagle_gpu_worker_host = "127.0.0.1"
    config.chemeagle_gpu_worker_port = port
    config.chemeagle_gpu_worker_task_dir.mkdir(parents=True, exist_ok=True)
    config.chemeagle_gpu_worker_ready_path.parent.mkdir(parents=True, exist_ok=True)
    config.chemeagle_gpu_worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        config.chemeagle_gpu_worker_ready_path.unlink()
    except FileNotFoundError:
        pass

    command = [
        config.chemeagle_python,
        str(WORKFLOW_DIR / "chemeagle_gpu_worker.py"),
        "--chemeagle-dir",
        str(Path(config.chemeagle_dir).resolve()),
        "--host",
        config.chemeagle_gpu_worker_host,
        "--port",
        str(port),
        "--ready-file",
        str(config.chemeagle_gpu_worker_ready_path),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["CHEMEAGLE_TIMING_EVENTS_PATH"] = str(config.chemeagle_timing_events_path)
    env["CHEMEAGLE_TIMING_REPORT_PATH"] = str(config.chemeagle_timing_report_path)
    if config.token_usage_run_id:
        env["CHEMEAGLE_TIMING_RUN_ID"] = config.token_usage_run_id
    log_file = config.chemeagle_gpu_worker_log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        command,
        cwd=str(Path(config.chemeagle_dir).resolve()),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._chemeagle_log_file = log_file  # type: ignore[attr-defined]
    config.chemeagle_gpu_worker_pid = process.pid
    deadline = time.time() + 120
    while time.time() < deadline:
        if process.poll() is not None:
            log_file.flush()
            tail = ""
            try:
                tail = config.chemeagle_gpu_worker_log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except Exception:
                pass
            log_file.close()
            raise RuntimeError(f"ChemEagle GPU worker exited before ready. Log tail:\n{tail}")
        if config.chemeagle_gpu_worker_ready_path.exists():
            try:
                ready = json.loads(config.chemeagle_gpu_worker_ready_path.read_text(encoding="utf-8"))
                config.chemeagle_gpu_worker_port = int(ready.get("port") or port)
            except Exception:
                pass
            return process
        time.sleep(0.25)
    stop_chemeagle_gpu_worker(process)
    raise TimeoutError(f"Timed out waiting for ChemEagle GPU worker readiness: {config.chemeagle_gpu_worker_log_path}")


def stop_chemeagle_gpu_worker(process) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    log_file = getattr(process, "_chemeagle_log_file", None)
    if log_file is not None:
        try:
            log_file.close()
        except Exception:
            pass


def run_pipeline(config: PipelineConfig, *, progress_callback=None) -> Dict:
    """Run a configured workflow and return its final LangGraph state."""
    gpu_monitor = None
    gpu_summary = None
    chemeagle_gpu_worker = None
    if config.token_usage_tracking:
        install_token_usage_tracking(
            config.token_usage_events_path,
            run_id=config.token_usage_run_id,
            enabled=True,
            default_stage="workflow",
        )
    if config.enable_gpu_monitor:
        gpu_monitor = GpuMonitor(
            csv_path=config.gpu_usage_samples_path,
            report_path=config.gpu_usage_report_path,
            png_path=config.gpu_timeline_path if config.gpu_monitor_plot else None,
            interval_seconds=config.gpu_monitor_interval,
        )
        gpu_monitor.start()
    if config.use_chemeagle and config.chemeagle_execution_mode == "task_gpu_worker":
        chemeagle_gpu_worker = start_chemeagle_gpu_worker(config)
    final_state = None
    try:
        if progress_callback:
            progress_callback("starting", {})
        app = build_graph(config)
        final_state = app.invoke({"steps": {}, "pdf_results": []}, config={"max_concurrency": config.max_parallel_pdfs})
    finally:
        stop_chemeagle_gpu_worker(chemeagle_gpu_worker)
        if gpu_monitor is not None:
            gpu_summary = gpu_monitor.stop()
        if config.token_usage_tracking:
            summarize_token_usage(config.token_usage_events_path, config.token_usage_report_path)
    print("\nPipeline complete.")
    if final_state is not None:
        print(f"  Report: {final_state.get('workflow_report')}")
        print(f"  Multimodal KG CSV: {final_state.get('kg_unified_multimodal_path')}")
    if config.enable_gpu_monitor:
        print(f"  GPU usage report: {config.gpu_usage_report_path}")
        if config.gpu_monitor_plot:
            print(f"  GPU timeline: {config.gpu_timeline_path}")
        if gpu_summary is not None:
            print(f"  GPU samples: {gpu_summary.get('valid_sample_count', 0)} valid / {gpu_summary.get('sample_count', 0)} total")
    if progress_callback:
        progress_callback("complete", final_state or {})
    return final_state or {}


def main():
    config = config_from_args(parse_args())
    return run_pipeline(config)


if __name__ == "__main__":
    main()
