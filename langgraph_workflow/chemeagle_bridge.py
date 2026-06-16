"""Subprocess entry point for running ChemEagle on one PDF.

This file is intentionally small and dependency-light from the workflow side.
It is executed with the Python interpreter configured for ChemEagle so the main
LangGraph workflow can run in a separate environment.
"""

import argparse
import contextlib
import concurrent.futures
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

BRIDGE_DIR = Path(__file__).resolve().parent
if str(BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(BRIDGE_DIR))

try:
    from token_usage import install_token_usage_tracking
except ImportError:
    install_token_usage_tracking = None

try:
    from chemeagle_gpu_worker_client import install_chemeagle_gpu_worker_client, worker_enabled
except ImportError:
    install_chemeagle_gpu_worker_client = None

    def worker_enabled() -> bool:
        return False

try:
    from chemeagle_timing import (
        TIMING_EVENTS_ENV,
        TIMING_PDF_NAME_ENV,
        TIMING_REPORT_ENV,
        TIMING_RUN_ID_ENV,
        image_timing_context,
        install_chemeagle_timing,
        patch_known_local_models,
        summarize_chemeagle_timing,
        timed_event,
    )
except ImportError:
    TIMING_EVENTS_ENV = "CHEMEAGLE_TIMING_EVENTS_PATH"
    TIMING_REPORT_ENV = "CHEMEAGLE_TIMING_REPORT_PATH"
    TIMING_RUN_ID_ENV = "CHEMEAGLE_TIMING_RUN_ID"
    TIMING_PDF_NAME_ENV = "CHEMEAGLE_TIMING_PDF_NAME"
    image_timing_context = None
    install_chemeagle_timing = None
    patch_known_local_models = None
    summarize_chemeagle_timing = None

    @contextlib.contextmanager
    def timed_event(*args, **kwargs):
        yield


def configure_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Run ChemEagle for a single PDF.")
    parser.add_argument("--chemeagle-dir", required=True)
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--raw-result-path", required=True)
    parser.add_argument("--pdf-model-size", default="large", choices=("base", "large"))
    parser.add_argument("--model-name", default="/models/Qwen3-VL-32B-Instruct-AWQ")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--max-parallel-images",
        type=int,
        default=1,
        help="Maximum number of extracted images to process concurrently. 1 preserves serial behavior.",
    )
    parser.add_argument("--no-plan-observer", action="store_true")
    parser.add_argument("--no-action-observer", action="store_true")
    return parser.parse_args()


def _process_image(
    *,
    index: int,
    image_path: Path,
    pdf_name: str,
    use_plan_observer: bool,
    use_action_observer: bool,
) -> Tuple[int, Dict[str, Any], float]:
    started = time.perf_counter()
    print(f"Processing image with ChemEagle/gpt-5-mini: {image_path}")
    context = image_timing_context(image_path.name) if image_timing_context else contextlib.nullcontext()
    with context:
        try:
            with timed_event(
                "ChemEagle(image_total)",
                pdf_name=pdf_name,
                image_name=image_path.name,
            ):
                extracted = ChemEagle(
                    str(image_path),
                    use_plan_observer=use_plan_observer,
                    use_action_observer=use_action_observer,
                )
            record = as_json_record(
                extracted,
                pdf_name=pdf_name,
                image_name=image_path.name,
                image_path=image_path,
            )
        except Exception as exc:
            record = {
                "pdf_name": pdf_name,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "status": "error",
                "reaction_count": 0,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
    elapsed = time.perf_counter() - started
    record["image_elapsed_seconds"] = round(elapsed, 3)
    return index, record, elapsed


def _write_results(path: Path, results: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _process_images_serial(
    *,
    image_paths: List[Path],
    pdf_name: str,
    raw_result_path: Path,
    use_plan_observer: bool,
    use_action_observer: bool,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    results = []
    elapsed_values = []
    for index, image_path in enumerate(image_paths):
        _, record, elapsed = _process_image(
            index=index,
            image_path=image_path,
            pdf_name=pdf_name,
            use_plan_observer=use_plan_observer,
            use_action_observer=use_action_observer,
        )
        results.append(record)
        elapsed_values.append(elapsed)
        _write_results(raw_result_path, results)
    return results, elapsed_values


def _process_images_parallel(
    *,
    image_paths: List[Path],
    pdf_name: str,
    raw_result_path: Path,
    max_parallel_images: int,
    use_plan_observer: bool,
    use_action_observer: bool,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    results_by_index: Dict[int, Dict[str, Any]] = {}
    elapsed_by_index: Dict[int, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_images) as executor:
        future_to_item = {
            executor.submit(
                _process_image,
                index=index,
                image_path=image_path,
                pdf_name=pdf_name,
                use_plan_observer=use_plan_observer,
                use_action_observer=use_action_observer,
            ): (index, image_path)
            for index, image_path in enumerate(image_paths)
        }
        for future in concurrent.futures.as_completed(future_to_item):
            index, image_path = future_to_item[future]
            try:
                _, record, elapsed = future.result()
            except Exception as exc:
                record = {
                    "pdf_name": pdf_name,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "status": "error",
                    "reaction_count": 0,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                elapsed = 0.0
            results_by_index[index] = record
            elapsed_by_index[index] = elapsed
    results = [results_by_index[index] for index in range(len(image_paths))]
    elapsed_values = [elapsed_by_index.get(index, 0.0) for index in range(len(image_paths))]
    _write_results(raw_result_path, results)
    return results, elapsed_values


def main():
    configure_utf8_stdio()
    if install_token_usage_tracking is not None and os.environ.get("OPENAI_USAGE_LOG_PATH"):
        install_token_usage_tracking(
            os.environ["OPENAI_USAGE_LOG_PATH"],
            run_id=os.environ.get("OPENAI_USAGE_RUN_ID"),
            enabled=os.environ.get("OPENAI_USAGE_ENABLED", "1") != "0",
            default_stage=os.environ.get("OPENAI_USAGE_STAGE", "chemeagle_subprocess"),
        )
    if install_chemeagle_timing is not None and os.environ.get(TIMING_EVENTS_ENV):
        install_chemeagle_timing(
            os.environ[TIMING_EVENTS_ENV],
            report_path=os.environ.get(TIMING_REPORT_ENV),
            run_id=os.environ.get(TIMING_RUN_ID_ENV),
            pdf_name=os.environ.get(TIMING_PDF_NAME_ENV),
        )
    args = parse_args()
    chemeagle_dir = Path(args.chemeagle_dir).resolve()
    pdf_path = Path(args.pdf_path).resolve()
    image_dir = Path(args.image_dir).resolve()
    raw_result_path = Path(args.raw_result_path).resolve()

    if not chemeagle_dir.exists():
        raise FileNotFoundError(f"ChemEagle directory does not exist: {chemeagle_dir}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {pdf_path}")

    sys.path.insert(0, str(chemeagle_dir))
    os.chdir(chemeagle_dir)

    if install_chemeagle_gpu_worker_client is not None and worker_enabled():
        install_chemeagle_gpu_worker_client()

    global ChemEagle
    from pdf_extraction import run_pdf
    from main import ChemEagle
    from run_pdf_folder import _clear_dummy_proxy_env

    if patch_known_local_models is not None:
        patch_known_local_models()

    _clear_dummy_proxy_env()
    image_dir.mkdir(parents=True, exist_ok=True)
    raw_result_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with timed_event("pdf_extraction.run_pdf", pdf_name=pdf_path.name):
            run_pdf(
                pdf_dir=str(pdf_path),
                image_dir=str(image_dir),
                model_size=args.pdf_model_size,
            )

        image_paths = sorted(image_dir.glob("*.png"))
        if args.max_images and args.max_images > 0:
            image_paths = image_paths[: args.max_images]
        max_parallel_images = max(1, args.max_parallel_images)
        if max_parallel_images <= 1:
            results, elapsed_values = _process_images_serial(
                image_paths=image_paths,
                pdf_name=pdf_path.name,
                raw_result_path=raw_result_path,
                use_plan_observer=not args.no_plan_observer,
                use_action_observer=not args.no_action_observer,
            )
        else:
            results, elapsed_values = _process_images_parallel(
                image_paths=image_paths,
                pdf_name=pdf_path.name,
                raw_result_path=raw_result_path,
                max_parallel_images=max_parallel_images,
                use_plan_observer=not args.no_plan_observer,
                use_action_observer=not args.no_action_observer,
            )

        successful_images = sum(1 for item in results if item.get("status") != "error")
        failed_images = len(results) - successful_images
        summary = {
            "max_parallel_images": max_parallel_images,
            "total_images": len(results),
            "successful_images": successful_images,
            "failed_images": failed_images,
            "image_elapsed_seconds_max": round(max(elapsed_values), 3) if elapsed_values else 0.0,
            "image_elapsed_seconds_sum": round(sum(elapsed_values), 3),
        }
        print(f"ChemEagle image processing summary: {json.dumps(summary, ensure_ascii=False)}")
        _write_results(raw_result_path, results)
    finally:
        if summarize_chemeagle_timing is not None and os.environ.get(TIMING_EVENTS_ENV) and os.environ.get(TIMING_REPORT_ENV):
            summarize_chemeagle_timing(os.environ[TIMING_EVENTS_ENV], os.environ[TIMING_REPORT_ENV])

    print(f"Saved ChemEagle raw JSON: {raw_result_path}")


def as_json_record(
    value: Any,
    *,
    pdf_name: str,
    image_name: str,
    image_path: Path,
) -> Dict[str, Any]:
    if isinstance(value, dict):
        record = dict(value)
    else:
        record = {"result": value}

    record["pdf_name"] = pdf_name
    record["image_name"] = image_name
    record["image_path"] = str(image_path)
    record["status"] = "ok"
    record["reaction_count"] = len(record.get("reactions", [])) if isinstance(record.get("reactions"), list) else 0
    return record


if __name__ == "__main__":
    main()
