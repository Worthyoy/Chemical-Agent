"""Subprocess entry point for running ChemEagle on one PDF.

This file is intentionally small and dependency-light from the workflow side.
It is executed with the Python interpreter configured for ChemEagle so the main
LangGraph workflow can run in a separate environment.
"""

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

BRIDGE_DIR = Path(__file__).resolve().parent
if str(BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(BRIDGE_DIR))

try:
    from token_usage import install_token_usage_tracking
except ImportError:
    install_token_usage_tracking = None

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
    parser.add_argument("--no-plan-observer", action="store_true")
    parser.add_argument("--no-action-observer", action="store_true")
    return parser.parse_args()


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

        results = []
        image_paths = sorted(image_dir.glob("*.png"))
        if args.max_images and args.max_images > 0:
            image_paths = image_paths[: args.max_images]
        for image_path in image_paths:
            print(f"Processing image with ChemEagle/gpt-5-mini: {image_path}")
            context = image_timing_context(image_path.name) if image_timing_context else contextlib.nullcontext()
            with context:
                try:
                    with timed_event(
                        "ChemEagle(image_total)",
                        pdf_name=pdf_path.name,
                        image_name=image_path.name,
                    ):
                        extracted = ChemEagle(
                            str(image_path),
                            use_plan_observer=not args.no_plan_observer,
                            use_action_observer=not args.no_action_observer,
                        )
                    results.append(
                        as_json_record(
                            extracted,
                            pdf_name=pdf_path.name,
                            image_name=image_path.name,
                            image_path=image_path,
                        )
                    )
                except Exception as exc:
                    results.append(
                        {
                            "pdf_name": pdf_path.name,
                            "image_name": image_path.name,
                            "image_path": str(image_path),
                            "status": "error",
                            "reaction_count": 0,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
            with raw_result_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        with raw_result_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
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
