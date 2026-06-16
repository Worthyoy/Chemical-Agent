"""Single-GPU task worker for ChemEagle local model calls."""

from __future__ import annotations

import argparse
import json
import os
import socketserver
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from PIL import Image

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from chemeagle_timing import record_event
except ImportError:  # pragma: no cover - worker can run without timing

    def record_event(**kwargs):
        return None


class WorkerState:
    def __init__(self, chemeagle_dir: Path):
        self.chemeagle_dir = chemeagle_dir
        self.lock = threading.Lock()
        self.rxnim_model = None
        self.chemie_model = None
        self.visualheist_models: Dict[str, Any] = {}

    def get_rxnim(self):
        if self.rxnim_model is None:
            import torch
            from rxnim import RxnIM

            ckpt_path = self.chemeagle_dir / "ChemEAGLEModel" / "rxn.ckpt"
            self.rxnim_model = RxnIM(
                str(ckpt_path),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
        return self.rxnim_model

    def get_chemie(self):
        if self.chemie_model is None:
            import torch
            from chemietoolkit import ChemIEToolkit

            self.chemie_model = ChemIEToolkit(
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        return self.chemie_model

    def get_visualheist(self, large_model: bool):
        key = "large" if large_model else "base"
        if key not in self.visualheist_models:
            from pdfmodel import methods

            self.visualheist_models[key] = methods.get_model_pool(
                num_models=1,
                large_model=large_model,
            ).get_model()
        return self.visualheist_models[key]

    def run_task(self, payload: Dict[str, Any]) -> Any:
        task_type = payload.get("task_type")
        started_at = time.perf_counter()
        status = "success"
        error_type = None
        error = None
        with self.lock:
            try:
                if task_type == "rxnim.predict_image_file":
                    model = self.get_rxnim()
                    return model.predict_image_file(
                        payload["image_file"],
                        **(payload.get("kwargs") or {}),
                    )
                if task_type == "chemietoolkit.extract_molecule_corefs_from_figures":
                    model = self.get_chemie()
                    images = []
                    for path in payload.get("image_files") or []:
                        with Image.open(path) as image:
                            images.append(image.convert("RGB").copy())
                    return model.extract_molecule_corefs_from_figures(
                        images,
                        **(payload.get("kwargs") or {}),
                    )
                if task_type == "visualheist.page_detection":
                    from pdfmodel import methods

                    model, processor = self.get_visualheist(bool(payload.get("large_model", True)))
                    with Image.open(payload["image_file"]) as image:
                        return methods._tf_id_detection(image.convert("RGB"), model, processor)
                raise ValueError(f"Unknown ChemEagle GPU worker task_type: {task_type}")
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                error = str(exc)
                raise
            finally:
                record_event(
                    operation="chemeagle.gpu_worker.run",
                    elapsed_seconds=time.perf_counter() - started_at,
                    status=status,
                    event_kind="leaf",
                    error_type=error_type,
                    error=error,
                    task_type=task_type,
                    worker_pid=os.getpid(),
                )


class JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline()
        if not line:
            return
        try:
            payload = json.loads(line.decode("utf-8"))
            result = self.server.state.run_task(payload)  # type: ignore[attr-defined]
            response = {"status": "ok", "result": make_json_safe(result)}
        except Exception as exc:
            response = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))


class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def parse_args():
    parser = argparse.ArgumentParser(description="ChemEagle single-GPU task worker.")
    parser.add_argument("--chemeagle-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chemeagle_dir = Path(args.chemeagle_dir).resolve()
    workflow_dir = Path(__file__).resolve().parent
    for path in (workflow_dir, chemeagle_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(chemeagle_dir)
    server = ThreadedServer((args.host, args.port), JsonLineHandler)
    server.state = WorkerState(chemeagle_dir)  # type: ignore[attr-defined]
    ready_file = Path(args.ready_file)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.write_text(
        json.dumps({"host": args.host, "port": args.port, "pid": os.getpid()}),
        encoding="utf-8",
    )
    print(f"ChemEagle GPU worker ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
