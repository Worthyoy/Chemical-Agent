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
        self.model_lock = threading.Lock()
        self.forward_lock = threading.Lock()
        self.rxnim_model = None
        self.chemie_model = None
        self.easyocr_model = None
        self.visualheist_models: Dict[str, Any] = {}

    def get_rxnim(self):
        with self.model_lock:
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
        with self.model_lock:
            if self.chemie_model is None:
                import torch
                from chemietoolkit import ChemIEToolkit

                self.chemie_model = ChemIEToolkit(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
        return self.chemie_model

    def get_easyocr(self):
        with self.model_lock:
            if self.rxnim_model is not None and getattr(self.rxnim_model, "ocr_model", None) is not None:
                return self.rxnim_model.ocr_model
            if self.chemie_model is not None:
                coref = getattr(self.chemie_model, "_coref", None)
                if coref is not None and getattr(coref, "ocr_model", None) is not None:
                    return coref.ocr_model
            if self.easyocr_model is None:
                import torch
                import easyocr

                self.easyocr_model = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        return self.easyocr_model

    def get_molnextr(self):
        with self.model_lock:
            if self.rxnim_model is not None and getattr(self.rxnim_model, "molnextr", None) is not None:
                return self.rxnim_model.molnextr
            if self.chemie_model is not None:
                coref = getattr(self.chemie_model, "_coref", None)
                if coref is not None and getattr(coref, "molnextr", None) is not None:
                    return coref.molnextr
                molnextr = getattr(self.chemie_model, "_molnextr", None)
                if molnextr is not None:
                    return molnextr
        chemie = self.get_chemie()
        with self.model_lock:
            return chemie.molnextr

    def get_visualheist(self, large_model: bool):
        key = "large" if large_model else "base"
        with self.model_lock:
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
        try:
            if task_type in {
                "rxnim.forward_reaction",
                "moldetect.forward_coref",
                "molnextr.forward_graph",
                "easyocr.readtext",
            }:
                if task_type == "rxnim.forward_reaction":
                    return self.run_rxnim_forward_reaction(payload)
                if task_type == "moldetect.forward_coref":
                    return self.run_moldetect_forward_coref(payload)
                if task_type == "molnextr.forward_graph":
                    return self.run_molnextr_forward_graph(payload)
                if task_type == "easyocr.readtext":
                    return self.run_easyocr_readtext(payload)
            with self.lock:
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

    def run_rxnim_forward_reaction(self, payload: Dict[str, Any]) -> Any:
        import torch

        model = self.get_rxnim()
        kwargs = payload.get("kwargs") or {}
        batch_size = int(kwargs.get("batch_size") or 16)
        input_images = []
        for path in payload.get("image_files") or []:
            with Image.open(path) as image:
                input_images.append(image.convert("RGB").copy())
        tokenizer = model.tokenizer["reaction"]
        predictions = []
        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx : idx + batch_size]
            images, refs = zip(*[model.transform(image) for image in batch_images])
            images = torch.stack(images, dim=0).to(model.device)
            forward_started = time.perf_counter()
            status = "success"
            error_type = None
            error = None
            try:
                with self.forward_lock:
                    with torch.no_grad():
                        pred_seqs, pred_scores = model.model(images, max_len=tokenizer.max_len)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                error = str(exc)
                raise
            finally:
                record_event(
                    operation="chemeagle.gpu_worker.forward_run",
                    elapsed_seconds=time.perf_counter() - forward_started,
                    status=status,
                    event_kind="leaf",
                    error_type=error_type,
                    error=error,
                    task_type="rxnim.forward_reaction",
                    worker_pid=os.getpid(),
                )
            for seqs, scores, ref in zip(pred_seqs, pred_scores, refs):
                predictions.append(tokenizer.sequence_to_data(seqs.tolist(), scores.tolist(), scale=ref["scale"]))
        return predictions

    def run_moldetect_forward_coref(self, payload: Dict[str, Any]) -> Any:
        import torch

        chemie = self.get_chemie()
        with self.model_lock:
            model = chemie.coref
        kwargs = payload.get("kwargs") or {}
        batch_size = int(kwargs.get("batch_size") or 16)
        input_images = []
        for path in payload.get("image_files") or []:
            with Image.open(path) as image:
                input_images.append(image.convert("RGB").copy())
        tokenizer = model.tokenizer["coref"]
        predictions = []
        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx : idx + batch_size]
            images, refs = zip(*[model.transform(image) for image in batch_images])
            images = torch.stack(images, dim=0).to(model.device)
            forward_started = time.perf_counter()
            status = "success"
            error_type = None
            error = None
            try:
                with self.forward_lock:
                    with torch.no_grad():
                        pred_seqs, pred_scores = model.model(images, max_len=tokenizer.max_len)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                error = str(exc)
                raise
            finally:
                record_event(
                    operation="chemeagle.gpu_worker.forward_run",
                    elapsed_seconds=time.perf_counter() - forward_started,
                    status=status,
                    event_kind="leaf",
                    error_type=error_type,
                    error=error,
                    task_type="moldetect.forward_coref",
                    worker_pid=os.getpid(),
                )
            for seqs, scores, ref in zip(pred_seqs, pred_scores, refs):
                predictions.append(tokenizer.sequence_to_data(seqs.tolist(), scores.tolist(), scale=ref["scale"]))
        return predictions

    def run_molnextr_forward_graph(self, payload: Dict[str, Any]) -> Any:
        import cv2
        import torch

        model = self.get_molnextr()
        kwargs = payload.get("kwargs") or {}
        batch_size = int(kwargs.get("batch_size") or 16)
        return_confidence = bool(kwargs.get("return_confidence"))
        input_images = []
        for path in payload.get("image_files") or []:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Failed to read image for MolNexTR worker task: {path}")
            input_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        predictions = []
        for idx in range(0, len(input_images), batch_size):
            batch_images = input_images[idx : idx + batch_size]
            images = [model.transform(image=image, keypoints=[])["image"] for image in batch_images]
            images = torch.stack(images, dim=0).to(model.device)
            forward_started = time.perf_counter()
            status = "success"
            error_type = None
            error = None
            try:
                with self.forward_lock:
                    model.decoder.compute_confidence = return_confidence
                    with torch.no_grad():
                        features, hiddens = model.encoder(images)
                        batch_predictions = model.decoder.decode(features, hiddens)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                error = str(exc)
                raise
            finally:
                record_event(
                    operation="chemeagle.gpu_worker.forward_run",
                    elapsed_seconds=time.perf_counter() - forward_started,
                    status=status,
                    event_kind="leaf",
                    error_type=error_type,
                    error=error,
                    task_type="molnextr.forward_graph",
                    worker_pid=os.getpid(),
                )
            predictions += batch_predictions
        return predictions

    def run_easyocr_readtext(self, payload: Dict[str, Any]) -> Any:
        reader = self.get_easyocr()
        with Image.open(payload["image_file"]) as image:
            image_array = np.asarray(image.convert("RGB")) if np is not None else image.convert("RGB")
        forward_started = time.perf_counter()
        status = "success"
        error_type = None
        error = None
        try:
            with self.forward_lock:
                return reader.readtext(
                    image_array,
                    *(payload.get("args") or []),
                    **(payload.get("kwargs") or {}),
                )
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            error = str(exc)
            raise
        finally:
            record_event(
                operation="chemeagle.gpu_worker.forward_run",
                elapsed_seconds=time.perf_counter() - forward_started,
                status=status,
                event_kind="leaf",
                error_type=error_type,
                error=error,
                task_type="easyocr.readtext",
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
