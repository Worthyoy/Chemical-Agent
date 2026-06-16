"""Client-side monkeypatches for ChemEagle task-level GPU worker mode."""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image

try:
    import numpy as np
except Exception:  # pragma: no cover - optional in the workflow env
    np = None

try:
    from chemeagle_timing import record_event
except ImportError:  # pragma: no cover - ChemEagle can run without workflow timing

    def record_event(**kwargs):
        return None


GPU_WORKER_ENABLED_ENV = "CHEMEAGLE_GPU_WORKER_ENABLED"
GPU_WORKER_HOST_ENV = "CHEMEAGLE_GPU_WORKER_HOST"
GPU_WORKER_PORT_ENV = "CHEMEAGLE_GPU_WORKER_PORT"
GPU_WORKER_TASK_DIR_ENV = "CHEMEAGLE_GPU_WORKER_TASK_DIR"
GPU_WORKER_TIMEOUT_ENV = "CHEMEAGLE_GPU_WORKER_TIMEOUT"

_PATCHED = False


def worker_enabled() -> bool:
    return os.environ.get(GPU_WORKER_ENABLED_ENV) == "1"


def install_chemeagle_gpu_worker_client() -> None:
    """Redirect known GPU-heavy ChemEagle calls to the external GPU worker."""

    global _PATCHED
    if _PATCHED or not worker_enabled():
        return
    patch_rxnim()
    patch_chemietoolkit()
    patch_visualheist()
    _PATCHED = True


def patch_rxnim() -> None:
    try:
        from rxnim.interface import RxnIM
    except Exception:
        return
    original_init = getattr(RxnIM, "__init__", None)
    if original_init is not None and not getattr(original_init, "_chemeagle_gpu_worker_wrapped", False):

        def init_proxy(self, model_path=None, device=None, *args, **kwargs):
            self._chemeagle_gpu_worker_proxy = True
            self.model_path = str(model_path) if model_path is not None else None
            self.device = device

        init_proxy._chemeagle_gpu_worker_wrapped = True
        RxnIM.__init__ = init_proxy

    original_file = getattr(RxnIM, "predict_image_file", None)
    if original_file is not None and not getattr(original_file, "_chemeagle_gpu_worker_wrapped", False):

        def predict_image_file_proxy(self, image_file: str, *args, **kwargs):
            if args:
                raise TypeError("RxnIM.predict_image_file proxy only supports keyword options.")
            return submit_gpu_task(
                "rxnim.predict_image_file",
                {"image_file": str(image_file), "kwargs": kwargs},
            )

        predict_image_file_proxy._chemeagle_gpu_worker_wrapped = True
        RxnIM.predict_image_file = predict_image_file_proxy

    original_files = getattr(RxnIM, "predict_image_files", None)
    if original_files is not None and not getattr(original_files, "_chemeagle_gpu_worker_wrapped", False):

        def predict_image_files_proxy(self, image_files: Iterable[Any], *args, **kwargs):
            if args:
                raise TypeError("RxnIM.predict_image_files proxy only supports keyword options.")
            return [
                submit_gpu_task(
                    "rxnim.predict_image_file",
                    {"image_file": str(image_file), "kwargs": kwargs},
                )
                for image_file in image_files
            ]

        predict_image_files_proxy._chemeagle_gpu_worker_wrapped = True
        RxnIM.predict_image_files = predict_image_files_proxy


def patch_chemietoolkit() -> None:
    try:
        from chemietoolkit.interface import ChemIEToolkit
    except Exception:
        return
    original_init = getattr(ChemIEToolkit, "__init__", None)
    if original_init is not None and not getattr(original_init, "_chemeagle_gpu_worker_wrapped", False):

        def init_proxy(self, device=None, *args, **kwargs):
            self._chemeagle_gpu_worker_proxy = True
            self.device = device
            self._molnextr = None
            self._rxnim = None
            self._pdfparser = None
            self._moldet = None
            self._chemrxnextractor = None
            self._chemner = None
            self._coref = None

        init_proxy._chemeagle_gpu_worker_wrapped = True
        ChemIEToolkit.__init__ = init_proxy

    original = getattr(ChemIEToolkit, "extract_molecule_corefs_from_figures", None)
    if original is None or getattr(original, "_chemeagle_gpu_worker_wrapped", False):
        return

    def extract_corefs_proxy(self, figures: Iterable[Any], batch_size=16, molnextr=True, ocr=True):
        image_files = [persist_image_for_worker(figure) for figure in figures]
        return submit_gpu_task(
            "chemietoolkit.extract_molecule_corefs_from_figures",
            {
                "image_files": image_files,
                "kwargs": {
                    "batch_size": batch_size,
                    "molnextr": molnextr,
                    "ocr": ocr,
                },
            },
        )

    extract_corefs_proxy._chemeagle_gpu_worker_wrapped = True
    ChemIEToolkit.extract_molecule_corefs_from_figures = extract_corefs_proxy


def patch_visualheist() -> None:
    try:
        from pdfmodel import methods
    except Exception:
        return

    def get_model_pool_proxy(num_models: int = 2, large_model: bool = True):
        return _VisualHeistProxyPool(large_model=large_model)

    def tf_id_detection_proxy(image, model, processor):
        image_file = persist_image_for_worker(image)
        large_model = True
        if isinstance(model, dict) and "large_model" in model:
            large_model = bool(model["large_model"])
        return submit_gpu_task(
            "visualheist.page_detection",
            {"image_file": image_file, "large_model": large_model},
        )

    methods.get_model_pool = get_model_pool_proxy
    methods._tf_id_detection = tf_id_detection_proxy


class _VisualHeistProxyPool:
    def __init__(self, large_model: bool):
        self.large_model = large_model

    def get_model(self):
        return {"large_model": self.large_model}, None


def submit_gpu_task(task_type: str, payload: Dict[str, Any]) -> Any:
    host = os.environ.get(GPU_WORKER_HOST_ENV, "127.0.0.1")
    port = int(os.environ.get(GPU_WORKER_PORT_ENV, "0"))
    timeout = float(os.environ.get(GPU_WORKER_TIMEOUT_ENV, "0") or 0)
    if port <= 0:
        raise RuntimeError("CHEMEAGLE_GPU_WORKER_PORT is not configured.")
    task = {"task_type": task_type, **payload}
    record_event(
        operation="chemeagle.gpu_worker.submit",
        elapsed_seconds=0.0,
        event_kind="leaf",
        task_type=task_type,
    )
    started_at = time.perf_counter()
    status = "success"
    error_type = None
    error = None
    try:
        with socket.create_connection((host, port), timeout=timeout or None) as sock:
            if timeout > 0:
                sock.settimeout(timeout)
            request = json.dumps(task, ensure_ascii=False).encode("utf-8") + b"\n"
            sock.sendall(request)
            response = read_json_line(sock)
        if response.get("status") != "ok":
            raise RuntimeError(
                f"ChemEagle GPU worker task failed: {response.get('error_type')}: {response.get('error')}"
            )
        return response.get("result")
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        error = str(exc)
        raise
    finally:
        record_event(
            operation="chemeagle.gpu_worker.wait",
            elapsed_seconds=time.perf_counter() - started_at,
            status=status,
            event_kind="leaf",
            error_type=error_type,
            error=error,
            task_type=task_type,
        )


def read_json_line(sock: socket.socket) -> Dict[str, Any]:
    chunks: List[bytes] = []
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    if not chunks:
        raise RuntimeError("ChemEagle GPU worker closed the connection without a response.")
    line = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


def persist_image_for_worker(image_like: Any) -> str:
    if isinstance(image_like, (str, os.PathLike)):
        return str(Path(image_like).resolve())
    task_dir = Path(os.environ.get(GPU_WORKER_TASK_DIR_ENV) or ".chemeagle_gpu_worker_tasks").resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    image_path = task_dir / f"{uuid.uuid4().hex}.png"
    if isinstance(image_like, Image.Image):
        image_like.convert("RGB").save(image_path)
        return str(image_path)
    if np is not None and isinstance(image_like, np.ndarray):
        Image.fromarray(image_like).convert("RGB").save(image_path)
        return str(image_path)
    if hasattr(image_like, "save"):
        image_like.save(image_path)
        return str(image_path)
    raise TypeError(f"Unsupported image object for ChemEagle GPU worker: {type(image_like).__name__}")
