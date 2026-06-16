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
GPU_WORKER_GRANULARITY_ENV = "CHEMEAGLE_GPU_WORKER_GRANULARITY"

_PATCHED = False
_RXNIM_FORWARD_KWARGS = {"batch_size", "molnextr", "ocr"}


def worker_enabled() -> bool:
    return os.environ.get(GPU_WORKER_ENABLED_ENV) == "1"


def worker_granularity() -> str:
    value = (os.environ.get(GPU_WORKER_GRANULARITY_ENV) or "coarse").strip().lower()
    if value not in {"coarse", "model_forward"}:
        return "coarse"
    return value


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
            if worker_granularity() == "model_forward" and set(kwargs).issubset(_RXNIM_FORWARD_KWARGS):
                return predict_reaction_with_forward_worker(str(image_file), kwargs)
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
            if worker_granularity() == "model_forward" and set(kwargs).issubset(_RXNIM_FORWARD_KWARGS):
                return [predict_reaction_with_forward_worker(str(image_file), kwargs) for image_file in image_files]
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
        if worker_granularity() == "model_forward":
            return predict_corefs_with_forward_worker(
                image_files,
                batch_size=batch_size,
                molnextr=molnextr,
                ocr=ocr,
            )
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


class _WorkerMolNexTR:
    def predict_images(self, input_images: Iterable[Any], return_atoms_bonds=False, return_confidence=False, batch_size=16):
        images = list(input_images)
        image_files = [persist_image_for_worker(image) for image in images]
        raw_predictions = submit_gpu_task(
            "molnextr.forward_graph",
            {
                "image_files": image_files,
                "kwargs": {
                    "batch_size": batch_size,
                    "return_confidence": return_confidence,
                },
            },
        )
        return finalize_molnextr_predictions(
            raw_predictions or [],
            images,
            return_atoms_bonds=return_atoms_bonds,
            return_confidence=return_confidence,
        )


class _WorkerEasyOCR:
    def readtext(self, image: Any, *args, **kwargs):
        image_file = persist_image_for_worker(image)
        return submit_gpu_task(
            "easyocr.readtext",
            {"image_file": image_file, "args": list(args), "kwargs": kwargs},
        )


def predict_reaction_with_forward_worker(image_file: str, kwargs: Dict[str, Any]) -> Any:
    started_at = time.perf_counter()
    status = "success"
    error_type = None
    error = None
    try:
        raw_predictions = submit_gpu_task(
            "rxnim.forward_reaction",
            {"image_files": [str(image_file)], "kwargs": kwargs},
        )
        from rxnim.data import postprocess_reactions

        with Image.open(image_file) as image:
            image_copy = image.convert("RGB").copy()
        result = postprocess_reactions(
            (raw_predictions or [[]])[0],
            image=image_copy,
            molnextr=_WorkerMolNexTR() if kwargs.get("molnextr") else None,
            ocr=_WorkerEasyOCR() if kwargs.get("ocr") else None,
        )
        return result
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        error = str(exc)
        raise
    finally:
        record_event(
            operation="chemeagle.client.postprocess",
            elapsed_seconds=time.perf_counter() - started_at,
            status=status,
            event_kind="leaf",
            error_type=error_type,
            error=error,
            task_type="rxnim.forward_reaction",
        )


def predict_corefs_with_forward_worker(
    image_files: List[str],
    batch_size: int = 16,
    molnextr: bool = True,
    ocr: bool = True,
) -> Any:
    started_at = time.perf_counter()
    status = "success"
    error_type = None
    error = None
    try:
        raw_predictions = submit_gpu_task(
            "moldetect.forward_coref",
            {"image_files": image_files, "kwargs": {"batch_size": batch_size}},
        )
        from rxnim.data import postprocess_coref_results

        results = []
        molnextr_proxy = _WorkerMolNexTR() if molnextr else None
        ocr_proxy = _WorkerEasyOCR() if ocr else None
        for image_file, raw in zip(image_files, raw_predictions or []):
            with Image.open(image_file) as image:
                image_copy = image.convert("RGB").copy()
            results.append(
                postprocess_coref_results(
                    raw,
                    image=image_copy,
                    molnextr=molnextr_proxy,
                    ocr=ocr_proxy,
                    batch_size=batch_size,
                )
            )
        return results
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        error = str(exc)
        raise
    finally:
        record_event(
            operation="chemeagle.client.postprocess",
            elapsed_seconds=time.perf_counter() - started_at,
            status=status,
            event_kind="leaf",
            error_type=error_type,
            error=error,
            task_type="moldetect.forward_coref",
        )


def finalize_molnextr_predictions(
    raw_predictions: List[Dict[str, Any]],
    input_images: List[Any],
    return_atoms_bonds: bool = False,
    return_confidence: bool = False,
) -> List[Dict[str, Any]]:
    import numpy as _np
    from molnextr.chemistry import convert_graph_to_smiles
    from molnextr.interface import BOND_TYPES

    node_coords = [pred["chartok_coords"]["coords"] for pred in raw_predictions]
    node_symbols = [pred["chartok_coords"]["symbols"] for pred in raw_predictions]
    edges = [pred["edges"] for pred in raw_predictions]
    images = [image if isinstance(image, _np.ndarray) else _np.asarray(image) for image in input_images]
    smiles_list, molblock_list, _ = convert_graph_to_smiles(
        node_coords,
        node_symbols,
        edges,
        images=images,
        num_workers=1,
    )
    outputs: List[Dict[str, Any]] = []
    for smiles, molblock, pred in zip(smiles_list, molblock_list, raw_predictions):
        pred_dict: Dict[str, Any] = {
            "smiles": smiles,
            "symbols": pred["chartok_coords"]["symbols"],
            "coords": pred["chartok_coords"]["coords"],
            "edges": pred["edges"],
            "molfile": molblock,
        }
        if return_confidence:
            pred_dict["confidence"] = pred.get("overall_score")
        if return_atoms_bonds:
            coords = pred["chartok_coords"]["coords"]
            symbols = pred["chartok_coords"]["symbols"]
            atom_scores = pred["chartok_coords"].get("atom_scores") or []
            atom_list = []
            for index, (symbol, coord) in enumerate(zip(symbols, coords)):
                atom = {"atom_symbol": symbol, "x": round(coord[0], 3), "y": round(coord[1], 3)}
                if return_confidence and index < len(atom_scores):
                    atom["confidence"] = atom_scores[index]
                atom_list.append(atom)
            bond_list = []
            edge_scores = pred.get("edge_scores") or []
            for i in range(len(symbols) - 1):
                for j in range(i + 1, len(symbols)):
                    bond_type_int = normalize_bond_type(get_matrix_value(edges, i, j, 0))
                    if 0 < bond_type_int < len(BOND_TYPES):
                        bond = {"bond_type": BOND_TYPES[bond_type_int], "endpoint_atoms": (i, j)}
                        if return_confidence:
                            bond["confidence"] = get_matrix_value(edge_scores, i, j)
                        bond_list.append(bond)
            pred_dict["atoms"] = atom_list
            pred_dict["bonds"] = bond_list
        outputs.append(pred_dict)
    return outputs


def normalize_bond_type(value: Any) -> int:
    if isinstance(value, list):
        if not value:
            return 0
        if len(value) == 1:
            return normalize_bond_type(value[0])
        return int(max(range(len(value)), key=lambda index: value[index]))
    try:
        return int(value)
    except Exception:
        return 0


def get_matrix_value(matrix: Any, i: int, j: int, default: Any = None) -> Any:
    try:
        row = matrix[i]
        return row[j]
    except Exception:
        return default


def submit_gpu_task(task_type: str, payload: Dict[str, Any]) -> Any:
    host = os.environ.get(GPU_WORKER_HOST_ENV, "127.0.0.1")
    port = int(os.environ.get(GPU_WORKER_PORT_ENV, "0"))
    timeout = float(os.environ.get(GPU_WORKER_TIMEOUT_ENV, "0") or 0)
    if port <= 0:
        raise RuntimeError("CHEMEAGLE_GPU_WORKER_PORT is not configured.")
    task = {"task_type": task_type, **payload}
    is_forward_task = task_type in {
        "rxnim.forward_reaction",
        "moldetect.forward_coref",
        "molnextr.forward_graph",
        "easyocr.readtext",
    }
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
            operation="chemeagle.gpu_worker.forward_wait" if is_forward_task else "chemeagle.gpu_worker.wait",
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
