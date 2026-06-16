import importlib.util
import time
from pathlib import Path


def load_bridge_module():
    bridge_path = (
        Path(__file__).resolve().parents[1]
        / "chemeagle_bridge.py"
    )
    spec = importlib.util.spec_from_file_location("chemeagle_bridge_for_test", bridge_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parallel_image_results_keep_input_order(tmp_path, monkeypatch):
    bridge = load_bridge_module()
    image_paths = [
        tmp_path / "page_1_image_1.png",
        tmp_path / "page_2_image_1.png",
        tmp_path / "page_3_image_1.png",
    ]
    for path in image_paths:
        path.write_bytes(b"png")

    sleeps = {
        "page_1_image_1.png": 0.03,
        "page_2_image_1.png": 0.01,
        "page_3_image_1.png": 0.02,
    }

    def fake_process_image(*, index, image_path, pdf_name, use_plan_observer, use_action_observer):
        time.sleep(sleeps[image_path.name])
        return index, {
            "pdf_name": pdf_name,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "status": "ok",
            "reaction_count": index,
        }, sleeps[image_path.name]

    monkeypatch.setattr(bridge, "_process_image", fake_process_image)

    results, elapsed = bridge._process_images_parallel(
        image_paths=image_paths,
        pdf_name="paper.pdf",
        raw_result_path=tmp_path / "raw.json",
        max_parallel_images=3,
        use_plan_observer=False,
        use_action_observer=False,
    )

    assert [item["image_name"] for item in results] == [path.name for path in image_paths]
    assert [item["reaction_count"] for item in results] == [0, 1, 2]
    assert len(elapsed) == 3


def test_parallel_image_failure_is_isolated(tmp_path, monkeypatch):
    bridge = load_bridge_module()
    image_paths = [
        tmp_path / "page_1_image_1.png",
        tmp_path / "page_2_image_1.png",
    ]
    for path in image_paths:
        path.write_bytes(b"png")

    def fake_process_image(*, index, image_path, pdf_name, use_plan_observer, use_action_observer):
        if index == 1:
            raise IndexError("list index out of range")
        return index, {
            "pdf_name": pdf_name,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "status": "ok",
            "reaction_count": 1,
        }, 0.0

    monkeypatch.setattr(bridge, "_process_image", fake_process_image)

    results, _ = bridge._process_images_parallel(
        image_paths=image_paths,
        pdf_name="paper.pdf",
        raw_result_path=tmp_path / "raw.json",
        max_parallel_images=2,
        use_plan_observer=False,
        use_action_observer=False,
    )

    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "error"
    assert "list index out of range" in results[1]["error"]
