import json
import sys
from pathlib import Path


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from batch_si_extractor import SIExtractor  # noqa: E402
from reaction_filter import filter_reaction_file, file_sha256  # noqa: E402
from langgraph_workflow.pipeline_agents import (  # noqa: E402
    PipelineConfig,
    metadata_matches,
    source_metadata,
)


class TextRegion:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def extract_text(self, **kwargs):
        self.calls.append(kwargs)
        return self.text


class FakePage(TextRegion):
    width = 600
    height = 800

    def __init__(self, text="whole", left="left", right="right"):
        super().__init__(text)
        self.regions = [TextRegion(left), TextRegion(right)]
        self.crop_boxes = []

    def crop(self, box):
        self.crop_boxes.append(box)
        return self.regions[len(self.crop_boxes) - 1]


def make_extractor(layout="single", x_tolerance=3.0, y_tolerance=5.0):
    extractor = object.__new__(SIExtractor)
    extractor.pdf_text_layout = layout
    extractor.pdf_text_x_tolerance = x_tolerance
    extractor.pdf_text_y_tolerance = y_tolerance
    return extractor


def test_single_page_extraction_uses_configured_tolerances():
    extractor = make_extractor(x_tolerance=2.5, y_tolerance=4.5)
    page = FakePage(text="Na2CO3")

    assert extractor._extract_pdfplumber_page_text(page) == "Na2CO3"
    assert page.calls == [{"x_tolerance": 2.5, "y_tolerance": 4.5}]


def test_two_column_extraction_uses_same_tolerances_for_both_crops():
    extractor = make_extractor(layout="two_column")
    page = FakePage(left="General Procedure", right="Na2CO3")

    assert extractor._extract_two_column_text(page) == "General Procedure\n\nNa2CO3"
    expected = {"x_tolerance": 3.0, "y_tolerance": 5.0}
    assert page.regions[0].calls == [expected]
    assert page.regions[1].calls == [expected]


def test_pipeline_cli_accepts_tolerance_overrides(monkeypatch):
    from langgraph_workflow.pipeline import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pipeline.py",
            "--pdf_text_x_tolerance",
            "2.5",
            "--pdf_text_y_tolerance",
            "4.5",
        ],
    )

    args = parse_args()
    assert args.pdf_text_x_tolerance == 2.5
    assert args.pdf_text_y_tolerance == 4.5


def test_tolerance_metadata_invalidates_old_or_changed_cache(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"pdf")
    config = PipelineConfig(
        si_folder=tmp_path,
        output_dir=tmp_path / "output",
        filtered_dir=tmp_path / "filtered",
        intermediate_dir=tmp_path / "intermediate",
        api_key="test",
    )
    expected = source_metadata(pdf_path, config)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"metadata": expected}), encoding="utf-8")

    assert metadata_matches(cache_path, expected)

    changed = dict(expected, pdf_text_y_tolerance=4.0)
    assert not metadata_matches(cache_path, changed)

    old_metadata = dict(expected)
    old_metadata.pop("pdf_text_y_tolerance")
    cache_path.write_text(json.dumps({"metadata": old_metadata}), encoding="utf-8")
    assert not metadata_matches(cache_path, expected)


def test_filtered_output_is_rebuilt_only_when_input_hash_changes(tmp_path, monkeypatch):
    input_path = tmp_path / "reactions.json"
    output_path = tmp_path / "filtered.json"
    input_path.write_text('{"reactions": []}', encoding="utf-8")
    calls = []

    def fake_filter_file(self, input_file, output_file):
        calls.append(input_file)
        Path(output_file).write_text(
            json.dumps({"reactions": [], "filter_stats": {}}),
            encoding="utf-8",
        )
        self.last_stats = {}
        return 0

    monkeypatch.setattr("reaction_filter.ReactionFilter.filter_file", fake_filter_file)

    first = filter_reaction_file(input_path, output_path)
    second = filter_reaction_file(input_path, output_path)
    input_path.write_text('{"reactions": []}\n', encoding="utf-8")
    third = filter_reaction_file(input_path, output_path)

    assert [first["status"], second["status"], third["status"]] == [
        "processed",
        "skipped",
        "processed",
    ]
    assert len(calls) == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["filter_provenance"]["input_sha256"] == file_sha256(input_path)
