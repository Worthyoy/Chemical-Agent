"""Tests for reaction source page provenance."""

import sys
from pathlib import Path


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

import cross_modal_kg  # noqa: E402
from batch_si_extractor import SIExtractor  # noqa: E402
from cross_modal_kg import (  # noqa: E402
    EDGE_FIELDNAMES,
    alignment_to_triple,
    build_image_reaction_triples,
    build_text_reaction_triples,
)
from pdf_to_gpt_extractor import PDFReactionExtractor  # noqa: E402


def make_extractor():
    return SIExtractor.__new__(SIExtractor)


def simple_reaction(**overrides):
    reaction = {
        "id": "GeneralProcedureA-Entry1",
        "reaction_type": "test reaction",
        "substrates": [{"name": "substrate A"}],
        "products": [{"name": "product B"}],
        "catalysts": [],
        "additives": [],
        "reagents": [],
        "conditions": {"solvent": "THF"},
        "targets": {"yield": "80%", "ee": None, "er": None},
    }
    reaction.update(overrides)
    return reaction


def test_source_page_prompt_contracts_are_present():
    assert "Page provenance" in PDFReactionExtractor.EXTRACTION_PROMPT
    assert '"source_pages": [1]' in PDFReactionExtractor.EXTRACTION_PROMPT
    assert "source_pages from the concrete entry" in SIExtractor.STAGE2_AUDIT_PROMPT


def test_source_pages_are_normalized_and_filtered_to_chunk_pages():
    extractor = make_extractor()
    reaction = simple_reaction(source_pages=[5, "Page 3", "2, 2", 0, -1, "Page 99"])

    normalized = extractor.sanitize_reaction_schema(
        reaction,
        allowed_page_nums=[2, 3, 5],
    )

    assert normalized["source_pages"] == [2, 3, 5]


def test_missing_invalid_or_out_of_chunk_pages_become_empty_lists():
    extractor = make_extractor()

    missing = extractor.sanitize_reaction_schema(simple_reaction(), allowed_page_nums=[10])
    invalid = extractor.sanitize_reaction_schema(
        simple_reaction(source_pages=["not a page", 0, "Page 99"]),
        allowed_page_nums=[10],
    )

    assert missing["source_pages"] == []
    assert invalid["source_pages"] == []


def test_chunk_page_markers_are_parsed_from_stage2_text():
    extractor = make_extractor()
    chunk_text = "--- Page 146 ---\nentry text\n--- Page 147 ---\ncontinued"

    assert extractor._source_page_numbers_from_chunk_text(chunk_text) == [146, 147]


def test_duplicate_reaction_merge_unions_source_pages_without_changing_signature():
    extractor = make_extractor()
    first = simple_reaction(source_pages=[10])
    second = simple_reaction(source_pages=[11])

    merged = extractor.merge_results([[first], [second]])

    assert len(merged) == 1
    assert merged[0]["source_pages"] == [10, 11]


def text_payload_with_source_pages():
    return {
        "source": "paper.pdf",
        "reactions": [
            {
                "id": "GeneralProcedureB-Entry1",
                "source_pages": [13, 12, 12],
                "reaction_type": "test reaction",
                "substrates": [
                    {
                        "name": "N-(2-bromophenyl)acetamide",
                        "scaffold": "acetamide",
                        "substituents": ["2-bromo"],
                    }
                ],
                "products": [{"name": "product B"}],
                "catalysts": [],
                "additives": [],
                "reagents": [],
                "intermediates": [
                    {
                        "name": "intermediate I",
                        "produced_in_step": 1,
                        "consumed_in_step": 2,
                    }
                ],
                "conditions": {"solvent": "THF"},
                "targets": {"yield": "62% yield"},
            }
        ],
    }


def test_text_kg_edges_carry_same_source_pages_for_direct_and_derived_edges(monkeypatch):
    payload = text_payload_with_source_pages()
    monkeypatch.setattr(
        cross_modal_kg,
        "iter_text_reaction_payloads",
        lambda _paths: iter([(Path("reaction.json"), payload)]),
    )

    triples = build_text_reaction_triples([Path("reaction.json")])
    text_rows = [row for row in triples if row["source_modality"] == "text"]

    assert text_rows
    assert {row["source_pages"] for row in text_rows} == {"12, 13"}
    assert {"USES_SUBSTRATE", "HAS_SCAFFOLD", "HAS_SUBSTITUENT", "PRODUCES_INTERMEDIATE", "USES_INTERMEDIATE"}.issubset(
        {row["relationship"] for row in text_rows}
    )


def test_image_and_alignment_kg_edges_leave_source_pages_empty(monkeypatch):
    image_payload = {
        "source": "paper.pdf",
        "reactions": [
            {
                "id": "image-entry-1",
                "source_pages": [12],
                "source_image": "scheme_1.png",
                "reaction_type": "test reaction",
                "substrates": [{"name": "substrate A"}],
                "products": [{"name": "product B"}],
                "conditions": {"solvent": "THF"},
                "targets": {"yield": "80%"},
            }
        ],
    }
    monkeypatch.setattr(
        cross_modal_kg,
        "iter_image_reaction_payloads",
        lambda _paths: iter([(Path("image_reaction.json"), image_payload)]),
    )

    image_triples = build_image_reaction_triples([Path("image_reaction.json")])
    assert image_triples
    assert {row["source_pages"] for row in image_triples} == {""}

    alignment = alignment_to_triple(
        {
            "relationship": "SAME_AS",
            "source_paper": "paper",
            "text_entity": {"name": "text compound", "role": "substrate"},
            "image_entity": {"name": "image compound", "role": "substrate"},
        }
    )
    assert alignment["source_pages"] == ""


def test_source_pages_column_order_in_kg_header():
    assert EDGE_FIELDNAMES[
        EDGE_FIELDNAMES.index("reaction_id") + 1
    ] == "source_pages"
