"""Tests for KG step edge attributes."""

import sys
from pathlib import Path


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

import cross_modal_kg  # noqa: E402
from cross_modal_kg import (  # noqa: E402
    EDGE_FIELDNAMES,
    alignment_to_triple,
    build_image_reaction_triples,
    build_text_reaction_triples,
)


def multistep_reaction():
    return {
        "id": "step-test-1",
        "source_pages": [24],
        "reaction_type": "two-step test reaction",
        "step_count": 2,
        "substrates": [
            {
                "name": "substrate A",
                "amount": "1.0 mmol",
                "step": 1,
                "scaffold": "benzene",
                "substituents": ["4-bromo"],
            },
            {"name": "shared substrate", "amount": "1.0 mmol", "step": 1},
            {"name": "shared substrate", "amount": "2.0 mmol", "step": 2},
        ],
        "products": [{"name": "product B", "amount": "25 mg", "step": 2}],
        "catalysts": [{"name": "catalyst C", "amount": "5 mol%", "step": "Step 1"}],
        "additives": [{"name": "additive D", "amount": "2 equiv", "step": 1}],
        "reagents": [{"name": "reagent E", "amount": "3 mmol", "step": "2"}],
        "intermediates": [
            {
                "name": "intermediate I",
                "produced_in_step": 1,
                "consumed_in_step": 2,
            }
        ],
        "conditions": {
            "solvent": [
                {"step": 1, "value": "Et3N"},
                {"step": 2, "value": "THF"},
            ],
            "temperature": [
                {"step": "Step 1", "value": "50 °C"},
                {"step": "2", "value": "0 °C to room temperature"},
            ],
            "time": [
                {"step": 1, "value": "24 h"},
                {"step": 2, "value": "0.5 h"},
            ],
            "atmosphere": [
                {"step": 1, "value": "argon"},
                {"step": 2, "value": "argon"},
            ],
        },
        "targets": {"yield": "66% yield over two steps"},
    }


def test_text_multistep_edges_carry_step_and_conditions_are_split(monkeypatch):
    payload = {"source": "paper.pdf", "reactions": [multistep_reaction()]}
    monkeypatch.setattr(
        cross_modal_kg,
        "iter_text_reaction_payloads",
        lambda _paths: iter([(Path("reaction.json"), payload)]),
    )

    triples = build_text_reaction_triples([Path("reaction.json")])
    by_relationship = {}
    for row in triples:
        by_relationship.setdefault(row["relationship"], []).append(row)

    assert next(row for row in by_relationship["USES_SUBSTRATE"] if row["y_name"] == "substrate A")["step"] == "1"
    assert sorted(
        row["step"]
        for row in by_relationship["USES_SUBSTRATE"]
        if row["y_name"] == "shared substrate"
    ) == ["1", "2"]
    assert by_relationship["PRODUCES"][0]["step"] == "2"
    assert by_relationship["USES_CATALYST"][0]["step"] == "1"
    assert by_relationship["USES_ADDITIVE"][0]["step"] == "1"
    assert by_relationship["USES_REAGENT"][0]["step"] == "2"

    assert by_relationship["PRODUCES_INTERMEDIATE"][0]["step"] == "1"
    assert by_relationship["USES_INTERMEDIATE"][0]["step"] == "2"

    condition_rows = by_relationship["HAS_CONDITION"]
    assert sorted(row["step"] for row in condition_rows) == ["1", "2"]
    assert {
        row["y_name"]
        for row in condition_rows
    } == {
        "solvent: Et3N; temp: 50 °C; time: 24 h; atm: argon",
        "solvent: THF; temp: 0 °C to room temperature; time: 0.5 h; atm: argon",
    }
    assert all("Step " not in row["y_name"] for row in condition_rows)

    assert by_relationship["HAS_SCAFFOLD"][0]["step"] == "1"
    assert by_relationship["HAS_SUBSTITUENT"][0]["step"] == "1"
    assert by_relationship["REPORTED_IN"][0]["step"] == ""
    assert by_relationship["OF_TYPE"][0]["step"] == ""


def test_single_step_text_condition_and_entities_keep_blank_step(monkeypatch):
    payload = {
        "source": "paper.pdf",
        "reactions": [
            {
                "id": "single-step-1",
                "reaction_type": "single-step test",
                "substrates": [{"name": "substrate A"}],
                "products": [{"name": "product B"}],
                "conditions": {"solvent": "THF", "temperature": "rt", "time": "12 h"},
                "targets": {"yield": "80%"},
            }
        ],
    }
    monkeypatch.setattr(
        cross_modal_kg,
        "iter_text_reaction_payloads",
        lambda _paths: iter([(Path("reaction.json"), payload)]),
    )

    triples = build_text_reaction_triples([Path("reaction.json")])

    assert next(row for row in triples if row["relationship"] == "HAS_CONDITION")["y_name"] == (
        "solvent: THF; temp: rt; time: 12 h"
    )
    assert {row["step"] for row in triples} == {""}


def test_image_entity_step_is_used_when_present_and_alignment_step_is_blank(monkeypatch):
    payload = {
        "source": "paper.pdf",
        "reactions": [
            {
                "id": "image-step-1",
                "source_image": "scheme_1.png",
                "reaction_type": "image test",
                "substrates": [{"name": "substrate A", "step": 1}],
                "products": [{"name": "product B", "step": 2}],
                "conditions": {"solvent": [{"step": 1, "value": "THF"}]},
                "targets": {"yield": "80%"},
            }
        ],
    }
    monkeypatch.setattr(
        cross_modal_kg,
        "iter_image_reaction_payloads",
        lambda _paths: iter([(Path("image.json"), payload)]),
    )

    triples = build_image_reaction_triples([Path("image.json")])

    assert next(row for row in triples if row["relationship"] == "USES_SUBSTRATE")["step"] == "1"
    assert next(row for row in triples if row["relationship"] == "PRODUCES")["step"] == "2"
    assert next(row for row in triples if row["relationship"] == "HAS_CONDITION")["step"] == "1"

    alignment = alignment_to_triple(
        {
            "relationship": "SAME_AS",
            "source_paper": "paper",
            "text_entity": {"name": "text compound", "role": "substrate"},
            "image_entity": {"name": "image compound", "role": "substrate"},
        }
    )
    assert alignment["step"] == ""


def test_step_column_order_in_kg_header():
    assert EDGE_FIELDNAMES[
        EDGE_FIELDNAMES.index("source_pages") + 1
    ] == "step"
