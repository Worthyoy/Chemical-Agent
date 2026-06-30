import csv
import copy
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
    write_unified_multimodal_kg,
)


AMOUNT_COLUMNS = [
    "substrate_amount",
    "product_amount",
    "catalyst_amount",
    "additive_amount",
    "reagent_amount",
]

ROLE_EXPECTATIONS = {
    "USES_SUBSTRATE": ("substrate_amount", "1.0 mmol, 1.0 equiv."),
    "PRODUCES": ("product_amount", "25.0 mg"),
    "USES_CATALYST": ("catalyst_amount", "5.0 mol%"),
    "USES_ADDITIVE": ("additive_amount", "2.0 equiv."),
    "USES_REAGENT": ("reagent_amount", "3.0 mmol"),
}


def reaction_with_amounts():
    return {
        "id": "amount-test-1",
        "reaction_type": "test reaction",
        "substrates": [
            {
                "name": "substrate A",
                "amount": "  1.0 mmol,   1.0 equiv. ",
                "scaffold": "benzene",
                "substituents": ["4-methyl"],
            },
            {"name": "duplicate substrate", "amount": "1.0 mmol"},
            {"name": "duplicate substrate", "amount": "2.0 mmol"},
            {"name": "substrate without amount", "amount": None},
        ],
        "products": [{"name": "product B", "amount": "25.0 mg"}],
        "catalysts": [{"name": "catalyst C", "amount": "5.0 mol%"}],
        "additives": [{"name": "additive D", "amount": "2.0 equiv."}],
        "reagents": [{"name": "reagent E", "amount": "3.0 mmol"}],
        "intermediates": [
            {
                "name": "intermediate I",
                "amount": "10.0 mg",
                "produced_in_step": 1,
                "consumed_in_step": 2,
            }
        ],
        "conditions": {"solvent": "THF"},
        "targets": {"yield": "80%"},
    }


def assert_role_amounts(triples, modality):
    for relationship, (expected_column, expected_value) in ROLE_EXPECTATIONS.items():
        matches = [
            row
            for row in triples
            if row["relationship"] == relationship
            and row["source_modality"] == modality
        ]
        row = next(
            item
            for item in matches
            if item[expected_column] == expected_value
        )
        assert all(
            row[column] == (expected_value if column == expected_column else "")
            for column in AMOUNT_COLUMNS
        )


def test_text_entity_edges_store_only_the_matching_amount(monkeypatch):
    first = reaction_with_amounts()
    second = copy.deepcopy(first)
    second["id"] = "amount-test-2"
    payload = {"source": "paper.pdf", "reactions": [first, second]}
    monkeypatch.setattr(
        cross_modal_kg,
        "iter_text_reaction_payloads",
        lambda _paths: iter([(Path("reaction.json"), payload)]),
    )

    triples = build_text_reaction_triples([Path("reaction.json")])

    assert_role_amounts(triples, "text")
    expected_reaction_ids = {
        "TextReaction:paper:amount-test-1",
        "TextReaction:paper:amount-test-2",
    }
    assert {row["reaction_id"] for row in triples} == expected_reaction_ids
    missing = next(row for row in triples if row["y_name"] == "substrate without amount")
    assert all(missing[column] == "" for column in AMOUNT_COLUMNS)

    duplicate_rows = [
        row
        for row in triples
        if row["relationship"] == "USES_SUBSTRATE"
        and row["y_name"] == "duplicate substrate"
        and row["reaction_id"] == "TextReaction:paper:amount-test-1"
    ]
    assert sorted(row["substrate_amount"] for row in duplicate_rows) == ["1.0 mmol", "2.0 mmol"]

    scaffold_rows = [row for row in triples if row["relationship"] == "HAS_SCAFFOLD"]
    assert len(scaffold_rows) == 2
    assert {row["reaction_id"] for row in scaffold_rows} == expected_reaction_ids

    non_direct_relationships = {
        "REPORTED_IN",
        "HAS_CONDITION",
        "OF_TYPE",
        "HAS_SCAFFOLD",
        "HAS_SUBSTITUENT",
        "PRODUCES_INTERMEDIATE",
        "USES_INTERMEDIATE",
    }
    for row in triples:
        if row["relationship"] in non_direct_relationships:
            assert all(row[column] == "" for column in AMOUNT_COLUMNS)


def test_image_entity_edges_store_only_the_matching_amount(monkeypatch):
    reaction = reaction_with_amounts()
    reaction.pop("intermediates")
    payload = {
        "source": "paper.pdf",
        "reactions": [{**reaction, "source_image": "scheme_1.png"}],
    }
    monkeypatch.setattr(
        cross_modal_kg,
        "iter_image_reaction_payloads",
        lambda _paths: iter([(Path("image_reaction.json"), payload)]),
    )

    triples = build_image_reaction_triples([Path("image_reaction.json")])

    assert_role_amounts(triples, "image")
    assert {row["reaction_id"] for row in triples} == {
        "ImageReaction:paper:scheme_1.png:amount-test-1"
    }


def test_cross_modal_alignment_does_not_claim_a_single_reaction_id():
    triple = alignment_to_triple(
        {
            "relationship": "SAME_AS",
            "source_paper": "paper",
            "text_entity": {
                "name": "text compound",
                "iupac_name": "text compound",
                "reaction_id": "text-entry-1",
                "role": "substrate",
            },
            "image_entity": {
                "name": "image compound",
                "smiles": "CC",
                "reaction_id": "image-entry-1",
                "source_image": "scheme.png",
                "role": "substrate",
            },
            "match_method": "test",
            "confidence": "high",
        }
    )

    assert triple["source_modality"] == "text+image"
    assert triple["reaction_id"] == ""


def test_amount_columns_are_in_csv_header_and_intermediate_amount_is_absent(tmp_path):
    expected_prefix = [
        "x_name",
        "x_type",
        "relationship",
        "y_name",
        "y_type",
        "pdf_name",
        "reaction_id",
        "source_pages",
        "step",
        *AMOUNT_COLUMNS,
        "yield",
        "ee",
        "er",
        "dr",
        "source_modality",
        "source_image",
    ]
    assert EDGE_FIELDNAMES == expected_prefix
    assert "intermediate_amount" not in EDGE_FIELDNAMES

    output_path = tmp_path / "kg.csv"
    write_unified_multimodal_kg(output_path, [])
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        assert next(csv.reader(handle)) == EDGE_FIELDNAMES
