import csv
import json

from cross_modal_kg import (
    EDGE_FIELDNAMES,
    build_image_reaction_triples,
    build_text_reaction_triples,
)
from kg_to_reaction_csv import (
    COL_CONDITIONS,
    COL_SOLVENT,
    convert_kg_csv,
    convert_kg_rows,
)


def _reaction(conditions, *, reaction_id="r1"):
    return {
        "id": reaction_id,
        "source_pages": [7],
        "reaction_type": "test reaction",
        "substrates": [{"name": "substrate"}],
        "products": [{"name": "product"}],
        "catalysts": [],
        "ligands": [],
        "other_components": [],
        "conditions": conditions,
        "targets": {"yield": "80%", "ee": None, "er": None, "dr": None},
    }


def _write_payload(tmp_path, reaction, *, image=False):
    payload = {
        "source_paper": "paper",
        "reactions": [reaction],
    }
    if image:
        reaction["source_paper"] = "paper"
        reaction["source_image"] = "scheme.png"
    path = tmp_path / ("image.json" if image else "text.json")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(triples):
    rows = convert_kg_rows(triples)
    assert len(rows) == 1
    return rows[0]


def test_single_step_solvent_amount_is_a_direct_edge_and_paired_in_summary(tmp_path):
    path = _write_payload(
        tmp_path,
        _reaction(
            {
                "solvent": "tert-butyl acetate",
                "solvent_amount": "16 mL",
                "temperature": "room temperature",
                "time": "24 h",
            }
        ),
    )
    triples = build_text_reaction_triples([path])

    solvent_edges = [row for row in triples if row["relationship"] == "USES_SOLVENT"]
    assert "solvent_amount" in EDGE_FIELDNAMES
    assert len(solvent_edges) == 1
    assert solvent_edges[0]["y_name"] == "tert-butyl acetate"
    assert solvent_edges[0]["y_type"] == "Solvent"
    assert solvent_edges[0]["solvent_amount"] == "16 mL"
    assert solvent_edges[0]["step"] == ""

    condition_text = "; ".join(
        row["y_name"] for row in triples if row["relationship"] == "HAS_CONDITION"
    )
    assert "solvent" not in condition_text.casefold()
    assert "temp: room temperature" in condition_text
    assert "time: 24 h" in condition_text

    summary = _summary(triples)
    assert summary[COL_SOLVENT] == "tert-butyl acetate (16 mL)"
    assert summary[COL_CONDITIONS] == "temp: room temperature; time: 24 h"


def test_multistep_and_mixed_solvent_text_are_paired_without_splitting(tmp_path):
    path = _write_payload(
        tmp_path,
        _reaction(
            {
                "solvent": [
                    {"step": 1, "value": "methanol and 6M HCl"},
                    {"step": 2, "value": "CH2Cl2"},
                ],
                "solvent_amount": [
                    {"step": 1, "value": "8 mL and 8 mL"},
                    {"step": 2, "value": "12 mL"},
                ],
                "temperature": [
                    {"step": 1, "value": "45 °C"},
                    {"step": 2, "value": "0 °C"},
                ],
            },
            reaction_id="r2",
        ),
    )
    triples = build_text_reaction_triples([path])
    solvent_edges = [row for row in triples if row["relationship"] == "USES_SOLVENT"]

    assert [
        (row["y_name"], row["solvent_amount"], row["step"])
        for row in solvent_edges
    ] == [
        ("methanol and 6M HCl", "8 mL and 8 mL", "1"),
        ("CH2Cl2", "12 mL", "2"),
    ]
    summary = _summary(triples)
    assert summary[COL_SOLVENT] == (
        "step1: methanol and 6M HCl (8 mL and 8 mL); "
        "step2: CH2Cl2 (12 mL)"
    )
    assert "solvent" not in summary[COL_CONDITIONS].casefold()


def test_missing_and_unmatched_amounts_are_not_invented_or_dropped(tmp_path):
    missing_path = _write_payload(
        tmp_path,
        _reaction({"solvent": "anhydrous CH2Cl2"}, reaction_id="missing"),
    )
    missing_triples = build_text_reaction_triples([missing_path])
    missing_summary = _summary(missing_triples)
    assert missing_summary[COL_SOLVENT] == "anhydrous CH2Cl2"

    unmatched_path = _write_payload(
        tmp_path,
        _reaction({"solvent_amount": "25 mL"}, reaction_id="unmatched"),
    )
    unmatched_triples = build_text_reaction_triples([unmatched_path])
    assert not any(row["relationship"] == "USES_SOLVENT" for row in unmatched_triples)
    unmatched_conditions = [
        row for row in unmatched_triples if row["relationship"] == "HAS_CONDITION"
    ]
    assert [row["y_name"] for row in unmatched_conditions] == [
        "solvent_amount: 25 mL"
    ]
    unmatched_summary = _summary(unmatched_triples)
    assert unmatched_summary[COL_SOLVENT] == "solvent_amount: 25 mL"
    assert unmatched_summary[COL_CONDITIONS] == ""


def test_legacy_condition_solvent_fields_are_paired_and_removed_from_conditions():
    rows = [
        {
            "reaction_id": "TextReaction:paper:r1",
            "pdf_name": "paper",
            "source_modality": "text",
            "relationship": "HAS_CONDITION",
            "y_name": "solvent: THF; solvent_amount: 25 mL; temp: 0 °C",
            "step": "1",
        },
        {
            "reaction_id": "TextReaction:paper:r1",
            "pdf_name": "paper",
            "source_modality": "text",
            "relationship": "HAS_CONDITION",
            "y_name": "solvent: CH2Cl2; vol: 50 mL; time: 30 min",
            "step": "2",
        },
    ]
    summary = _summary(rows)
    assert summary[COL_SOLVENT] == "step1: THF (25 mL); step2: CH2Cl2 (50 mL)"
    assert summary[COL_CONDITIONS] == "step1: temp: 0 °C; step2: time: 30 min"


def test_normalized_image_reaction_emits_solvent_edge(tmp_path):
    path = _write_payload(
        tmp_path,
        _reaction(
            {"solvent": "toluene", "solvent_amount": "2 mL", "time": "1 h"}
        ),
        image=True,
    )
    triples = build_image_reaction_triples([path])
    solvent_edges = [row for row in triples if row["relationship"] == "USES_SOLVENT"]

    assert len(solvent_edges) == 1
    assert solvent_edges[0]["source_modality"] == "image"
    assert solvent_edges[0]["source_image"] == "scheme.png"
    assert solvent_edges[0]["y_name"] == "toluene"
    assert solvent_edges[0]["solvent_amount"] == "2 mL"
    assert _summary(triples)[COL_SOLVENT] == "toluene (2 mL)"


def test_csv_output_sorts_papers_descending_and_preserves_within_paper_order(tmp_path):
    input_path = tmp_path / "kg.csv"
    output_path = tmp_path / "summary.csv"
    rows = [
        {
            "reaction_id": "TextReaction:alpha:r1",
            "pdf_name": "Alpha",
            "source_modality": "text",
            "relationship": "PRODUCES",
            "y_name": "alpha product 1",
        },
        {
            "reaction_id": "TextReaction:charlie:r1",
            "pdf_name": "Charlie",
            "source_modality": "text",
            "relationship": "PRODUCES",
            "y_name": "charlie product",
        },
        {
            "reaction_id": "TextReaction:alpha:r2",
            "pdf_name": "Alpha",
            "source_modality": "text",
            "relationship": "PRODUCES",
            "y_name": "alpha product 2",
        },
        {
            "reaction_id": "TextReaction:bravo:r1",
            "pdf_name": "bravo",
            "source_modality": "text",
            "relationship": "PRODUCES",
            "y_name": "bravo product",
        },
    ]
    with input_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    convert_kg_csv(input_path, output_path)

    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert [row["文献"] for row in summary] == ["Charlie", "bravo", "Alpha", "Alpha"]
    assert [row["产物"] for row in summary[-2:]] == ["alpha product 1", "alpha product 2"]
