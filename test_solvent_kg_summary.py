import csv
import json

from cross_modal_kg import (
    EDGE_FIELDNAMES,
    TEXT_SUBSTRATE_NAME_POLICY_ORIGINAL_IF_AVAILABLE,
    TEXT_SUBSTRATE_NAME_POLICY_REGISTRY_IF_RESOLVED_ELSE_ORIGINAL,
    build_image_reaction_triples,
    build_text_reaction_triples,
    extract_text_compounds,
)
from kg_to_reaction_csv import (
    COL_CONDITIONS,
    COL_INTRINSIC_EXTRACTION_ERROR,
    COL_LIGAND,
    COL_PROCEDURAL_FIDELITY_ERROR,
    COL_SOLVENT,
    COL_SUBSTRATE,
    OUTPUT_FIELDNAMES,
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


def test_ligand_amount_is_preserved_on_text_and_image_kg_edges(tmp_path):
    text_reaction = _reaction({}, reaction_id="text-ligand")
    text_reaction["ligands"] = [
        {
            "name": "L10",
            "symbol": "L10",
            "amount": "9.4 mg, 0.015 mmol, 15 mol%",
        }
    ]
    text_path = _write_payload(tmp_path, text_reaction)

    image_reaction = _reaction({}, reaction_id="image-ligand")
    image_reaction["ligands"] = [{"name": "BINAP", "amount": "20 mol%"}]
    image_path = _write_payload(tmp_path, image_reaction, image=True)

    text_edge = next(
        edge
        for edge in build_text_reaction_triples([text_path])
        if edge["relationship"] == "USES_LIGAND"
    )
    image_edge = next(
        edge
        for edge in build_image_reaction_triples([image_path])
        if edge["relationship"] == "USES_LIGAND"
    )

    assert "ligand_amount" in EDGE_FIELDNAMES
    assert text_edge["y_name"] == "L10"
    assert text_edge["ligand_amount"] == "9.4 mg, 0.015 mmol, 15 mol%"
    assert text_edge["catalyst_amount"] == ""
    assert image_edge["y_name"] == "BINAP"
    assert image_edge["ligand_amount"] == "20 mol%"
    assert _summary(build_text_reaction_triples([text_path]))[COL_LIGAND] == (
        "L10 (9.4 mg, 0.015 mmol, 15 mol%)"
    )


def test_original_substrate_name_policy_propagates_to_kg_and_summary(tmp_path):
    reaction = _reaction({}, reaction_id="original-substrate")
    reaction["substrates"] = [
        {
            "name": "4-fluorophenylboronic acid",
            "original_name": "arylboronic acid 2",
            "resolution_source": "product_name",
            "resolution_method": "product_name_to_substrate_mapping",
            "amount": "0.2 mmol",
            "step": 1,
            "scaffold": "arylboronic acid",
            "substituents": ["R"],
        },
        {
            "name": "resolved registry compound",
            "original_name": "aryl halide",
            "resolution_source": "name_registry",
            "amount": "0.1 mmol",
        },
        {"name": "unchanged substrate", "amount": "0.3 mmol"},
    ]
    reaction["products"] = [
        {
            "name": "resolved product",
            "original_name": "generic product",
        }
    ]
    path = _write_payload(tmp_path, reaction)

    default_triples = build_text_reaction_triples([path])
    default_substrates = [
        row["y_name"]
        for row in default_triples
        if row["relationship"] == "USES_SUBSTRATE"
    ]
    assert default_substrates == [
        "4-fluorophenylboronic acid",
        "resolved registry compound",
        "unchanged substrate",
    ]

    triples = build_text_reaction_triples(
        [path],
        text_substrate_name_policy=(
            TEXT_SUBSTRATE_NAME_POLICY_ORIGINAL_IF_AVAILABLE
        ),
    )
    substrate_edges = [
        row for row in triples if row["relationship"] == "USES_SUBSTRATE"
    ]
    assert [row["y_name"] for row in substrate_edges] == [
        "arylboronic acid 2",
        "aryl halide",
        "unchanged substrate",
    ]
    assert substrate_edges[0]["substrate_amount"] == "0.2 mmol"
    assert substrate_edges[0]["step"] == "1"

    scaffold_edge = next(row for row in triples if row["relationship"] == "HAS_SCAFFOLD")
    substituent_edge = next(
        row for row in triples if row["relationship"] == "HAS_SUBSTITUENT"
    )
    assert scaffold_edge["x_name"] == "arylboronic acid 2"
    assert substituent_edge["x_name"] == "arylboronic acid 2"

    product_edge = next(row for row in triples if row["relationship"] == "PRODUCES")
    assert product_edge["y_name"] == "resolved product"

    summary = _summary(triples)
    assert summary[COL_SUBSTRATE] == (
        "aryl halide (0.1 mmol); unchanged substrate (0.3 mmol); "
        "step1: arylboronic acid 2 (0.2 mmol)"
    )

    substrate_entities = [
        entity
        for entity in extract_text_compounds(
            [path],
            text_substrate_name_policy=(
                TEXT_SUBSTRATE_NAME_POLICY_ORIGINAL_IF_AVAILABLE
            ),
        )
        if entity.role == "substrate"
    ]
    assert [entity.name for entity in substrate_entities] == [
        "arylboronic acid 2",
        "aryl halide",
        "unchanged substrate",
    ]


def test_original_policy_uses_reported_symbol_for_truncated_series_label(tmp_path):
    reaction = _reaction({}, reaction_id="truncated-series-label")
    reaction["substrates"] = [
        {
            "name": "Methyl substituted enamine",
            "symbol": "3k",
            "original_name": "3",
            "amount": "0.2 mmol",
            "scaffold": "enamine",
            "substituents": ["3-chloro-2-methylphenyl"],
            "resolution_source": "name_registry",
            "resolution_method": "same_paper_symbol",
            "resolution_evidence": {
                "match_type": "underspecified_series_label",
                "reported_name": "3",
                "reported_symbol": "3k",
                "registry_symbol": "3k",
            },
        }
    ]
    path = _write_payload(tmp_path, reaction)

    triples = build_text_reaction_triples(
        [path],
        text_substrate_name_policy=(
            TEXT_SUBSTRATE_NAME_POLICY_ORIGINAL_IF_AVAILABLE
        ),
    )
    substrate_edge = next(
        row for row in triples if row["relationship"] == "USES_SUBSTRATE"
    )
    scaffold_edge = next(row for row in triples if row["relationship"] == "HAS_SCAFFOLD")
    substituent_edge = next(
        row for row in triples if row["relationship"] == "HAS_SUBSTITUENT"
    )
    assert substrate_edge["y_name"] == "3k"
    assert scaffold_edge["x_name"] == "3k"
    assert substituent_edge["x_name"] == "3k"
    assert _summary(triples)[COL_SUBSTRATE] == "3k (0.2 mmol)"

    substrate_entities = [
        entity
        for entity in extract_text_compounds(
            [path],
            text_substrate_name_policy=(
                TEXT_SUBSTRATE_NAME_POLICY_ORIGINAL_IF_AVAILABLE
            ),
        )
        if entity.role == "substrate"
    ]
    assert [entity.name for entity in substrate_entities] == ["3k"]


def test_registry_success_else_original_policy_propagates_to_kg_and_summary(tmp_path):
    reaction = _reaction({}, reaction_id="registry-success-else-original")
    reaction["substrates"] = [
        {
            "name": "registry-resolved full substrate",
            "iupac_name": "stale pre-registry IUPAC name",
            "symbol": "3l",
            "original_name": "3",
            "resolution_source": "name_registry",
            "resolution_method": "same_paper_symbol",
            "amount": "0.2 mmol",
            "scaffold": "resolved scaffold",
            "substituents": ["resolved substituent"],
        },
        {
            "name": "model-inferred substrate",
            "original_name": "generic aryl halide",
            "resolution_source": "product_name",
            "resolution_method": "product_name_to_substrate_mapping",
            "amount": "0.1 mmol",
        },
        {
            "name": "directly extracted concrete name",
            "original_name": "directly extracted original name",
            "symbol": "5a",
            "registry_name": "conflicting registry name",
            "registry_match_status": "conflict",
            "resolution_source": "name_registry",
            "amount": "0.3 mmol",
        },
        {
            "name": "direct conflict without original_name",
            "symbol": "6a",
            "registry_name": "another conflicting registry name",
            "registry_match_status": "conflict",
            "amount": "0.4 mmol",
        },
        {"name": "unchanged extracted substrate", "amount": "0.5 mmol"},
    ]
    reaction["products"] = [{"name": "resolved product"}]
    path = _write_payload(tmp_path, reaction)

    policy = TEXT_SUBSTRATE_NAME_POLICY_REGISTRY_IF_RESOLVED_ELSE_ORIGINAL
    triples = build_text_reaction_triples(
        [path],
        text_substrate_name_policy=policy,
    )
    substrate_edges = [
        row for row in triples if row["relationship"] == "USES_SUBSTRATE"
    ]
    assert [row["y_name"] for row in substrate_edges] == [
        "registry-resolved full substrate",
        "generic aryl halide",
        "directly extracted original name",
        "direct conflict without original_name",
        "unchanged extracted substrate",
    ]
    assert "conflicting registry name" not in {
        row["y_name"] for row in substrate_edges
    }
    scaffold_edge = next(row for row in triples if row["relationship"] == "HAS_SCAFFOLD")
    substituent_edge = next(
        row for row in triples if row["relationship"] == "HAS_SUBSTITUENT"
    )
    assert scaffold_edge["x_name"] == "registry-resolved full substrate"
    assert substituent_edge["x_name"] == "registry-resolved full substrate"
    product_edge = next(row for row in triples if row["relationship"] == "PRODUCES")
    assert product_edge["y_name"] == "resolved product"

    summary = _summary(triples)
    assert summary[COL_SUBSTRATE] == (
        "registry-resolved full substrate (0.2 mmol); "
        "generic aryl halide (0.1 mmol); "
        "directly extracted original name (0.3 mmol); "
        "direct conflict without original_name (0.4 mmol); "
        "unchanged extracted substrate (0.5 mmol)"
    )
    substrate_entities = [
        entity
        for entity in extract_text_compounds(
            [path],
            text_substrate_name_policy=policy,
        )
        if entity.role == "substrate"
    ]
    assert [entity.name for entity in substrate_entities] == [
        "registry-resolved full substrate",
        "generic aryl halide",
        "directly extracted original name",
        "direct conflict without original_name",
        "unchanged extracted substrate",
    ]


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
        reader = csv.DictReader(handle)
        summary = list(reader)
    assert [row["文献"] for row in summary] == ["Charlie", "bravo", "Alpha", "Alpha"]
    assert [row["产物"] for row in summary[-2:]] == ["alpha product 1", "alpha product 2"]
    assert reader.fieldnames == OUTPUT_FIELDNAMES
    assert reader.fieldnames[-2:] == [
        COL_INTRINSIC_EXTRACTION_ERROR,
        COL_PROCEDURAL_FIDELITY_ERROR,
    ]
    for row in summary:
        assert row[COL_INTRINSIC_EXTRACTION_ERROR] == ""
        assert row[COL_PROCEDURAL_FIDELITY_ERROR] == ""
