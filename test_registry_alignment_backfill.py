import csv
import json

from cross_modal_kg import (
    TEXT_SUBSTRATE_NAME_POLICY_REGISTRY_IF_RESOLVED_ELSE_ORIGINAL,
)
from kg_to_reaction_csv import COL_SUBSTRATE, COL_YIELD
from registry_alignment_backfill import run_backfill


def test_backfill_writes_corrected_json_audit_kg_and_summary_without_model(tmp_path):
    input_dir = tmp_path / "input"
    output_root = tmp_path / "registry_backfill_v2"
    input_dir.mkdir()
    registry_name = "Methyl substituted enamine"
    source_payload = {
        "source": "paper.pdf",
        "total_reactions": 1,
        "name_registry": {"3k": registry_name},
        "reactions": [
            {
                "id": "GeneralProcedureF-Entry11",
                "source_pages": [106],
                "reaction_type": "Pd-catalyzed asymmetric amination",
                "substrates": [
                    {
                        "name": registry_name,
                        "symbol": "3k",
                        "original_name": "3",
                        "registry_name": registry_name,
                        "registry_match_status": "conflict",
                        "registry_conflict_reason": (
                            "extracted_name_differs_from_registry_symbol_name"
                        ),
                        "identity_status": "generic",
                        "resolution_status": "resolved_from_product",
                        "resolution_source": "product_name",
                        "resolution_method": "product_name_to_substrate_mapping",
                        "resolution_evidence": {"product_name": "product 4k"},
                        "amount": "0.2 mmol, 1.0 equiv.",
                    }
                ],
                "products": [{"name": "product 4k", "symbol": "4k"}],
                "catalysts": [],
                "ligands": [],
                "other_components": [],
                "conditions": {"temperature": "80 °C", "time": "24 h"},
                "targets": {"yield": "91%", "ee": "96%", "er": "98:2", "dr": None},
            }
        ],
    }
    input_path = input_dir / "paper.json"
    input_path.write_text(json.dumps(source_payload), encoding="utf-8")

    report = run_backfill(
        input_dir,
        output_root,
        text_substrate_name_policy=(
            TEXT_SUBSTRATE_NAME_POLICY_REGISTRY_IF_RESOLVED_ELSE_ORIGINAL
        ),
    )

    assert report["model_calls"] == 0
    assert report["files_processed"] == 1
    assert report["total_reactions"] == 1
    assert report["changed_compounds"] == 1
    corrected_path = output_root / "output" / "paper.json"
    corrected = json.loads(corrected_path.read_text(encoding="utf-8"))
    substrate = corrected["reactions"][0]["substrates"][0]
    assert substrate["name"] == registry_name
    assert substrate["symbol"] == "3k"
    assert substrate["original_name"] == "3"
    assert substrate["resolution_source"] == "name_registry"
    assert substrate["resolution_status"] == "resolved_from_registry"
    assert substrate["resolution_evidence"]["match_type"] == (
        "underspecified_series_label"
    )
    assert "registry_match_status" not in substrate
    assert source_payload["reactions"][0]["targets"] == corrected["reactions"][0]["targets"]

    audit = json.loads(
        (output_root / "registry_alignment_audit.json").read_text(encoding="utf-8")
    )
    assert audit["model_calls"] == 0
    assert audit["changes"][0]["reaction_id"] == "GeneralProcedureF-Entry11"
    assert audit["changes"][0]["reason"] == "underspecified_series_label"

    kg_path = output_root / "kg_original" / "kg_triples_unified_multimodal.csv"
    with open(kg_path, newline="", encoding="utf-8-sig") as handle:
        kg_rows = list(csv.DictReader(handle))
    substrate_edge = next(
        row for row in kg_rows if row["relationship"] == "USES_SUBSTRATE"
    )
    assert substrate_edge["y_name"] == registry_name

    summary_path = output_root / "kg_original" / "reaction_summary_from_kg.csv"
    with open(summary_path, newline="", encoding="utf-8-sig") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 1
    assert summary_rows[0][COL_SUBSTRATE] == (
        f"{registry_name} (0.2 mmol, 1.0 equiv.)"
    )
    assert summary_rows[0][COL_YIELD] == "91%"
    assert summary_rows[0]["ee"] == "96%"
    assert summary_rows[0]["er"] == "98:2"
