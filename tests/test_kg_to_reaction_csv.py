"""Tests for converting KG edge CSV rows to reaction summary CSV rows."""

import csv
import sys
from pathlib import Path


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from kg_to_reaction_csv import (  # noqa: E402
    OUTPUT_FIELDNAMES,
    convert_kg_csv,
    convert_kg_rows,
)


def row(
    reaction_id="TextReaction:paper:GeneralProcedure1-Entry1",
    relationship="USES_SUBSTRATE",
    y_name="substrate A",
    step="",
    source_pages="1",
    pdf_name="paper",
    **kwargs,
):
    base = {
        "x_name": reaction_id,
        "x_type": "Reaction",
        "relationship": relationship,
        "y_name": y_name,
        "y_type": "",
        "pdf_name": pdf_name,
        "reaction_id": reaction_id,
        "source_pages": source_pages,
        "step": step,
        "substrate_amount": "",
        "product_amount": "",
        "catalyst_amount": "",
        "additive_amount": "",
        "reagent_amount": "",
        "yield": "80%",
        "ee": "",
        "er": "",
        "dr": "",
        "source_modality": "text",
        "source_image": "",
    }
    base.update(kwargs)
    return base


def test_convert_single_step_kg_rows_to_summary_row():
    rows = [
        row(relationship="REPORTED_IN", y_name="paper"),
        row(relationship="USES_SUBSTRATE", y_name="substrate A", substrate_amount="1.0 mmol"),
        row(relationship="PRODUCES", y_name="product B", product_amount="25 mg"),
        row(relationship="USES_REAGENT", y_name="reagent C", reagent_amount="2 equiv"),
        row(relationship="HAS_CONDITION", y_name="solvent: THF; temp: rt; time: 12 h"),
    ]

    output = convert_kg_rows(rows)

    assert len(output) == 1
    assert output[0] == {
        "文献": "paper",
        "反应位置": "GeneralProcedure1-Entry1; PDF pages: 1",
        "底物": "substrate A (1.0 mmol)",
        "产物": "product B (25 mg)",
        "中间体": "",
        "催化剂": "",
        "添加剂": "",
        "试剂": "reagent C (2 equiv)",
        "条件（温度、时间、溶剂等）": "solvent: THF; temp: rt; time: 12 h",
        "产率": "80%",
        "dr": "",
        "ee": "",
        "er": "",
    }


def test_convert_multistep_kg_rows_groups_values_by_step_and_intermediate():
    rows = [
        row(relationship="USES_SUBSTRATE", y_name="substrate A", step="1", substrate_amount="1 mmol"),
        row(relationship="USES_SUBSTRATE", y_name="substrate B", step="2", substrate_amount="2 mmol"),
        row(relationship="PRODUCES", y_name="product C", step="2", product_amount="10 mg"),
        row(relationship="USES_CATALYST", y_name="catalyst D", step="1", catalyst_amount="5 mol%"),
        row(relationship="USES_ADDITIVE", y_name="additive E", step="1", additive_amount="2 equiv"),
        row(relationship="USES_REAGENT", y_name="reagent F", step="2", reagent_amount="3 mmol"),
        row(relationship="HAS_CONDITION", y_name="solvent: Et3N; temp: 50 °C", step="1"),
        row(relationship="HAS_CONDITION", y_name="solvent: THF; temp: rt", step="2"),
        row(relationship="PRODUCES_INTERMEDIATE", y_name="intermediate I", step="1"),
        row(relationship="USES_INTERMEDIATE", y_name="intermediate I", step="2"),
    ]

    output = convert_kg_rows(rows)

    assert output[0]["底物"] == "step1: substrate A (1 mmol); step2: substrate B (2 mmol)"
    assert output[0]["产物"] == "step2: product C (10 mg)"
    assert output[0]["催化剂"] == "step1: catalyst D (5 mol%)"
    assert output[0]["添加剂"] == "step1: additive E (2 equiv)"
    assert output[0]["试剂"] == "step2: reagent F (3 mmol)"
    assert output[0]["条件（温度、时间、溶剂等）"] == (
        "step1: solvent: Et3N; temp: 50 °C; step2: solvent: THF; temp: rt"
    )
    assert output[0]["中间体"] == "step1: intermediate I"


def test_reaction_location_merges_and_sorts_source_pages():
    rows = [
        row(source_pages="118, 117", relationship="USES_SUBSTRATE"),
        row(source_pages="117", relationship="PRODUCES", y_name="product B"),
    ]

    output = convert_kg_rows(rows)

    assert output[0]["反应位置"] == "GeneralProcedure1-Entry1; PDF pages: 117, 118"


def test_reaction_location_omits_pdf_pages_when_missing():
    rows = [
        row(source_pages="", relationship="USES_SUBSTRATE"),
    ]

    output = convert_kg_rows(rows)

    assert output[0]["反应位置"] == "GeneralProcedure1-Entry1"


def test_convert_mixed_step_and_plain_values_keeps_plain_first():
    rows = [
        row(relationship="USES_REAGENT", y_name="plain reagent", reagent_amount="1 equiv"),
        row(relationship="USES_REAGENT", y_name="step reagent", step="2", reagent_amount="2 equiv"),
    ]

    output = convert_kg_rows(rows)

    assert output[0]["试剂"] == "plain reagent (1 equiv); step2: step reagent (2 equiv)"


def test_convert_kg_csv_writes_utf8_sig_csv_with_expected_header(tmp_path):
    kg_path = tmp_path / "kg.csv"
    output_path = tmp_path / "reactions.csv"
    rows = [
        row(relationship="USES_SUBSTRATE", y_name="substrate A", substrate_amount="1 mmol"),
        row(relationship="PRODUCES", y_name="product B", product_amount="2 mg"),
    ]
    with kg_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = convert_kg_csv(kg_path, output_path)

    assert result["input_edges"] == 2
    assert result["output_reactions"] == 1
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader) == OUTPUT_FIELDNAMES
