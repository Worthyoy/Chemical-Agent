import json

from batch_si_extractor import SIExtractor
from cross_modal_kg import target_values
from langgraph_workflow.benchmark_utils import (
    _score_from_targets,
    generate_q2_benchmark_package,
)
from kg_to_reaction_csv import convert_kg_rows
from pdf_to_gpt_extractor import PDFReactionExtractor
from reaction_filter import ReactionFilter, filter_reaction_file


def _reaction(targets=None, *, reaction_id="r1", variable_name="substrate A"):
    return {
        "id": reaction_id,
        "source_paper": "paper",
        "source_modality": "text",
        "source_pages": [1],
        "reaction_type": "test reaction",
        "substrates": [
            {"name": variable_name, "scaffold": "variable scaffold", "substituents": []},
            {"name": "fixed substrate", "scaffold": "fixed scaffold", "substituents": []},
        ],
        "products": [{"name": f"product from {variable_name}", "scaffold": "product"}],
        "catalysts": [{"name": "catalyst", "amount": "1 mol%"}],
        "ligands": [],
        "other_components": [],
        "conditions": {"solvent": "solvent", "time": "1 h"},
        "targets": targets or {"yield": None, "ee": None, "er": None, "dr": None},
    }


def test_reaction_prompts_include_dr_and_no_longer_forbid_it():
    for prompt in (
        SIExtractor.STRUCTURED_REACTION_PROMPT,
        SIExtractor.NON_GP_REACTION_PROMPT,
        SIExtractor.MIXED_REACTION_PROMPT,
    ):
        assert '"dr":null' in prompt
        assert "91:9 dr" in prompt
        assert "dr = 95:5" in prompt
        assert ">20:1 dr" in prompt
        assert "coverage lists, dr" not in prompt
    assert '"dr": "..."' in PDFReactionExtractor.EXTRACTION_PROMPT


def test_sanitizer_normalizes_nested_dr_suffix_without_changing_ratio():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    sanitized = extractor.sanitize_reaction_schema(
        _reaction({"yield": None, "ee": None, "er": None, "dr": "  >20:1 dr  "})
    )
    assert sanitized["targets"] == {
        "yield": None,
        "ee": None,
        "er": None,
        "dr": ">20:1",
    }


def test_sanitizer_migrates_legacy_top_level_dr():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = _reaction({"yield": None, "ee": None, "er": None})
    reaction["dr"] = "dr = 95:5"
    sanitized = extractor.sanitize_reaction_schema(reaction)
    assert "dr" not in {key for key in sanitized if key != "targets"}
    assert sanitized["targets"]["dr"] == "95:5"


def test_sanitizer_does_not_infer_dr_from_de_or_unlabeled_fields():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = _reaction({"yield": None, "ee": None, "er": None})
    reaction["de"] = "82%"
    reaction["selectivity"] = "91:9"
    sanitized = extractor.sanitize_reaction_schema(reaction)
    assert sanitized["targets"]["dr"] is None


def test_filter_accepts_valid_dr_only_and_rejects_invalid_dr_only():
    reaction_filter = ReactionFilter()
    assert reaction_filter.check_numeric_criteria(
        _reaction({"yield": None, "ee": None, "er": None, "dr": "91:9"})
    )
    assert not reaction_filter.check_numeric_criteria(
        _reaction({"yield": None, "ee": None, "er": None, "dr": "high dr"})
    )


def test_filtered_json_preserves_ligand_amount_and_null(tmp_path):
    reported = _reaction({"yield": "80%", "ee": None, "er": None, "dr": None})
    reported["ligands"] = [
        {
            "name": "L10",
            "symbol": "L10",
            "amount": "9.4 mg, 0.015 mmol, 15 mol%",
        }
    ]
    unreported = _reaction(
        {"yield": "75%", "ee": None, "er": None, "dr": None},
        reaction_id="r2",
        variable_name="substrate B",
    )
    unreported["ligands"] = [{"name": "BINAP", "amount": None}]

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "filtered.json"
    input_path.write_text(
        json.dumps({"source": "paper", "reactions": [reported, unreported]}),
        encoding="utf-8",
    )

    result = filter_reaction_file(input_path, output_path, overwrite=True)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["reactions"] == 2
    assert payload["reactions"][0]["ligands"] == reported["ligands"]
    assert payload["reactions"][1]["ligands"] == unreported["ligands"]


def test_stage2_target_cue_detects_both_dr_orders():
    assert SIExtractor._chunk_has_target_cue("chiral analysis gave dr = 95:5")
    assert SIExtractor._chunk_has_target_cue("the product was obtained in >20:1 dr")


def test_kg_target_values_preserve_dr_string():
    assert target_values({"targets": {"dr": ">20:1"}})["dr"] == ">20:1"


def test_reaction_summary_csv_conversion_preserves_dr_string():
    rows = [
        {
            "reaction_id": "paper::r1",
            "pdf_name": "paper.pdf",
            "source_modality": "text",
            "relationship": "PRODUCES",
            "y_name": "product",
            "source_pages": "[1]",
            "dr": "93:7",
        }
    ]
    summary = convert_kg_rows(rows)
    assert summary[0]["dr"] == "93:7"


def test_dr_does_not_change_benchmark_score():
    without_dr = _score_from_targets({"yield": "80%", "ee": "90%", "dr": None})
    with_dr = _score_from_targets({"yield": "80%", "ee": "90%", "dr": "99:1"})
    assert without_dr == with_dr


def test_q2_dr_only_group_is_reviewed_without_arbitrary_gold():
    package = generate_q2_benchmark_package(
        [
            _reaction(
                {"yield": None, "ee": None, "er": None, "dr": "91:9"},
                reaction_id="r1",
                variable_name="substrate A",
            ),
            _reaction(
                {"yield": None, "ee": None, "er": None, "dr": "93:7"},
                reaction_id="r2",
                variable_name="substrate B",
            ),
        ]
    )
    assert package["benchmark"] == []
    assert any(
        review.get("reason") == "no_rankable_yield_or_ee"
        for review in package["review_set"]
    )


def test_q2_keeps_dr_only_option_as_unranked_metadata_when_group_is_rankable():
    package = generate_q2_benchmark_package(
        [
            _reaction(
                {"yield": "80%", "ee": None, "er": None, "dr": None},
                reaction_id="r1",
                variable_name="substrate A",
            ),
            _reaction(
                {"yield": "75%", "ee": None, "er": None, "dr": None},
                reaction_id="r2",
                variable_name="substrate B",
            ),
            _reaction(
                {"yield": None, "ee": None, "er": None, "dr": "91:9"},
                reaction_id="r3",
                variable_name="substrate C",
            ),
        ]
    )
    question = package["benchmark"][0]
    result = next(
        item
        for item in question["metadata_hidden"]["option_results"]
        if item["reaction_id"] == "r3"
    )
    assert result["dr"] == "91:9"
    assert result["dr_raw"] == "91:9"
    assert result["rankable"] is False
    assert result["option_id"] not in question["gold_option_ids"]


def test_q2_duplicate_result_conflict_includes_dr():
    package = generate_q2_benchmark_package(
        [
            _reaction(
                {"yield": "80%", "ee": None, "er": None, "dr": "91:9"},
                reaction_id="r1",
                variable_name="substrate A",
            ),
            _reaction(
                {"yield": "80%", "ee": None, "er": None, "dr": "93:7"},
                reaction_id="r2",
                variable_name="substrate A",
            ),
            _reaction(
                {"yield": "79%", "ee": None, "er": None, "dr": "95:5"},
                reaction_id="r3",
                variable_name="substrate B",
            ),
            _reaction(
                {"yield": "78%", "ee": None, "er": None, "dr": "96:4"},
                reaction_id="r4",
                variable_name="substrate C",
            ),
        ]
    )
    assert len(package["benchmark"]) == 1
    reviews = [
        review
        for review in package["review_set"]
        if review.get("reason") == "duplicate_variable_substrate_conflicting_results"
    ]
    assert len(reviews) == 1
    assert {item["dr"] for item in reviews[0]["details"][0]["results"]} == {
        "91:9",
        "93:7",
    }
