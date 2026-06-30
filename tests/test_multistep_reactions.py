import sys
from pathlib import Path

import pytest


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from batch_si_extractor import GP_CONTEXT_CHAR_LIMIT, SIExtractor  # noqa: E402
import cross_modal_kg  # noqa: E402
from cross_modal_kg import build_text_reaction_triples, relationship_counts  # noqa: E402
from langgraph_workflow.benchmark_utils import format_conditions_en  # noqa: E402
from pdf_to_gpt_extractor import PDFReactionExtractor  # noqa: E402
from split_by_substrate import has_valid_condition  # noqa: E402


def make_extractor():
    return SIExtractor.__new__(SIExtractor)


def gp1_reaction():
    return {
        "id": "GeneralProcedure1-Entry1",
        "reaction_type": "Sonogashira coupling followed by desilylation",
        "step_count": "Step 2",
        "substrates": [
            {"name": "1-bromo-4-vinylbenzene", "amount": "182.0 mg", "step": "Step 1"},
            {"name": "trimethylsilylacetylene", "amount": "196.0 mg", "step": "1"},
        ],
        "products": [
            {"name": "1-Ethynyl-4-vinylbenzene", "symbol": "S22", "amount": "84.5 mg", "step": 2}
        ],
        "catalysts": [
            {"name": "bis(triphenylphosphine)palladium(II) chloride", "step": 1},
            {"name": "copper(I) iodide", "step": "Step 1"},
        ],
        "additives": [{"name": "triethylamine", "amount": "4.0 mL", "step": 1}],
        "reagents": [{"name": "tetra-n-butylammonium fluoride", "step": "2"}],
        "intermediates": [
            {
                "name": "crude product thus obtained",
                "produced_in_step": 1,
                "consumed_in_step": 2,
            }
        ],
        "conditions": {
            "solvent": [
                {"step": "Step 1", "value": "triethylamine"},
                {"step": "2", "value": "anhydrous THF"},
            ],
            "temperature": [
                {"step": 1, "value": "50 C"},
                {"step": 2, "value": "0 C to room temperature"},
            ],
            "time": [
                {"step": 1, "value": "24 h"},
                {"step": 2, "value": "0.5 h"},
            ],
            "light source": None,
        },
        "targets": {"yield": "66% yield over two steps", "ee": None, "er": None},
    }


def test_gp1_multistep_schema_is_normalized_without_inventing_intermediate():
    reaction = make_extractor().sanitize_reaction_schema(gp1_reaction())

    assert GP_CONTEXT_CHAR_LIMIT == 1500
    assert "Multi-step reactions and General Procedures" in PDFReactionExtractor.EXTRACTION_PROMPT
    assert "two or more sequential chemical transformations" in PDFReactionExtractor.EXTRACTION_PROMPT
    assert "workup or purification" in PDFReactionExtractor.EXTRACTION_PROMPT
    assert "every concrete entry that references that GP must inherit the multi-step schema" in PDFReactionExtractor.EXTRACTION_PROMPT
    assert "the residue obtained above" not in PDFReactionExtractor.EXTRACTION_PROMPT
    assert "Determine whether the supplied GP describes multiple chemical transformations" in SIExtractor.GP_INJECTION_TEMPLATE
    assert "If any substrate, product, catalyst, additive, reagent, intermediate, or condition uses a step field" in PDFReactionExtractor.EXTRACTION_PROMPT
    assert "the reaction must include integer step_count" in SIExtractor.STAGE2_AUDIT_PROMPT
    assert reaction["step_count"] == 2
    assert [item["step"] for item in reaction["substrates"]] == [1, 1]
    assert reaction["products"][0]["step"] == 2
    assert [item["step"] for item in reaction["catalysts"]] == [1, 1]
    assert reaction["reagents"][0]["step"] == 2
    assert reaction["intermediates"] == []
    assert reaction["conditions"]["solvent"] == [
        {"step": 1, "value": "triethylamine"},
        {"step": 2, "value": "anhydrous THF"},
    ]
    assert reaction["conditions"]["light source"] == []
    assert reaction["targets"]["yield"] == "66% yield over two steps"


def test_named_intermediate_is_retained_and_registry_resolved():
    reaction = gp1_reaction()
    reaction["intermediates"] = [
        {
            "name": "S130-1",
            "symbol": "S130-1",
            "amount": "ca. 0.20 mmol",
            "produced_in_step": "Step 1",
            "consumed_in_step": "2",
        }
    ]
    normalized = make_extractor().sanitize_reaction_schema(reaction)
    resolved = make_extractor().align_names_in_reactions(
        [normalized], {"S130-1": "explicit intermediate name"}
    )[0]

    intermediate = resolved["intermediates"][0]
    assert intermediate["name"] == "explicit intermediate name"
    assert intermediate["produced_in_step"] == 1
    assert intermediate["consumed_in_step"] == 2


def test_multistep_entity_without_step_is_rejected():
    reaction = gp1_reaction()
    reaction["reagents"][0].pop("step")

    with pytest.raises(ValueError, match="invalid step"):
        make_extractor().sanitize_reaction_schema(reaction)


def test_missing_step_count_is_inferred_from_step_annotations():
    reaction = gp1_reaction()
    reaction.pop("step_count")
    reaction["conditions"]["atmosphere"] = "argon"

    normalized = make_extractor().sanitize_reaction_schema(reaction)

    assert normalized["step_count"] == 2
    assert normalized["products"][0]["step"] == 2
    assert normalized["conditions"]["solvent"] == [
        {"step": 1, "value": "triethylamine"},
        {"step": 2, "value": "anhydrous THF"},
    ]
    assert normalized["conditions"]["atmosphere"] == [{"step": 1, "value": "argon"}]
    assert normalized["intermediates"] == []


def test_single_step_step_one_pollution_is_removed():
    reaction = {
        "id": "single-step-polluted",
        "substrates": [{"name": "A", "amount": "1 mmol", "step": 1}],
        "products": [{"name": "B", "amount": "10 mg", "step": "Step 1"}],
        "catalysts": [],
        "additives": [],
        "reagents": [{"name": "base", "step": "1"}],
        "intermediates": [],
        "conditions": {
            "solvent": [{"step": 1, "value": "THF"}],
            "temperature": [{"step": "Step 1", "value": "room temperature"}],
            "time": "1 h",
        },
        "targets": {"yield": "80%"},
    }

    normalized = make_extractor().sanitize_reaction_schema(reaction)

    assert "step_count" not in normalized
    assert "intermediates" not in normalized
    assert "step" not in normalized["substrates"][0]
    assert "step" not in normalized["products"][0]
    assert "step" not in normalized["reagents"][0]
    assert normalized["conditions"] == {
        "solvent": "THF",
        "temperature": "room temperature",
        "time": "1 h",
    }


def test_single_step_reaction_remains_legacy_schema():
    reaction = {
        "id": "single-1",
        "substrates": [{"name": "A"}],
        "products": [{"name": "B"}],
        "catalysts": [],
        "additives": [],
        "reagents": [],
        "conditions": {"solvent": "THF", "time": "1 h"},
        "targets": {"yield": "80%"},
    }

    normalized = make_extractor().sanitize_reaction_schema(reaction)

    assert "step_count" not in normalized
    assert "intermediates" not in normalized
    assert "step" not in normalized["substrates"][0]
    assert normalized["conditions"] == {"solvent": "THF", "time": "1 h"}


def test_multistep_review_signal_is_weak_and_not_workup_only():
    extractor = make_extractor()

    assert extractor._multistep_review_needed(
        "The coupling product was obtained in 66% yield over two steps."
    )
    assert extractor._multistep_review_needed(
        "The crude residue was dissolved in THF and TBAF was added."
    )
    assert extractor._multistep_review_needed(
        "A one-pot two-step sequence furnished the final product."
    )
    assert not extractor._multistep_review_needed(
        "The reaction was quenched, extracted, washed, dried, concentrated, "
        "and purified by column chromatography."
    )
    assert not extractor._multistep_review_needed(
        "The mixture was then purified by column chromatography."
    )


def test_structured_conditions_are_visible_to_downstream_helpers():
    conditions = {
        "solvent": [
            {"step": 1, "value": "Et3N"},
            {"step": 2, "value": "THF"},
        ]
    }

    assert has_valid_condition({"conditions": conditions})
    assert format_conditions_en(conditions) == "solvent: Step 1: Et3N; Step 2: THF"


def test_intermediate_creates_both_kg_relationships(monkeypatch):
    reaction = gp1_reaction()
    reaction["intermediates"] = [
        {
            "name": "named intermediate",
            "symbol": "I1",
            "produced_in_step": 1,
            "consumed_in_step": 2,
        }
    ]
    reaction = make_extractor().sanitize_reaction_schema(reaction)
    payload = {"source": "paper.pdf", "reactions": [reaction]}
    monkeypatch.setattr(
        cross_modal_kg,
        "iter_text_reaction_payloads",
        lambda _paths: iter([(Path("reaction.json"), payload)]),
    )

    triples = build_text_reaction_triples([Path("reaction.json")])
    relationships = [
        triple["relationship"]
        for triple in triples
        if triple["y_name"] == "named intermediate"
    ]
    assert relationships.count("PRODUCES_INTERMEDIATE") == 1
    assert relationships.count("USES_INTERMEDIATE") == 1
    counts = relationship_counts(triples)
    assert counts["PRODUCES_INTERMEDIATE"] == 1
    assert counts["USES_INTERMEDIATE"] == 1


def test_reaction_signature_includes_intermediate_and_step_count():
    extractor = make_extractor()
    first = gp1_reaction()
    second = gp1_reaction()
    second["step_count"] = 3

    assert extractor._reaction_signature(first) != extractor._reaction_signature(second)
