from batch_si_extractor import (
    GP_TEMPLATE_PROMPT_VERSION,
    GP_TEMPLATE_SCHEMA_VERSION,
    MIXED_REACTION_PROMPT_VERSION,
    SIExtractor,
)
from langgraph_workflow.pipeline_agents import PipelineConfig
from pdf_to_gpt_extractor import PDFReactionExtractor


def _gp_c_payload():
    return {
        "reaction_type": "nickel-catalyzed photochemical cross-coupling",
        "step_count": 3,
        "substrates": [
            {"name": "alkenyl boronate", "amount": "0.5 mmol", "step": 1},
            {"name": "NHPI ester", "amount": "0.2 mmol", "step": 3},
        ],
        "catalysts": [
            {
                "name": "Ni(BF4)•6H2O",
                "amount": "6.8 mg, 0.02 mmol, 10 mol%",
                "step": 2,
            }
        ],
        "ligands": [
            {
                "name": "L6",
                "amount": "9.4 mg, 0.015 mmol, 15 mol%",
                "step": 2,
            }
        ],
        "other_components": [
            {"name": "ZrCp2HCl", "amount": "139.2 mg, 0.5 mmol, 2.7 equiv", "step": 1}
        ],
        "intermediates": [],
        "conditions": {
            "solvent": [{"step": 1, "value": "THF"}],
            "solvent_amount": [{"step": 1, "value": "1.0 mL"}],
            "temperature": [{"step": 1, "value": "50 °C"}],
            "time": [{"step": 1, "value": "1 h"}],
            "atmosphere": [],
            "light_source": [],
            "wavelength": [],
        },
        "procedure_details": [],
        "evidence": {},
    }


def test_gp_template_prompt_requires_atomic_name_and_amount_fields():
    prompt = SIExtractor.GP_TEMPLATE_PROMPT

    assert GP_TEMPLATE_SCHEMA_VERSION == "gp_template_v2_ligand_amount"
    assert GP_TEMPLATE_PROMPT_VERSION == "gp_template_prompt_v8_ligand_amount"
    assert "name stores chemical identity only" in prompt
    assert "amount stores every explicitly reported numerical quantity" in prompt
    assert "Move any parenthetical text that reports a numerical quantity or loading into amount" in prompt
    assert '"name":"reported catalyst","amount":"reported mass, reported mmol, reported mol%"' in prompt
    assert '"name":"reported catalyst (reported mass, reported mol%)","amount":null' in prompt
    assert "preformed metal-ligand complex remains a complete catalyst name" in prompt
    assert "Preserve explicitly reported free-ligand quantities exactly" in prompt
    assert "use amount=null when unreported" in prompt
    assert "Do not infer ligand amount from catalyst-complex amount" in prompt


def test_ligand_amount_version_bumps_invalidate_reaction_and_gp_caches():
    assert MIXED_REACTION_PROMPT_VERSION == "mixed_reaction_prompt_v6_ligand_amount"
    assert (
        PipelineConfig.__dataclass_fields__["pipeline_version"].default
        == "parallel_pdf_v24_ligand_amount"
    )


def test_gp_template_prompt_counts_main_lineage_and_not_parallel_catalyst_premixing():
    prompt = SIExtractor.GP_TEMPLATE_PROMPT

    assert "main substrate-to-final-product transformation lineage" in prompt
    assert "Parallel preparation, premixing, activation, or aging" in prompt
    assert "does not create step 3" in prompt
    assert "A -> intermediate X -> final product" in prompt
    assert "General Procedure C" in prompt
    assert "Never use the GP id" in prompt
    assert "reaction boundary controls role assignment" in prompt


def test_base_pdf_prompt_requires_ligand_amount_without_complex_inference():
    prompt = PDFReactionExtractor.EXTRACTION_PROMPT

    assert "ligands items contain name and amount" in prompt
    assert "amount=null when unreported" in prompt
    assert "Do not infer a separate ligand amount" in prompt
    assert '"ligands": [{"name": "...", "amount": "... or null"}]' in prompt


def test_gp_c_atomic_catalyst_fixture_preserves_name_amount_and_step():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    template = extractor.validate_gp_template(
        _gp_c_payload(),
        gp_label="GeneralProcedureC",
        gp_text="General Procedure C fixture",
        source_pages=[32],
    )

    assert template["catalysts"] == [
        {
            "name": "Ni(BF4)•6H2O",
            "amount": "6.8 mg, 0.02 mmol, 10 mol%",
            "step": 2,
        }
    ]
    assert template["ligands"] == [
        {
            "name": "L6",
            "amount": "9.4 mg, 0.015 mmol, 15 mol%",
            "step": 2,
        }
    ]
    assert template["conditions"]["solvent_amount"] == [{"step": 1, "value": "1.0 mL"}]
    assert template["conditions"]["temperature"] == [{"step": 1, "value": "50 °C"}]
    assert template["conditions"]["time"] == [{"step": 1, "value": "1 h"}]


def test_gp_template_keeps_catalyst_complex_amount_separate_from_free_ligand():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    payload = _gp_c_payload()
    payload["catalysts"] = [
        {"name": "NiCl2[(R,R)-QuinoxP*]", "amount": "10 mol%", "step": 2}
    ]
    payload["ligands"] = [{"name": "(R,R)-QuinoxP*", "step": 2}]
    template = extractor.validate_gp_template(
        payload,
        gp_label="GeneralProcedureC",
        gp_text="preformed-complex fixture",
    )
    assert template["catalysts"][0]["name"] == "NiCl2[(R,R)-QuinoxP*]"
    assert template["catalysts"][0]["amount"] == "10 mol%"
    assert template["ligands"] == [
        {"name": "(R,R)-QuinoxP*", "amount": None, "step": 2}
    ]


def test_gp_template_normalizes_missing_amount_and_merges_duplicate_ligand_amounts():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    payload = _gp_c_payload()
    payload["ligands"] = [
        {"name": "L10", "amount": None, "step": 2},
        {"name": "L10", "amount": "9.4 mg", "step": 2},
        {"name": "L10", "amount": "15 mol%", "step": 2},
        {"name": "L10", "amount": "15 mol%", "step": 2},
    ]

    template = extractor.validate_gp_template(
        payload,
        gp_label="GeneralProcedureC",
        gp_text="duplicate ligand fixture",
    )

    assert template["ligands"] == [
        {"name": "L10", "amount": "9.4 mg; 15 mol%", "step": 2}
    ]


def test_gp_reaction_context_copies_ligand_amount_and_step():
    context = SIExtractor._reaction_gp_context(
        "GeneralProcedureC",
        {
            "gp_id": "GeneralProcedureC",
            "reaction_type": "test",
            "step_count": 2,
            "ligands": [
                {
                    "name": "L10",
                    "amount": "9.4 mg, 0.015 mmol, 15 mol%",
                    "step": 2,
                }
            ],
        },
    )

    assert context["ligands"] == [
        {
            "name": "L10",
            "amount": "9.4 mg, 0.015 mmol, 15 mol%",
            "step": 2,
        }
    ]
