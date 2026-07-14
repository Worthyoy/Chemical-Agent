import copy

import pytest

from batch_si_extractor import GP_TEMPLATE_PROMPT_VERSION, SIExtractor


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
        "ligands": [{"name": "L6", "step": 2}],
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

    assert GP_TEMPLATE_PROMPT_VERSION == "gp_template_prompt_v7"
    assert "name stores chemical identity only" in prompt
    assert "amount stores every reported numerical quantity" in prompt
    assert "Move any parenthetical text that reports a numerical quantity or loading into amount" in prompt
    assert '"name":"reported catalyst","amount":"reported mass, reported mmol, reported mol%"' in prompt
    assert '"name":"reported catalyst (reported mass, reported mol%)","amount":null' in prompt
    assert "preformed metal-ligand complex remains a complete catalyst name" in prompt
    assert "ligands without an amount" in prompt


def test_gp_template_prompt_counts_main_lineage_and_not_parallel_catalyst_premixing():
    prompt = SIExtractor.GP_TEMPLATE_PROMPT

    assert "main substrate-to-final-product transformation lineage" in prompt
    assert "Parallel preparation, premixing, activation, or aging" in prompt
    assert "does not create step 3" in prompt
    assert "A -> intermediate X -> final product" in prompt
    assert "General Procedure C" in prompt
    assert "Never use the GP id" in prompt
    assert "reaction boundary controls role assignment" in prompt


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
    assert template["ligands"] == [{"name": "L6", "step": 2}]
    assert template["conditions"]["solvent_amount"] == [{"step": 1, "value": "1.0 mL"}]
    assert template["conditions"]["temperature"] == [{"step": 1, "value": "50 °C"}]
    assert template["conditions"]["time"] == [{"step": 1, "value": "1 h"}]


def test_gp_template_keeps_chemical_identity_parentheses_and_rejects_ligand_amount():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    payload = _gp_c_payload()
    payload["catalysts"] = [
        {"name": "NiCl2[(R,R)-QuinoxP*]", "amount": "10 mol%", "step": 2}
    ]
    template = extractor.validate_gp_template(
        payload,
        gp_label="GeneralProcedureC",
        gp_text="preformed-complex fixture",
    )
    assert template["catalysts"][0]["name"] == "NiCl2[(R,R)-QuinoxP*]"
    assert template["catalysts"][0]["amount"] == "10 mol%"

    invalid = copy.deepcopy(payload)
    invalid["ligands"] = [{"name": "(R,R)-QuinoxP*", "amount": "10 mol%", "step": 2}]
    with pytest.raises(ValueError, match="ligand item has unsupported fields"):
        extractor.validate_gp_template(
            invalid,
            gp_label="GeneralProcedureC",
            gp_text="invalid ligand fixture",
        )
