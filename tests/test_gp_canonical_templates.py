import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from batch_si_extractor import SIExtractor  # noqa: E402
from cross_modal_kg import build_text_reaction_triples, image_role_to_edge  # noqa: E402
from langgraph_workflow.benchmark_utils import _public_condition_option  # noqa: E402


GP_TEXT = (
    "General Procedure C. A reaction vial was charged with "
    "NiCl2[(R,R)-QuinoxP*] (10 mol%), the substrate amide "
    "(0.30 mmol, 1.0 equiv), Na2CO3 (3.0 equiv), Mn (3.0 equiv), "
    "DMF (1.0 mL) in a glove box. The sealed vial was stirred at "
    "20 °C for 1 h and then at 60 °C for 20 h."
)


def make_extractor():
    extractor = object.__new__(SIExtractor)
    extractor.extract_model = "test-model"
    extractor.enable_stage2_audit = False
    return extractor


def single_gp_payload():
    return {
        "reaction_type": "enantioselective Mizoroki-Heck cyclization",
        "substrates": [
            {
                "name": "substrate amide",
                "amount": "0.30 mmol, 1.0 equiv",
                "is_generic_class": True,
            }
        ],
        "catalysts": [
            {"name": "NiCl2[(R,R)-QuinoxP*]", "amount": "10 mol%"}
        ],
        "ligands": [{"name": "(R,R)-QuinoxP*"}],
        "other_components": [
            {"name": "Na2CO3", "amount": "3.0 equiv"},
            {"name": "Mn", "amount": "3.0 equiv"},
            {"name": "(R,R)-QuinoxP*", "amount": None},
        ],
        "conditions": {
            "solvent": "DMF",
            "solvent_amount": "1.0 mL",
            "concentration": None,
            "temperature": "20 °C then 60 °C",
            "time": "1 h then 20 h",
            "atmosphere": "glove box; sealed vial after removal",
            "light_source": None,
            "wavelength": None,
        },
        "procedure_details": [],
        "evidence": {"ligands": [{"index": 0, "text": "(R,R)-QuinoxP*"}]},
    }


def validate_single(extractor):
    return extractor.validate_gp_template(
        single_gp_payload(),
        gp_label="GeneralProcedureC",
        gp_text=GP_TEXT,
        source_pages=[21],
    )


def test_single_step_template_keeps_complex_and_name_only_ligand():
    template = validate_single(make_extractor())

    assert "step_count" not in template
    assert template["catalysts"] == [
        {"name": "NiCl2[(R,R)-QuinoxP*]", "amount": "10 mol%"}
    ]
    assert template["ligands"] == [{"name": "(R,R)-QuinoxP*"}]
    assert {item["name"] for item in template["other_components"]} == {"Na2CO3", "Mn"}
    assert template["raw_text_sha256"] == hashlib.sha256(GP_TEXT.encode("utf-8")).hexdigest()
    assert template["source_pages"] == [21]


def test_ligand_amount_is_rejected():
    payload = single_gp_payload()
    payload["ligands"] = [{"name": "BINAP", "amount": "5 mol%"}]

    with pytest.raises(ValueError, match="unsupported fields"):
        make_extractor().validate_gp_template(
            payload, gp_label="GeneralProcedureA", gp_text="BINAP", source_pages=[]
        )


def test_multistep_ligand_requires_step_and_keeps_only_name_and_step():
    payload = single_gp_payload()
    payload["step_count"] = 2
    for field in ("substrates", "catalysts", "other_components"):
        for item in payload[field]:
            item["step"] = 1
    payload["ligands"] = [{"name": "BINAP"}]
    payload["conditions"] = {
        key: ([] if value is None else [{"step": 1, "value": value}])
        for key, value in payload["conditions"].items()
    }

    with pytest.raises(ValueError, match="invalid step"):
        make_extractor().validate_gp_template(
            payload, gp_label="GeneralProcedureA", gp_text="two steps", source_pages=[]
        )

    payload["ligands"] = [{"name": "BINAP", "step": 2}]
    template = make_extractor().validate_gp_template(
        payload, gp_label="GeneralProcedureA", gp_text="two steps", source_pages=[]
    )
    assert template["step_count"] == 2
    assert template["ligands"] == [{"name": "BINAP", "step": 2}]


def test_gp_template_is_materialized_identically_across_chunks():
    extractor = make_extractor()
    template = validate_single(extractor)
    reactions = []
    for entry in range(1, 12):
        reactions.append(
            {
                "id": f"GeneralProcedureC-Entry{entry}",
                "reaction_type": "Mizoroki-Heck cyclization",
                "substrates": [{"name": f"amide {entry}"}],
                "products": [{"name": f"oxindole {entry}"}],
                "catalysts": [{"name": "inconsistent catalyst text"}],
                "additives": [{"name": "Na carbonate"}],
                "conditions": {"temperature": f"variant {entry}"},
                "targets": {"yield": "70%", "ee": "90%", "er": None},
            }
        )

    materialized = extractor.apply_gp_templates(
        reactions, {"GeneralProcedureC": template}
    )
    signatures = {
        json.dumps(
            {
                "catalysts": item["catalysts"],
                "ligands": item["ligands"],
                "other_components": item["other_components"],
                "conditions": item["conditions"],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for item in materialized
    }

    assert len(signatures) == 1
    assert all(item["conditions"]["atmosphere"].startswith("glove box") for item in materialized)
    assert all("additives" not in item and "reagents" not in item for item in materialized)


def test_explicit_entry_condition_override_wins():
    extractor = make_extractor()
    template = validate_single(extractor)
    reaction = {
        "id": "GeneralProcedureC-Entry1",
        "substrates": [{"name": "amide"}],
        "products": [{"name": "product"}],
        "_entry_overrides": {"conditions": {"atmosphere": "nitrogen"}},
    }

    result = extractor.apply_gp_templates(
        [reaction], {"GeneralProcedureC": template}
    )[0]
    assert result["conditions"]["atmosphere"] == "nitrogen"
    assert result["conditions"]["temperature"] == "20 °C then 60 °C"


class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_gp_structuring_calls_llm_once_per_gp():
    extractor = make_extractor()
    completions = FakeCompletions(
        [json.dumps(single_gp_payload()), json.dumps(single_gp_payload())]
    )
    extractor.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    templates = extractor.build_gp_templates(
        {"GeneralProcedureA": GP_TEXT, "GeneralProcedureC": GP_TEXT}
    )

    assert len(completions.calls) == 2
    assert set(templates) == {"GeneralProcedureA", "GeneralProcedureC"}
    assert all(item["status"] == "valid" for item in templates.values())


def test_reaction_prompt_injects_template_but_not_gp_raw_text():
    extractor = make_extractor()
    template = validate_single(extractor)
    completions = FakeCompletions(["[]"])
    extractor.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    extractor.stage2_extract_with_meta(
        "--- Page 24 ---\nOxindole 3: Prepared according to General Procedure C. 71% yield.",
        "chunk",
        gp_texts={"GeneralProcedureC": "General Procedure C SECRET_RAW_GP_TEXT"},
        gp_templates={"GeneralProcedureC": template},
    )

    prompt = completions.calls[0]["messages"][1]["content"]
    assert "SECRET_RAW_GP_TEXT" not in prompt
    assert template["raw_text_sha256"] in prompt
    assert '"ligands"' in prompt


def test_q2_condition_signature_includes_ligand_name_but_not_amount():
    base = {
        "conditions": {"temperature": "60 °C"},
        "catalysts": [{"name": "Pd2(dba)3", "amount": "2 mol%"}],
        "ligands": [{"name": "BINAP", "amount": "ignored"}],
        "other_components": [{"name": "Na2CO3", "amount": "2 equiv"}],
    }
    signature = _public_condition_option(base)

    assert signature["ligands"] == [{"name": "BINAP"}]
    assert signature["catalysts"][0]["amount"] == "2 mol%"
    assert signature["other_components"][0]["amount"] == "2 equiv"


def test_kg_uses_distinct_ligand_and_other_component_relations(tmp_path):
    path = tmp_path / "reactions.json"
    path.write_text(
        json.dumps(
            {
                "source": "paper.pdf",
                "reactions": [
                    {
                        "id": "r1",
                        "substrates": [{"name": "substrate"}],
                        "products": [{"name": "product"}],
                        "catalysts": [{"name": "Pd2(dba)3", "amount": "2 mol%"}],
                        "ligands": [{"name": "BINAP"}],
                        "other_components": [{"name": "Na2CO3", "amount": "2 equiv"}],
                        "conditions": {},
                        "targets": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    triples = build_text_reaction_triples([path])
    by_relation = {row["relationship"]: row for row in triples}
    assert by_relation["USES_LIGAND"]["y_name"] == "BINAP"
    assert by_relation["USES_OTHER_COMPONENT"]["other_component_amount"] == "2 equiv"
    assert image_role_to_edge("ligand") == ("USES_LIGAND", "Ligand")
    assert image_role_to_edge("other_component") == (
        "USES_OTHER_COMPONENT",
        "OtherComponent",
    )
