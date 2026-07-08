import json
from types import SimpleNamespace

import pytest

from batch_si_extractor import SIExtractor


class _FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    def create(self, **_kwargs):
        index = min(self.calls, len(self.contents) - 1)
        self.calls += 1
        message = SimpleNamespace(content=self.contents[index])
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, *contents):
        self.chat = SimpleNamespace(completions=_FakeCompletions(contents))


def _extractor_with_response(content):
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    extractor.client = _FakeClient(content)
    return extractor


def test_generic_substrate_resolution_prompt_is_generic_and_has_two_current_examples():
    prompt = SIExtractor.GENERIC_SUBSTRATE_RESOLUTION_PROMPT

    for phrase in (
        "complete starting-material identity",
        "not the minimal scaffold",
        "heteroatom substituents",
        "protecting groups",
        "one-to-one",
        "final product scaffold alone",
        "chemically meaningful fragment",
        "uniquely recoverable",
        "can_resolve=false",
    ):
        assert phrase in prompt

    assert prompt.count("generic substrate:") == 2
    assert prompt.count("resolved substrate:") == 2
    assert "2-bromoaniline" in prompt
    assert "2-bromo-N-(methoxymethyl)aniline" in prompt
    assert "6-bromo-2,2-difluorobenzo[d][1,3]dioxol-5-amine" not in prompt
    assert "2-bromo-4-methoxyaniline" not in prompt


def test_generic_substrate_resolution_parser_accepts_json_object():
    extractor = _extractor_with_response('{"reaction_id":"r1","resolutions":[]}')

    parsed = extractor._call_generic_substrate_resolution_llm({"reaction_id": "r1"})

    assert parsed == {"reaction_id": "r1", "resolutions": []}


def test_generic_substrate_resolution_parser_accepts_fenced_json_object():
    extractor = _extractor_with_response(
        '```json\n{"reaction_id":"r1","resolutions":[]}\n```'
    )

    parsed = extractor._call_generic_substrate_resolution_llm({"reaction_id": "r1"})

    assert parsed["reaction_id"] == "r1"
    assert parsed["resolutions"] == []


def test_generic_substrate_resolution_parser_rejects_json_list():
    extractor = _extractor_with_response('[{"reaction_id":"r1","resolutions":[]}]')

    with pytest.raises(ValueError, match="must be a JSON object"):
        extractor._call_generic_substrate_resolution_llm({"reaction_id": "r1"})


def test_generic_substrate_resolution_writes_back_valid_resolution():
    llm_payload = {
        "reaction_id": "GeneralProcedureA-Entry1",
        "resolutions": [
            {
                "substrate_index": 0,
                "can_resolve": True,
                "resolved_name": "2-bromoaniline",
                "original_name": "aniline",
                "resolution_source": "product_name",
                "resolution_method": "gp_product_to_substrate_mapping",
                "resolution_confidence": "high",
                "resolution_evidence": {
                    "product_name": "(Z)-N-(2-bromophenyl)-N,2-dimethylbut-2-enamide",
                    "reason": "The N-(2-bromophenyl) fragment maps to 2-bromoaniline.",
                },
            }
        ],
    }
    extractor = _extractor_with_response(json.dumps(llm_payload))
    reactions = [
        {
            "id": "GeneralProcedureA-Entry1",
            "reaction_type": "test",
            "substrates": [{"name": "aniline", "amount": "0.8 equiv"}],
            "products": [
                {"name": "(Z)-N-(2-bromophenyl)-N,2-dimethylbut-2-enamide"}
            ],
            "targets": {"yield": "81%", "ee": None, "er": None},
        }
    ]

    resolved = extractor.resolve_generic_substrates_from_products(reactions)
    validated = extractor.validate_substrate_name_resolutions(resolved)

    substrate = validated[0]["substrates"][0]
    assert substrate["name"] == "2-bromoaniline"
    assert substrate["original_name"] == "aniline"
    assert substrate["resolution_source"] == "product_name"
    assert extractor.last_substrate_name_resolution_stats["resolved"] == 1


def test_generic_substrate_resolution_writes_back_complete_heteroatom_substituted_aniline():
    product_name = "(Z)-N-(2-bromophenyl)-N-(methoxymethyl)-2-methylbut-2-enamide"
    llm_payload = {
        "reaction_id": "GeneralProcedureA-Entry10",
        "resolutions": [
            {
                "substrate_index": 0,
                "can_resolve": True,
                "resolved_name": "2-bromo-N-(methoxymethyl)aniline",
                "original_name": "aniline",
                "resolution_source": "product_name",
                "resolution_method": "gp_product_to_substrate_mapping",
                "resolution_confidence": "high",
                "resolution_evidence": {
                    "product_name": product_name,
                    "reason": "The product preserves the N-(2-bromophenyl)-N-(methoxymethyl) aniline identity.",
                },
            }
        ],
    }
    extractor = _extractor_with_response(json.dumps(llm_payload))
    reactions = [
        {
            "id": "GeneralProcedureA-Entry10",
            "reaction_type": "test",
            "substrates": [{"name": "aniline", "amount": "0.8 equiv"}],
            "products": [{"name": product_name}],
            "targets": {"yield": "46%", "ee": None, "er": None},
        }
    ]

    resolved = extractor.resolve_generic_substrates_from_products(reactions)
    validated = extractor.validate_substrate_name_resolutions(resolved)

    substrate = validated[0]["substrates"][0]
    assert substrate["name"] == "2-bromo-N-(methoxymethyl)aniline"
    assert substrate["original_name"] == "aniline"
    assert extractor.last_substrate_name_resolution_stats["resolved"] == 1


def test_generic_substrate_resolution_validator_rolls_back_invalid_product_copy():
    product_name = "(Z)-N-(2-bromophenyl)-N,2-dimethylbut-2-enamide"
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reactions = [
        {
            "id": "r1",
            "substrates": [
                {
                    "name": product_name,
                    "original_name": "aniline",
                    "resolution_source": "product_name",
                    "resolution_method": "gp_product_to_substrate_mapping",
                    "resolution_confidence": "high",
                    "resolution_evidence": {"product_name": product_name},
                }
            ],
            "products": [{"name": product_name}],
        }
    ]

    validated = extractor.validate_substrate_name_resolutions(reactions)

    assert validated[0]["substrates"][0]["name"] == "aniline"
    assert "resolution_source" not in validated[0]["substrates"][0]
    assert extractor.last_substrate_name_resolution_stats["reviewed"] == 1


def test_generic_substrate_resolution_validator_rolls_back_low_confidence():
    product_name = "(Z)-N-(2-bromophenyl)-N,2-dimethylbut-2-enamide"
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reactions = [
        {
            "id": "r1",
            "substrates": [
                {
                    "name": "2-bromoaniline",
                    "original_name": "aniline",
                    "resolution_source": "product_name",
                    "resolution_method": "gp_product_to_substrate_mapping",
                    "resolution_confidence": "low",
                    "resolution_evidence": {"product_name": product_name},
                }
            ],
            "products": [{"name": product_name}],
        }
    ]

    validated = extractor.validate_substrate_name_resolutions(reactions)

    assert validated[0]["substrates"][0]["name"] == "aniline"
    assert "resolution_source" not in validated[0]["substrates"][0]
    assert extractor.last_substrate_name_resolution_reviews[-1]["reason"] == "resolution_confidence_is_not_high"


def test_generic_substrate_resolution_skips_label_only_product_without_llm_call():
    extractor = _extractor_with_response('{"reaction_id":"unused","resolutions":[]}')
    reactions = [
        {
            "id": "GeneralProcedureC-Entry1",
            "reaction_type": "test",
            "substrates": [{"name": "substrate amide"}],
            "products": [{"name": "Oxindole 2"}],
            "targets": {"yield": "92%", "ee": "95%", "er": None},
        }
    ]

    resolved = extractor.resolve_generic_substrates_from_products(reactions)

    assert resolved[0]["substrates"][0]["name"] == "substrate amide"
    assert extractor.client.chat.completions.calls == 0
    assert extractor.last_generic_substrate_resolution_stats["llm_calls"] == 0
    assert extractor.last_generic_substrate_resolution_reviews[-1]["reason"] == "label_only_product_name"


def test_generic_substrate_resolution_multi_product_requires_explicit_evidence_product():
    product_a = "(Z)-N-(2-bromophenyl)-N,2-dimethylbut-2-enamide"
    product_b = "specific byproduct name"
    llm_payload = {
        "reaction_id": "r1",
        "resolutions": [
            {
                "substrate_index": 0,
                "can_resolve": True,
                "resolved_name": "2-bromoaniline",
                "original_name": "aniline",
                "resolution_source": "product_name",
                "resolution_method": "gp_product_to_substrate_mapping",
                "resolution_confidence": "high",
                "resolution_evidence": {
                    "product_name": product_a,
                    "reason": "The named product contains the one-to-one aniline fragment.",
                },
            }
        ],
    }
    extractor = _extractor_with_response(json.dumps(llm_payload))
    reactions = [
        {
            "id": "r1",
            "reaction_type": "test",
            "substrates": [{"name": "aniline"}],
            "products": [{"name": product_a}, {"name": product_b}],
        }
    ]

    resolved = extractor.resolve_generic_substrates_from_products(reactions)
    validated = extractor.validate_substrate_name_resolutions(resolved)

    assert validated[0]["substrates"][0]["name"] == "2-bromoaniline"
    assert extractor.last_substrate_name_resolution_stats["resolved"] == 1
