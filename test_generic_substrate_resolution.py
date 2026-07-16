import json
from types import SimpleNamespace

import pytest

from batch_si_extractor import GENERIC_SUBSTRATE_RESOLUTION_SCHEMA_VERSION, SIExtractor


class _FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = 0

    def create(self, **_kwargs):
        index = min(self.calls, len(self.contents) - 1)
        self.calls += 1
        message = SimpleNamespace(content=self.contents[index])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _DynamicCompletions:
    def __init__(self, *, fail_multi=False):
        self.calls = 0
        self.fail_multi = fail_multi

    def create(self, **kwargs):
        self.calls += 1
        user = kwargs["messages"][1]["content"].split("\n\nPrevious response", 1)[0]
        payload = json.loads(user)
        content = "not json" if self.fail_multi and len(payload["reactions"]) > 1 else json.dumps(_specific_response(payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


def _extractor_with_contents(*contents, batch_size=10):
    extractor = SIExtractor(
        api_key="test", enable_stage2_audit=False,
        generic_resolution_batch_size=batch_size,
    )
    extractor.client = _FakeClient(_FakeCompletions(contents))
    return extractor


def _reaction(reaction_id, substrate="aniline", products=None):
    return {
        "id": reaction_id,
        "reaction_type": "test",
        "substrates": [{"name": substrate, "amount": "0.8 equiv"}],
        "products": products or [{"name": "reported concrete product"}],
        "targets": {"yield": "81%", "ee": None, "er": None},
    }


def _specific_response(payload):
    return {
        "schema_version": GENERIC_SUBSTRATE_RESOLUTION_SCHEMA_VERSION,
        "batch_id": payload["batch_id"],
        "reaction_assessments": [
            {
                "reaction_id": reaction["reaction_id"],
                "substrate_assessments": [
                    {
                        "substrate_index": substrate["index"],
                        "identity_status": "specific",
                        "classification_confidence": "high",
                        "can_resolve": False,
                        "resolved_name": None,
                        "resolution_confidence": None,
                        "evidence_product_index": None,
                        "reason": "The name is a concrete identity.",
                    }
                    for substrate in reaction["substrates"]
                ],
            }
            for reaction in payload["reactions"]
        ],
    }


def _resolution_response(batch_id, reaction_id, resolved_name, product_index=0):
    return {
        "schema_version": GENERIC_SUBSTRATE_RESOLUTION_SCHEMA_VERSION,
        "batch_id": batch_id,
        "reaction_assessments": [{
            "reaction_id": reaction_id,
            "substrate_assessments": [{
                "substrate_index": 0,
                "identity_status": "generic",
                "classification_confidence": "high",
                "can_resolve": True,
                "resolved_name": resolved_name,
                "resolution_confidence": "high",
                "evidence_product_index": product_index,
                "reason": "A unique complete starting-material fragment is retained.",
            }],
        }],
    }


def test_prompt_combines_classification_and_resolution_without_generic_vocabulary():
    prompt = SIExtractor.GENERIC_SUBSTRATE_RESOLUTION_PROMPT
    for phrase in (
        "specific, generic, label_only, or ambiguous",
        "Do not use or invent a fixed vocabulary",
        "complete starting-material identity",
        "not the minimal scaffold",
        "evidence_product_index",
        "Every supplied substrate_index must appear exactly once",
    ):
        assert phrase in prompt
    assert prompt.count("generic substrate:") == 2
    assert "2-bromoaniline" in prompt
    assert "2-bromo-N-(methoxymethyl)aniline" in prompt


def test_prompt_requires_role_preserving_resolution_with_canonical_gp_context():
    prompt = SIExtractor.GENERIC_SUBSTRATE_RESOLUTION_PROMPT

    assert "Resolution is role-preserving" in prompt
    assert "same externally supplied starting material" in prompt
    assert "Never resolve a substrate into an intermediate" in prompt
    assert "canonical GP role context as authoritative" in prompt
    assert "return can_resolve=false" in prompt
    assert "symbol_evidence" in prompt


def test_batch_parser_accepts_object_and_fenced_object():
    payload = {"schema_version": GENERIC_SUBSTRATE_RESOLUTION_SCHEMA_VERSION, "batch_id": "b", "reactions": []}
    response = _specific_response(payload)
    extractor = _extractor_with_contents(json.dumps(response), f"```json\n{json.dumps(response)}\n```")
    assert extractor._call_generic_substrate_resolution_llm(payload) == response
    assert extractor._call_generic_substrate_resolution_llm(payload) == response


def test_batch_parser_rejects_json_list():
    extractor = _extractor_with_contents("[]")
    with pytest.raises(ValueError, match="must be a JSON object"):
        extractor._call_generic_substrate_resolution_llm({"batch_id": "b", "reactions": []})


def test_valid_resolution_writes_back_and_survives_validator():
    product = "(Z)-N-(2-bromophenyl)-N,2-dimethylbut-2-enamide"
    response = _resolution_response("paper_batch_0001", "r1", "2-bromoaniline")
    extractor = _extractor_with_contents(json.dumps(response))
    resolved = extractor.resolve_generic_substrates_from_products([_reaction("r1", products=[{"name": product}])])
    substrate = extractor.validate_substrate_name_resolutions(resolved)[0]["substrates"][0]
    assert substrate["name"] == "2-bromoaniline"
    assert substrate["original_name"] == "aniline"
    assert substrate["resolution_method"] == "product_name_to_substrate_mapping"
    assert substrate["resolution_evidence"]["product_index"] == 0
    assert not any(key.startswith("_resolution_") for key in substrate)


def test_resolution_batch_payload_includes_compact_gp_roles_and_symbol_evidence_once():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    captured = []

    def _fake_batch(payload, _cache, _logs, _stats):
        captured.append(payload)
        return _specific_response(payload)

    extractor._run_generic_resolution_batch = _fake_batch
    reaction = _reaction("r1", substrate="generic starting material")
    reaction["_gp_source"] = "GeneralProcedureA"
    reaction["step_count"] = 2
    reaction["substrates"][0].update({"symbol": "S1", "step": 1})
    reaction["products"][0]["step"] = 2
    reaction["intermediates"] = [{
        "name": "internal intermediate X", "produced_in_step": 1, "consumed_in_step": 2,
    }]
    reaction["catalysts"] = [{"name": "catalyst C", "amount": "10 mol%", "step": 2}]
    reaction["ligands"] = [{"name": "ligand L", "step": 2}]
    reaction["other_components"] = [{"name": "reagent R", "amount": "2 equiv", "step": 2}]
    gp_templates = {
        "GeneralProcedureA": {
            "status": "valid",
            "gp_id": "GeneralProcedureA",
            "reaction_type": "test transformation",
            "step_count": 2,
            "substrates": [{"name": "generic starting material", "step": 1}],
            "intermediates": [{
                "name": "internal intermediate X", "produced_in_step": 1,
                "consumed_in_step": 2,
            }],
            "catalysts": [{"name": "catalyst C", "amount": "10 mol%", "step": 2}],
            "ligands": [{"name": "ligand L", "step": 2}],
            "other_components": [{"name": "reagent R", "amount": "2 equiv", "step": 2}],
        }
    }

    extractor.resolve_generic_substrates_from_products(
        [reaction], gp_templates=gp_templates, symbol_registry={"S1": "specific substrate S1"}
    )

    assert len(captured) == 1
    payload = captured[0]
    assert list(payload["gp_templates"]) == ["GeneralProcedureA"]
    assert payload["gp_templates"]["GeneralProcedureA"]["step_count"] == 2
    assert payload["gp_templates"]["GeneralProcedureA"]["intermediates"][0]["name"] == "internal intermediate X"
    assert payload["symbol_evidence"] == {"S1": "specific substrate S1"}
    candidate = payload["reactions"][0]
    assert candidate["catalysts"] == [{"name": "catalyst C", "symbol": None, "step": 2}]
    assert candidate["ligands"] == [{"name": "ligand L", "symbol": None, "step": 2}]
    assert candidate["other_components"] == [{"name": "reagent R", "symbol": None, "step": 2}]
    assert "amount" not in candidate["substrates"][0]
    assert "amount" not in candidate["catalysts"][0]


def test_validator_rejects_resolution_that_matches_an_intermediate_role():
    product = "mapped product"
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = _reaction("r1", substrate="generic starting material", products=[{"name": product}])
    reaction["intermediates"] = [{
        "name": "internal intermediate X", "produced_in_step": 1, "consumed_in_step": 2,
    }]
    reaction["substrates"][0].update({
        "name": "internal intermediate X",
        "original_name": "generic starting material",
        "resolution_source": "product_name",
        "resolution_method": "product_name_to_substrate_mapping",
        "resolution_confidence": "high",
        "resolution_evidence": {"product_index": 0, "product_name": product},
        "_resolution_identity_status": "generic",
        "_resolution_classification_confidence": "high",
        "_resolution_input_name": "generic starting material",
        "_resolution_substrate_index": 0,
    })

    validated = extractor.validate_substrate_name_resolutions([reaction])

    assert validated[0]["substrates"][0]["name"] == "generic starting material"
    assert extractor.last_substrate_name_resolution_reviews[-1]["reason"].startswith(
        "resolved_name_matches_non_substrate_role:intermediates"
    )


def test_complete_heteroatom_substituted_name_is_preserved():
    product = "(Z)-N-(2-bromophenyl)-N-(methoxymethyl)-2-methylbut-2-enamide"
    response = _resolution_response("paper_batch_0001", "r10", "2-bromo-N-(methoxymethyl)aniline")
    extractor = _extractor_with_contents(json.dumps(response))
    result = extractor.validate_substrate_name_resolutions(
        extractor.resolve_generic_substrates_from_products([_reaction("r10", products=[{"name": product}])])
    )
    assert result[0]["substrates"][0]["name"] == "2-bromo-N-(methoxymethyl)aniline"


def test_out_of_vocabulary_generic_name_is_sent_to_llm_and_resolved():
    response = _resolution_response("paper_batch_0001", "r1", "specific radical precursor name")
    extractor = _extractor_with_contents(json.dumps(response))
    result = extractor.validate_substrate_name_resolutions(
        extractor.resolve_generic_substrates_from_products([_reaction("r1", substrate="radical precursor")])
    )
    assert extractor.client.chat.completions.calls == 1
    assert result[0]["substrates"][0]["name"] == "specific radical precursor name"


def test_label_only_product_is_still_assessed_without_forced_resolution():
    response = {
        "schema_version": GENERIC_SUBSTRATE_RESOLUTION_SCHEMA_VERSION,
        "batch_id": "paper_batch_0001",
        "reaction_assessments": [{
            "reaction_id": "r1",
            "substrate_assessments": [{
                "substrate_index": 0,
                "identity_status": "generic",
                "classification_confidence": "high",
                "can_resolve": False,
                "resolved_name": None,
                "resolution_confidence": None,
                "evidence_product_index": None,
                "reason": "The product is label-only.",
            }],
        }],
    }
    extractor = _extractor_with_contents(json.dumps(response))
    result = extractor.resolve_generic_substrates_from_products(
        [_reaction("r1", substrate="substrate amide", products=[{"name": "Oxindole 2"}])]
    )
    assert extractor.client.chat.completions.calls == 1
    assert result[0]["substrates"][0]["name"] == "substrate amide"
    assert extractor.last_generic_substrate_resolution_reviews[-1]["reason"] == "generic_substrate_not_resolved"


def test_multi_product_resolution_uses_explicit_product_index():
    response = _resolution_response("paper_batch_0001", "r1", "2-bromoaniline", product_index=1)
    extractor = _extractor_with_contents(json.dumps(response))
    products = [{"name": "unrelated product"}, {"name": "mapped concrete product"}]
    result = extractor.validate_substrate_name_resolutions(
        extractor.resolve_generic_substrates_from_products([_reaction("r1", products=products)])
    )
    evidence = result[0]["substrates"][0]["resolution_evidence"]
    assert evidence["product_index"] == 1
    assert evidence["product_name"] == "mapped concrete product"


def test_missing_confidence_is_not_defaulted_to_high():
    product = "mapped product"
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = _reaction("r1", products=[{"name": product}])
    reaction["substrates"][0].update({
        "name": "resolved name", "original_name": "radical precursor",
        "resolution_source": "product_name",
        "resolution_method": "product_name_to_substrate_mapping",
        "resolution_evidence": {"product_index": 0, "product_name": product},
        "_resolution_identity_status": "generic",
        "_resolution_classification_confidence": "high",
        "_resolution_input_name": "radical precursor",
        "_resolution_substrate_index": 0,
    })
    validated = extractor.validate_substrate_name_resolutions([reaction])
    assert validated[0]["substrates"][0]["name"] == "radical precursor"
    assert extractor.last_substrate_name_resolution_reviews[-1]["reason"] == "resolution_confidence_is_not_high"


def test_twenty_three_reactions_use_three_batch_calls():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False, generic_resolution_batch_size=10)
    completions = _DynamicCompletions()
    extractor.client = _FakeClient(completions)
    result = extractor.resolve_generic_substrates_from_products(
        [_reaction(f"r{i}", substrate=f"specific substrate {i}") for i in range(23)]
    )
    assert len(result) == 23
    assert completions.calls == 3
    assert extractor.last_generic_substrate_resolution_stats["substrates_assessed_by_llm"] == 23


def test_failed_multi_batch_is_bisected_without_losing_reactions():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False, generic_resolution_batch_size=10)
    completions = _DynamicCompletions(fail_multi=True)
    extractor.client = _FakeClient(completions)
    result = extractor.resolve_generic_substrates_from_products([_reaction("r1"), _reaction("r2")])
    assert len(result) == 2
    assert completions.calls == 4
    assert extractor.last_generic_substrate_resolution_stats["resolution_batch_retries"] == 1


def test_resolution_batch_cache_avoids_llm_call():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    completions = _DynamicCompletions()
    extractor.client = _FakeClient(completions)
    candidate = extractor._generic_substrate_resolution_candidate(
        _reaction("r1", substrate="specific substrate")
    )
    payload = {
        "schema_version": GENERIC_SUBSTRATE_RESOLUTION_SCHEMA_VERSION,
        "batch_id": "p_batch_0001",
        "reactions": [candidate],
    }
    response = _specific_response(payload)
    cache = {extractor._generic_resolution_cache_key(payload): response}
    stats = {"resolution_cache_hits": 0, "llm_calls": 0, "resolution_batch_calls": 0,
             "resolution_batch_retries": 0, "resolution_batch_failures": 0}
    result = extractor._run_generic_resolution_batch(payload, cache, [], stats)
    assert result == response
    assert completions.calls == 0
    assert stats["resolution_cache_hits"] == 1
