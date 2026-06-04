import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from langgraph_workflow import chemeagle_adapter as adapter  # noqa: E402


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(response))
                )
            ]
        )


class FakeOpenAI:
    completions = None

    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))
        FakeOpenAI.completions = self.chat.completions


def install_fake_openai(monkeypatch, responses):
    def factory(**_kwargs):
        return FakeOpenAI(responses)

    monkeypatch.setattr(adapter, "OpenAI", factory)


def reagent_condition(text):
    return {"role": "reagent", "text": text, "smiles": "None"}


def solvent_condition(text="H2O"):
    return {"role": "solvent", "text": text, "smiles": "O"}


def payload_with_two_pending_reactions():
    return {
        "pdf_name": "paper.pdf",
        "image_name": "image_1.png",
        "reactions": [
            {
                "reaction_id": "r1",
                "reactants": [{"label": "1", "smiles": "CC"}],
                "products": [{"label": "2", "smiles": "CCC"}],
                "conditions": [reagent_condition("CuI (5 mol%)"), solvent_condition()],
            },
            {
                "reaction_id": "r2",
                "reactants": [{"label": "3", "smiles": "CO"}],
                "products": [{"label": "4", "smiles": "CCO"}],
                "conditions": [reagent_condition("Et3N (1 equiv)"), solvent_condition()],
            },
        ],
    }


def batch_response_for(*reaction_indexes):
    return {
        "reactions": [
            {
                "reaction_index": index,
                "reaction_id": f"r{index + 1}",
                "conditions": [
                    {
                        "condition_index": 0,
                        "original_role": "reagent",
                        "text": "condition",
                        "refined_role": "catalyst" if index == 0 else "base",
                        "reason": "classified from amount and chemistry context",
                    }
                ],
            }
            for index in reaction_indexes
        ]
    }


def fallback_response(role="base"):
    return {
        "conditions": [
            {
                "condition_index": 0,
                "original_role": "reagent",
                "text": "condition",
                "refined_role": role,
                "reason": "fallback classification",
            }
        ]
    }


def test_role_refinement_batches_multiple_reactions_in_one_llm_call(monkeypatch):
    install_fake_openai(monkeypatch, [batch_response_for(0, 1)])

    refined, stats = adapter.refine_chemeagle_condition_roles(
        payload_with_two_pending_reactions(),
        model="gpt-4o",
        api_key="test",
        base_url="http://example.test/v1",
    )

    calls = FakeOpenAI.completions.calls
    assert len(calls) == 1
    assert "multiple reactions from one image" in calls[0]["messages"][1]["content"]
    assert stats["role_refinement_llm_batch_requests"] == 1
    assert stats["role_refinement_llm_fallback_requests"] == 0
    assert stats["conditions_llm_refined"] == 2
    assert refined["reactions"][0]["conditions"][0]["refined_role"] == "catalyst"
    assert refined["reactions"][1]["conditions"][0]["refined_role"] == "base"
    assert "role_confidence" not in refined["reactions"][0]["conditions"][0]
    assert refined["reactions"][0]["conditions"][1]["role_source"] == "original_chemeagle_role"
    assert "role_confidence" not in refined["reactions"][0]["conditions"][1]


def test_role_refinement_skips_llm_when_all_roles_are_direct(monkeypatch):
    install_fake_openai(monkeypatch, [])
    payload = {
        "image_name": "image_1.png",
        "reactions": [
            {
                "reaction_id": "r1",
                "conditions": [
                    solvent_condition(),
                    {"role": "temperature", "text": "80 C"},
                    {"role": "time", "text": "1 h"},
                ],
            }
        ],
    }

    refined, stats = adapter.refine_chemeagle_condition_roles(
        payload,
        model="gpt-4o",
        api_key="test",
        base_url="http://example.test/v1",
    )

    assert FakeOpenAI.completions is not None
    assert FakeOpenAI.completions.calls == []
    assert stats["role_refinement_llm_batch_requests"] == 0
    assert stats["conditions_direct"] == 3
    assert all(condition["refined_role"] for condition in refined["reactions"][0]["conditions"])
    assert all("role_confidence" not in condition for condition in refined["reactions"][0]["conditions"])


def test_role_refinement_falls_back_only_for_missing_batch_reaction(monkeypatch):
    install_fake_openai(monkeypatch, [batch_response_for(0), fallback_response("base")])

    refined, stats = adapter.refine_chemeagle_condition_roles(
        payload_with_two_pending_reactions(),
        model="gpt-4o",
        api_key="test",
        base_url="http://example.test/v1",
    )

    calls = FakeOpenAI.completions.calls
    assert len(calls) == 2
    assert "multiple reactions from one image" in calls[0]["messages"][1]["content"]
    assert "one image-extracted reaction" in calls[1]["messages"][1]["content"]
    assert stats["role_refinement_llm_batch_requests"] == 1
    assert stats["role_refinement_llm_fallback_requests"] == 1
    assert stats["role_refinement_reactions_fallback"] == 1
    assert stats["conditions_llm_refined"] == 2
    assert refined["reactions"][0]["conditions"][0]["refined_role"] == "catalyst"
    assert refined["reactions"][1]["conditions"][0]["refined_role"] == "base"


def test_role_refinement_falls_back_all_reactions_when_batch_json_fails(monkeypatch):
    install_fake_openai(
        monkeypatch,
        [
            RuntimeError("invalid json"),
            fallback_response("catalyst"),
            fallback_response("base"),
        ],
    )

    refined, stats = adapter.refine_chemeagle_condition_roles(
        payload_with_two_pending_reactions(),
        model="gpt-4o",
        api_key="test",
        base_url="http://example.test/v1",
    )

    assert len(FakeOpenAI.completions.calls) == 3
    assert stats["role_refinement_llm_batch_requests"] == 1
    assert stats["role_refinement_llm_fallback_requests"] == 2
    assert stats["role_refinement_reactions_fallback"] == 2
    assert stats["conditions_llm_refined"] == 2
    assert stats["conditions_fallback"] == 0
    assert stats["role_refinement_batch_errors"]
    assert refined["reactions"][0]["conditions"][0]["refined_role"] == "catalyst"
    assert refined["reactions"][1]["conditions"][0]["refined_role"] == "base"


def normalize_single_condition(condition):
    payload = {
        "pdf_name": "paper.pdf",
        "image_name": "image_1.png",
        "reactions": [
            {
                "reaction_id": "r1",
                "reactants": [{"label": "1", "smiles": "CC"}],
                "products": [{"label": "2", "smiles": "CCC"}],
                "conditions": [condition],
            }
        ],
    }
    normalized = adapter.normalize_chemeagle_payload(
        payload,
        pdf_path=Path("paper.pdf"),
        artifact_stem="paper",
        source_paper="paper",
    )
    return normalized["reactions"][0]


def test_normalization_uses_refined_role_without_confidence():
    reaction = normalize_single_condition(
        {
            "role": "reagent",
            "text": "PdCl2 (0.3 mol-%)",
            "refined_role": "catalyst",
            "role_confidence": "low",
            "role_source": "llm_role_refinement",
            "role_reason": "Pd catalyst",
        }
    )

    assert reaction["catalysts"][0]["name"] == "PdCl2 (0.3 mol-%)"
    assert reaction["catalysts"][0]["refined_role"] == "catalyst"
    assert "role_confidence" not in reaction["catalysts"][0]
    assert reaction["reagents"] == []


def test_normalization_falls_back_to_original_role_when_refined_unknown():
    reaction = normalize_single_condition(
        {
            "role": "reagent",
            "text": "ambiguous reagent",
            "refined_role": "unknown",
            "role_confidence": "high",
            "role_source": "llm_role_refinement",
            "role_reason": "not enough evidence",
        }
    )

    assert reaction["reagents"][0]["name"] == "ambiguous reagent"
    assert reaction["reagents"][0]["refined_role"] == "unknown"
    assert "role_confidence" not in reaction["reagents"][0]
    assert reaction["catalysts"] == []


def test_normalization_maps_ligand_and_base_refined_roles():
    ligand_reaction = normalize_single_condition(
        {"role": "reagent", "text": "L1", "refined_role": "ligand"}
    )
    base_reaction = normalize_single_condition(
        {"role": "reagent", "text": "Et3N", "refined_role": "base"}
    )
    solvent_reaction = normalize_single_condition(
        {"role": "reagent", "text": "MeCN", "refined_role": "solvent"}
    )

    assert ligand_reaction["catalysts"][0]["name"] == "L1"
    assert base_reaction["additives"][0]["name"] == "Et3N"
    assert solvent_reaction["conditions"]["solvent"] == "MeCN"
