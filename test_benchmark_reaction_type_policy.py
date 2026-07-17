from pathlib import Path

import pytest

import langgraph_workflow.pipeline_agents as pipeline_agents
from langgraph_workflow.benchmark_utils import (
    generate_benchmark_packages_by_modality,
    generate_q1_benchmark_package,
    generate_q2_benchmark_package,
)


def _reaction(
    reaction_id,
    reaction_type,
    variable_name,
    solvent,
    yield_value,
):
    return {
        "id": reaction_id,
        "source_paper": "paper",
        "source_modality": "text",
        "reaction_type": reaction_type,
        "substrates": [
            {
                "name": variable_name,
                "scaffold": "benzaldehyde",
                "substituents": [],
            },
            {
                "name": "glycine",
                "scaffold": "glycine",
                "substituents": [],
            },
        ],
        "products": [
            {
                "name": "chiral amino alcohol",
                "scaffold": "amino alcohol",
            }
        ],
        "catalysts": [{"name": "catalyst", "amount": "1 mol%"}],
        "ligands": [],
        "other_components": [],
        "conditions": {"solvent": solvent, "time": "12 h"},
        "targets": {"yield": yield_value, "ee": None, "er": None, "dr": None},
    }


def test_q1_ignored_merges_types_and_keeps_only_hidden_audit():
    reactions = [
        _reaction("r1", "unknown reaction", "benzaldehyde", "THF", "80%"),
        _reaction("r2", "named reaction", "benzaldehyde", "toluene", "90%"),
    ]

    required = generate_q1_benchmark_package(reactions)
    ignored = generate_q1_benchmark_package(
        reactions,
        reaction_type_policy="ignored",
    )

    assert required["benchmark"] == []
    assert any(
        review["reason"] == "unknown_reaction_type"
        for review in required["review_set"]
    )
    assert len(ignored["benchmark"]) == 1
    question = ignored["benchmark"][0]
    assert "reaction_type" not in question
    assert "Reaction type:" not in question["question_en"]
    assert question["metadata_hidden"]["source_reaction_types"] == [
        "named reaction",
        "unknown reaction",
    ]
    assert question["metadata_hidden"]["reaction_type_conflict"] is True
    assert not any(
        review["reason"] == "unknown_reaction_type"
        for review in ignored["review_set"]
    )
    assert ignored["report"]["reaction_type_policy"] == "ignored"
    assert ignored["report"]["reaction_type_conflict_questions"] == 1


def test_q2_ignored_merges_types_with_same_scaffold_and_conditions():
    reactions = [
        _reaction("r1", "type A", "benzaldehyde", "THF", "80%"),
        _reaction("r2", "type B", "4-fluorobenzaldehyde", "THF", "90%"),
    ]

    assert generate_q2_benchmark_package(reactions)["benchmark"] == []
    ignored = generate_q2_benchmark_package(
        reactions,
        reaction_type_policy="ignored",
    )

    assert len(ignored["benchmark"]) == 1
    question = ignored["benchmark"][0]
    assert "reaction_type" not in question
    assert "Reaction type:" not in question["question_en"]
    assert question["metadata_hidden"]["source_reaction_types"] == [
        "type A",
        "type B",
    ]
    assert question["metadata_hidden"]["reaction_type_conflict"] is True


def test_ignored_modality_output_is_deterministic_and_reports_final_ids():
    q1_reactions = [
        _reaction("r1", "type A", "benzaldehyde", "THF", "80%"),
        _reaction("r2", "type B", "benzaldehyde", "toluene", "90%"),
    ]

    first = generate_benchmark_packages_by_modality(
        q1_reactions,
        [],
        reaction_type_policy="ignored",
    )
    second = generate_benchmark_packages_by_modality(
        q1_reactions,
        [],
        reaction_type_policy="ignored",
    )

    assert first == second
    assert first["q1"]["report"]["reaction_type_conflict_question_ids"] == [
        "Q1_TEXT_0001"
    ]
    assert "reaction_type" not in first["question_option_counts"][0]


@pytest.mark.parametrize("generator", [generate_q1_benchmark_package, generate_q2_benchmark_package])
def test_invalid_reaction_type_policy_is_rejected(generator):
    with pytest.raises(ValueError, match="Unsupported reaction_type_policy"):
        generator([], reaction_type_policy="invalid")


def test_langgraph_ignored_policy_skips_reaction_type_api(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("reaction type normalization API should be skipped")

    monkeypatch.setattr(
        pipeline_agents,
        "normalize_reaction_types_file",
        fail_if_called,
    )
    written = {}
    monkeypatch.setattr(
        pipeline_agents,
        "write_json",
        lambda path, payload: written.update({str(path): payload}),
    )
    root = Path("reaction-type-policy-test")
    config = pipeline_agents.PipelineConfig(
        si_folder=root / "si",
        output_dir=root / "output",
        filtered_dir=root / "filtered",
        intermediate_dir=root / "intermediate",
        api_key="unused",
        benchmark_reaction_type_policy="ignored",
    )
    state = {
        "merged_reactions_path": str(root / "merged.json"),
        "steps": {},
    }

    result = pipeline_agents.ReactionTypeNormalizationAgent(config).run(state)

    step = result["steps"]["reaction_type_normalization"]
    assert step["status"] == "skipped"
    assert step["reason"] == "benchmark_reaction_type_policy_ignored"
