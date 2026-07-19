from langgraph_workflow.benchmark_utils import (
    OBJECTIVE_COMBINED,
    OBJECTIVE_YIELD_ONLY,
    generate_q1_benchmark_package,
)


def _reaction(reaction_id, solvent, targets):
    return {
        "id": reaction_id,
        "source_paper": "paper",
        "source_modality": "text",
        "reaction_type": "reaction",
        "substrates": [{"name": "substrate", "scaffold": "scaffold"}],
        "products": [{"name": "product", "scaffold": "product scaffold"}],
        "conditions": {"solvent": solvent},
        "catalysts": [],
        "ligands": [],
        "other_components": [],
        "targets": targets,
    }


def test_mixed_group_splits_into_independent_objective_questions():
    reactions = [
        _reaction("y1", "THF", {"yield": "80%", "ee": None, "er": None}),
        _reaction("y2", "DMF", {"yield": "70%", "ee": "N/A", "er": "N/A"}),
        _reaction("c1", "toluene", {"yield": "90%", "ee": "60%", "er": None}),
        _reaction("c2", "EtOH", {"yield": "75%", "ee": "95%", "er": None}),
    ]
    combined = generate_q1_benchmark_package(
        reactions, objective=OBJECTIVE_COMBINED
    )["benchmark"]
    yield_only = generate_q1_benchmark_package(
        reactions, objective=OBJECTIVE_YIELD_ONLY
    )["benchmark"]

    assert len(combined) == len(yield_only) == 1
    assert combined[0]["objective"] == OBJECTIVE_COMBINED
    assert yield_only[0]["objective"] == OBJECTIVE_YIELD_ONLY
    assert combined[0]["option_count"] == yield_only[0]["option_count"] == 2
    assert "best combined ee and yield" in combined[0]["question_en"]
    assert "best reported yield" in yield_only[0]["question_en"]
    assert yield_only[0]["source_missingness_verified"] is False
    assert all(
        result["yield"] is not None and result["ee"] is not None
        for result in combined[0]["metadata_hidden"]["option_results"]
    )
    assert all(
        result["yield"] is not None and result["ee"] is None
        for result in yield_only[0]["metadata_hidden"]["option_results"]
    )


def test_single_option_subset_goes_to_review():
    reactions = [
        _reaction("y1", "THF", {"yield": "80%", "ee": None, "er": None}),
        _reaction("c1", "DMF", {"yield": "90%", "ee": "60%", "er": None}),
    ]
    for objective in (OBJECTIVE_COMBINED, OBJECTIVE_YIELD_ONLY):
        package = generate_q1_benchmark_package(reactions, objective=objective)
        assert package["benchmark"] == []
        assert any(
            review["reason"] == "insufficient_objective_options"
            for review in package["review_set"]
        )


def test_ambiguous_multiple_er_is_neither_combined_nor_yield_only():
    reactions = [
        _reaction("r1", "THF", {"yield": "80%", "ee": None, "er": "96:4/88:12"}),
        _reaction("r2", "DMF", {"yield": "70%", "ee": None, "er": "96:4/90:10"}),
    ]
    for objective in (OBJECTIVE_COMBINED, OBJECTIVE_YIELD_ONLY):
        package = generate_q1_benchmark_package(reactions, objective=objective)
        assert package["benchmark"] == []
        exclusions = [
            excluded
            for review in package["review_set"]
            for excluded in review.get("excluded_options", [])
        ]
        assert exclusions
        assert {item["reason"] for item in exclusions} == {"ambiguous_er_without_ee"}
