from langgraph_workflow.benchmark_utils import (
    _condition_grouping_signature,
    _public_option_key,
    generate_q2_benchmark_package,
)


def _group_key(condition_option):
    return _public_option_key(_condition_grouping_signature(condition_option))


def test_q2_condition_grouping_ignores_amount_punctuation_only():
    left = {
        "other_components": [
            {"name": "SOCl2", "amount": "711 μL 9.75 mmol, 3.25 equiv."}
        ]
    }
    right = {
        "other_components": [
            {"name": "SOCl2", "amount": "711 μL, 9.75 mmol, 3.25 equiv."}
        ]
    }
    assert _group_key(left) == _group_key(right)


def test_q2_condition_grouping_ignores_parentheses_around_condition_qualifier():
    left = {
        "conditions": {
            "light_source": "Kessil lamp set at 50% of its maximum output power, 427 nm"
        }
    }
    right = {
        "conditions": {
            "light_source": "Kessil lamp set at 50% of its maximum output power (427 nm)"
        }
    }
    assert _group_key(left) == _group_key(right)


def test_q2_condition_grouping_does_not_infer_missing_amount_tokens():
    left = {
        "catalysts": [
            {"name": "Pd2(dba)3", "amount": "4.6 mg, 0.005 mmol, 5 mol%"}
        ]
    }
    right = {
        "catalysts": [
            {"name": "Pd2(dba)3", "amount": "4.6 mg, 5 mol%"}
        ]
    }
    assert _group_key(left) != _group_key(right)


def _q2_reaction(reaction_id, variable_name, yield_value):
    return {
        "id": reaction_id,
        "source_paper": "paper",
        "source_modality": "text",
        "reaction_type": "test reaction",
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
                "name": f"product from {variable_name}",
                "scaffold": "amino alcohol",
            }
        ],
        "catalysts": [{"name": "photocatalyst", "amount": "1 mol%"}],
        "ligands": [],
        "other_components": [],
        "conditions": {"time": "12 h", "light_source": "photoreactor"},
        "targets": {"yield": yield_value, "ee": None, "er": None},
    }


def test_q2_omits_conflicting_duplicate_variable_substrate_not_whole_question():
    reactions = [
        _q2_reaction("r1", "benzaldehyde", "91%"),
        _q2_reaction("r2", "benzaldehyde", "92%"),
        _q2_reaction("r3", "benzaldehyde", "90%"),
        _q2_reaction("r4", "4-chlorobenzaldehyde", "88%"),
        _q2_reaction("r5", "4-bromobenzaldehyde", "85%"),
    ]

    package = generate_q2_benchmark_package(reactions)
    questions = package["benchmark"]
    reviews = package["review_set"]

    assert len(questions) == 1
    assert questions[0]["option_count"] == 2
    option_names = {
        option["variable_substrate"]["name"]
        for option in questions[0]["options"]
    }
    assert option_names == {"4-chlorobenzaldehyde", "4-bromobenzaldehyde"}
    assert questions[0]["metadata_hidden"]["omitted_conflicting_duplicate_options"]

    duplicate_reviews = [
        review
        for review in reviews
        if review.get("reason") == "duplicate_variable_substrate_conflicting_results"
    ]
    assert len(duplicate_reviews) == 1
    assert duplicate_reviews[0]["action"] == "omitted_conflicting_variable_substrate"
