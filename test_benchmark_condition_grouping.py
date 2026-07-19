from langgraph_workflow.benchmark_utils import (
    OBJECTIVE_YIELD_ONLY,
    _condition_grouping_signature,
    _public_option_key,
    format_fixed_conditions_en,
    generate_benchmark_packages_by_modality,
    generate_q1_benchmark_package,
    generate_q2_benchmark_package,
    has_visible_q1_condition,
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


def _q2_reaction(
    reaction_id,
    variable_name,
    yield_value,
    *,
    paper="paper",
    ligand=None,
    symbol=None,
    fixed_symbol=None,
):
    return {
        "id": reaction_id,
        "source_paper": paper,
        "source_modality": "text",
        "reaction_type": "test reaction",
        "substrates": [
            {
                "name": variable_name,
                "scaffold": "benzaldehyde",
                "substituents": [],
                "symbol": symbol,
            },
            {
                "name": "glycine",
                "scaffold": "glycine",
                "substituents": [],
                "symbol": fixed_symbol,
            },
        ],
        "products": [
            {
                "name": f"product from {variable_name}",
                "scaffold": "amino alcohol",
            }
        ],
        "catalysts": [{"name": "photocatalyst", "amount": "1 mol%"}],
        "ligands": [{"name": ligand}] if ligand else [],
        "other_components": [],
        "conditions": {"time": "12 h", "light_source": "photoreactor"},
        "targets": {"yield": yield_value, "ee": None, "er": None},
    }


def test_q2_question_exposes_every_grouping_condition():
    reactions = [
        _q2_reaction("r1", "benzaldehyde", "91%", ligand="(R,S)-L1"),
        _q2_reaction("r2", "4-fluorobenzaldehyde", "88%", ligand="(R,S)-L1"),
    ]

    question = generate_q2_benchmark_package(reactions)["benchmark"][0]

    assert question["condition_signature"]["catalysts"] == [
        {"name": "photocatalyst", "amount": "1 mol%"}
    ]
    assert question["condition_signature"]["ligands"] == [{"name": "(R,S)-L1"}]
    assert "Catalyst(s): photocatalyst (amount: 1 mol%)." in question["question_en"]
    assert "Ligand(s): (R,S)-L1." in question["question_en"]
    assert format_fixed_conditions_en(question["condition_signature"])[-1] == (
        "Ligand(s): (R,S)-L1."
    )


def test_q2_different_grouping_ligands_are_both_visible():
    reactions = [
        _q2_reaction("a1", "benzaldehyde", "91%", paper="paper-a", ligand="(R,S)-L1"),
        _q2_reaction("a2", "4-fluorobenzaldehyde", "88%", paper="paper-a", ligand="(R,S)-L1"),
        _q2_reaction("b1", "benzaldehyde", "91%", paper="paper-b", ligand="(S,R)-L1"),
        _q2_reaction("b2", "4-fluorobenzaldehyde", "88%", paper="paper-b", ligand="(S,R)-L1"),
    ]

    bundle = generate_benchmark_packages_by_modality(
        [], reactions, reaction_type_policy="ignored"
    )

    questions = bundle["q2"]["benchmarks"][OBJECTIVE_YIELD_ONLY]
    assert len(questions) == 2
    assert "(R,S)-L1" in questions[0]["question_en"]
    assert "(S,R)-L1" in questions[1]["question_en"]


def test_q2_identical_public_questions_from_different_papers_are_not_deduplicated():
    reactions = [
        _q2_reaction("a1", "benzaldehyde", "91%", paper="paper-a"),
        _q2_reaction("a2", "4-fluorobenzaldehyde", "88%", paper="paper-a"),
        _q2_reaction("b1", "benzaldehyde", "91%", paper="paper-b"),
        _q2_reaction("b2", "4-fluorobenzaldehyde", "88%", paper="paper-b"),
    ]

    bundle = generate_benchmark_packages_by_modality(
        [], reactions, reaction_type_policy="ignored"
    )

    questions = bundle["q2"]["benchmarks"][OBJECTIVE_YIELD_ONLY]
    assert len(questions) == 2
    assert [question["source_paper"] for question in questions] == [
        "paper-a",
        "paper-b",
    ]
    assert questions[0]["gold_option_ids"]
    assert questions[0]["gold_option_ids"] == questions[1]["gold_option_ids"]


def test_q2_symbol_is_hidden_and_same_result_duplicates_merge_with_provenance():
    reactions = [
        _q2_reaction(
            "r1", "benzaldehyde", "91%", symbol="135-1", fixed_symbol="G1"
        ),
        _q2_reaction(
            "r2", "benzaldehyde", "91%", symbol="S127-4", fixed_symbol="G1"
        ),
        _q2_reaction(
            "r3", "4-fluorobenzaldehyde", "88%", symbol="4-F", fixed_symbol="G1"
        ),
    ]

    question = generate_q2_benchmark_package(reactions)["benchmark"][0]

    assert question["option_count"] == 2
    assert all(
        "symbol" not in option["variable_substrate"]
        for option in question["options"]
    )
    assert all("symbol" not in substrate for substrate in question["fixed_substrates"])
    assert question["metadata_hidden"]["fixed_substrate_source_symbols"] == [
        {"name": "glycine", "symbol": "G1"}
    ]
    benzaldehyde_option = next(
        option
        for option in question["options"]
        if option["variable_substrate"]["name"] == "benzaldehyde"
    )
    result = next(
        result
        for result in question["metadata_hidden"]["option_results"]
        if result["option_id"] == benzaldehyde_option["option_id"]
    )
    assert result["source_symbols"] == ["135-1", "S127-4"]
    assert result["source_reaction_ids"] == ["r1", "r2"]


def test_q2_symbol_and_missing_symbol_do_not_hide_conflicting_results():
    reactions = [
        _q2_reaction("r1", "benzaldehyde", "91%", symbol="135-1"),
        _q2_reaction("r2", "benzaldehyde", "83%"),
        _q2_reaction("r3", "4-fluorobenzaldehyde", "88%"),
        _q2_reaction("r4", "4-bromobenzaldehyde", "85%"),
    ]

    package = generate_q2_benchmark_package(reactions)
    question = package["benchmark"][0]

    assert question["option_count"] == 2
    assert {
        option["variable_substrate"]["name"] for option in question["options"]
    } == {"4-fluorobenzaldehyde", "4-bromobenzaldehyde"}
    conflict = next(
        review
        for review in package["review_set"]
        if review["reason"] == "duplicate_variable_substrate_conflicting_results"
    )
    assert conflict["details"][0]["source_symbols"] == ["135-1"]
    assert conflict["details"][0]["reaction_ids"] == ["r1", "r2"]


def test_q2_same_symbol_different_chemistry_remains_distinct():
    reactions = [
        _q2_reaction("r1", "benzaldehyde", "91%", symbol="1"),
        _q2_reaction("r2", "4-fluorobenzaldehyde", "88%", symbol="1"),
    ]

    question = generate_q2_benchmark_package(reactions)["benchmark"][0]

    assert question["option_count"] == 2
    assert {
        option["variable_substrate"]["name"] for option in question["options"]
    } == {"benzaldehyde", "4-fluorobenzaldehyde"}
    assert all(
        "symbol" not in option["variable_substrate"]
        for option in question["options"]
    )


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


def _q1_reaction(reaction_id, condition_name, yield_value):
    return {
        "id": reaction_id,
        "source_paper": "paper",
        "source_modality": "text",
        "reaction_type": "test reaction",
        "substrates": [
            {
                "name": "benzaldehyde",
                "scaffold": "benzaldehyde",
                "substituents": [],
            }
        ],
        "products": [
            {
                "name": "(R)-1-phenylethanol",
                "scaffold": "benzene",
            }
        ],
        "catalysts": [],
        "ligands": [],
        "other_components": [],
        "conditions": {"solvent": condition_name},
        "targets": {"yield": yield_value, "ee": None, "er": None, "dr": None},
    }


def test_q1_accepts_any_visible_structured_condition_field():
    field_values = {
        "conditions": {"solvent": "THF"},
        "catalysts": [{"name": "Pd(OAc)2", "amount": "5 mol%"}],
        "ligands": [{"name": "BINAP"}],
        "reagents": [{"name": "K2CO3"}],
        "additives": [{"name": "LiCl"}],
        "other_components": [{"name": "base", "amount": "2 equiv"}],
        "electrodes": ["graphite"],
        "atmosphere": "nitrogen",
        "scale": "0.1 mmol",
    }
    for field, value in field_values.items():
        reaction = _q1_reaction("r", "", "80%")
        reaction["conditions"] = {}
        reaction[field] = value
        assert has_visible_q1_condition(reaction), field


def test_q1_rejects_reaction_without_visible_structured_conditions():
    reaction = _q1_reaction("r", "", "80%")
    reaction["conditions"] = {}
    reaction["targets"] = {"yield": "80%", "ee": "90%"}
    reaction["source_pages"] = [1]
    assert not has_visible_q1_condition(reaction)


def test_q1_groups_by_name_and_hides_public_substrate_symbol():
    first = _q1_reaction("r1", "THF", "91%")
    second = _q1_reaction("r2", "DMF", "88%")
    first["substrates"][0].update(
        {"symbol": "1a", "scaffold": "scaffold-a", "substituents": ["4-F"]}
    )
    second["substrates"][0].update(
        {"symbol": "S-22", "scaffold": "scaffold-b", "substituents": ["4-Cl"]}
    )

    package = generate_q1_benchmark_package([first, second])

    assert len(package["benchmark"]) == 1
    question = package["benchmark"][0]
    assert question["option_count"] == 2
    assert all("symbol" not in substrate for substrate in question["substrate_combo"])
    provenance = {
        result["reaction_id"]: result["source_substrate_symbols"]
        for result in question["metadata_hidden"]["option_results"]
    }
    assert provenance == {
        "r1": [{"name": "benzaldehyde", "symbol": "1a"}],
        "r2": [{"name": "benzaldehyde", "symbol": "S-22"}],
    }


def test_q1_omits_conflicting_duplicate_condition_not_whole_question():
    reactions = [
        _q1_reaction("r1", "condition A", "91%"),
        _q1_reaction("r2", "condition A", "82%"),
        _q1_reaction("r3", "condition B", "88%"),
        _q1_reaction("r4", "condition C", "95%"),
    ]

    package = generate_q1_benchmark_package(reactions)

    assert len(package["benchmark"]) == 1
    question = package["benchmark"][0]
    assert question["option_count"] == 2
    assert {
        option["conditions"]["solvent"] for option in question["options"]
    } == {"condition B", "condition C"}
    assert len(question["gold_option_ids"]) == 1
    gold_option_id = question["gold_option_ids"][0]
    gold_option = next(
        option for option in question["options"] if option["option_id"] == gold_option_id
    )
    assert gold_option["conditions"]["solvent"] == "condition C"

    omitted = question["metadata_hidden"]["omitted_conflicting_duplicate_options"]
    assert len(omitted) == 1
    assert omitted[0]["public_condition"]["conditions"]["solvent"] == "condition A"
    assert omitted[0]["reaction_ids"] == ["r1", "r2"]

    conflict_reviews = [
        review
        for review in package["review_set"]
        if review.get("reason") == "duplicate_public_conditions_conflicting_results"
    ]
    assert len(conflict_reviews) == 1
    assert conflict_reviews[0]["action"] == "omitted_conflicting_public_condition"


def test_q1_conflict_removal_can_leave_too_few_distinct_conditions():
    reactions = [
        _q1_reaction("r1", "condition A", "91%"),
        _q1_reaction("r2", "condition A", "82%"),
        _q1_reaction("r3", "condition B", "88%"),
    ]

    package = generate_q1_benchmark_package(reactions)

    assert package["benchmark"] == []
    review_by_reason = {review["reason"]: review for review in package["review_set"]}
    assert "duplicate_public_conditions_conflicting_results" in review_by_reason
    insufficient = review_by_reason["insufficient_distinct_conditions"]
    assert insufficient["option_count"] == 1
    assert len(insufficient["omitted_conflicting_duplicate_options"]) == 1


def test_q1_dedupes_identical_condition_results_without_conflict_review():
    reactions = [
        _q1_reaction("r1", "condition A", "91%"),
        _q1_reaction("r2", "condition A", "91%"),
        _q1_reaction("r3", "condition B", "88%"),
    ]

    package = generate_q1_benchmark_package(reactions)

    assert len(package["benchmark"]) == 1
    question = package["benchmark"][0]
    assert question["option_count"] == 2
    assert question["metadata_hidden"]["omitted_conflicting_duplicate_options"] == []
    assert not any(
        review.get("reason") == "duplicate_public_conditions_conflicting_results"
        for review in package["review_set"]
    )


def test_q1_omits_all_conflicting_condition_groups():
    reactions = [
        _q1_reaction("r1", "condition A", "91%"),
        _q1_reaction("r2", "condition A", "82%"),
        _q1_reaction("r3", "condition B", "88%"),
        _q1_reaction("r4", "condition B", "77%"),
        _q1_reaction("r5", "condition C", "75%"),
        _q1_reaction("r6", "condition D", "80%"),
    ]

    package = generate_q1_benchmark_package(reactions)

    assert len(package["benchmark"]) == 1
    question = package["benchmark"][0]
    assert {
        option["conditions"]["solvent"] for option in question["options"]
    } == {"condition C", "condition D"}
    omitted = question["metadata_hidden"]["omitted_conflicting_duplicate_options"]
    assert len(omitted) == 2
    assert {
        item["public_condition"]["conditions"]["solvent"] for item in omitted
    } == {"condition A", "condition B"}
