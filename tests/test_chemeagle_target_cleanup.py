from langgraph_workflow.chemeagle_adapter import cleanup_reaction_targets
from reaction_filter import ReactionFilter


def _reaction(reaction_id, targets):
    return {
        "id": reaction_id,
        "substrates": [{"name": "substrate"}],
        "products": [{"name": "product A"}],
        "targets": targets,
    }


def test_cleanup_reaction_targets_strips_outer_parentheses():
    payload = {
        "reactions": [
            _reaction(
                "r1",
                {
                    "yield": " (99%) ",
                    "ee": "(95%)",
                    "er": "(99:1)",
                    "dr": None,
                },
            ),
            _reaction("r2", {"yield": "96 (99%)"}),
        ]
    }

    cleanup_reaction_targets(payload)

    assert payload["reactions"][0]["targets"]["yield"] == "99%"
    assert payload["reactions"][0]["targets"]["ee"] == "95%"
    assert payload["reactions"][0]["targets"]["er"] == "99:1"
    assert payload["reactions"][0]["targets"]["dr"] is None
    assert payload["reactions"][1]["targets"]["yield"] == "96 (99%)"


def test_yield_ee_and_dr_presence_are_enough_for_target_filter():
    payload = {
        "reactions": [
            _reaction("kept_yield_parenthesized", {"yield": "(~100)"}),
            _reaction("kept_ee_parenthesized", {"ee": "(95%)"}),
            _reaction("kept_dr_presence", {"dr": ">20:1"}),
            _reaction("filtered_null", {"yield": "Null"}),
            _reaction("filtered_none", {"yield": None}),
            _reaction("filtered_empty", {"yield": "  "}),
            _reaction("filtered_trace", {"yield": "trace"}),
            _reaction("filtered_na_ee", {"ee": "N/A"}),
            _reaction("filtered_not_specified_dr", {"dr": "not specified"}),
        ]
    }
    cleanup_reaction_targets(payload)

    filtered = ReactionFilter().filter_reactions(payload["reactions"])

    assert [reaction["id"] for reaction in filtered] == [
        "kept_yield_parenthesized",
        "kept_ee_parenthesized",
        "kept_dr_presence",
    ]


def test_er_keeps_strict_numeric_or_ratio_filter():
    payload = {
        "reactions": [
            _reaction("kept_er_ratio", {"er": "99:1"}),
            _reaction("kept_er_greater_than_ratio", {"er": ">99:1"}),
            _reaction("filtered_er_not_specified", {"er": "not specified"}),
            _reaction("filtered_er_text", {"er": "high"}),
        ]
    }
    cleanup_reaction_targets(payload)

    filtered = ReactionFilter().filter_reactions(payload["reactions"])

    assert [reaction["id"] for reaction in filtered] == [
        "kept_er_ratio",
        "kept_er_greater_than_ratio",
    ]
