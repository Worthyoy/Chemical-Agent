import json
import sys
from pathlib import Path
from types import SimpleNamespace


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from batch_si_extractor import SIExtractor  # noqa: E402


def make_extractor():
    return SIExtractor.__new__(SIExtractor)


def resolve_compounds(compounds, registry):
    extractor = make_extractor()
    reactions = [{"id": "r1", "substrates": compounds, "products": [{"name": "product"}]}]
    resolved = extractor.align_names_in_reactions(reactions, registry)
    return resolved[0]["substrates"], extractor.last_registry_resolution_stats


def test_registry_resolves_name_equal_to_symbol():
    compounds, stats = resolve_compounds(
        [{"name": "1a", "symbol": "1a"}],
        {"1a": "methyl cinnamate"},
    )

    assert compounds[0]["name"] == "methyl cinnamate"
    assert compounds[0]["symbol"] == "1a"
    assert compounds[0]["resolution_source"] == "name_registry"
    assert compounds[0]["resolution_method"] == "same_paper_symbol"
    assert stats["resolved"] == 1


def test_registry_resolves_empty_name_from_symbol():
    compounds, _stats = resolve_compounds(
        [{"name": "", "symbol": "1a"}],
        {"1a": "methyl cinnamate"},
    )

    assert compounds[0]["name"] == "methyl cinnamate"
    assert compounds[0]["symbol"] == "1a"


def test_registry_resolves_generic_label_name():
    compounds, _stats = resolve_compounds(
        [{"name": "substrate 1a", "symbol": "1a"}],
        {"1a": "methyl cinnamate"},
    )

    assert compounds[0]["name"] == "methyl cinnamate"
    assert compounds[0]["symbol"] == "1a"


def test_registry_marks_conflict_without_overwriting_explicit_full_name():
    compounds, stats = resolve_compounds(
        [{"name": "explicit full chemical name", "symbol": "1a"}],
        {"1a": "methyl cinnamate"},
    )

    assert compounds[0]["name"] == "explicit full chemical name"
    assert compounds[0]["symbol"] == "1a"
    assert "resolution_source" not in compounds[0]
    assert compounds[0]["registry_name"] == "methyl cinnamate"
    assert compounds[0]["registry_match_status"] == "conflict"
    assert compounds[0]["registry_conflict_reason"] == "extracted_name_differs_from_registry_symbol_name"
    assert stats["resolved"] == 0
    assert stats["conflicts"] == 1


def test_registry_marks_equivalent_explicit_name_as_verified():
    compounds, stats = resolve_compounds(
        [{"name": "methyl - cinnamate", "symbol": "1a"}],
        {"1a": "Methyl-cinnamate"},
    )

    assert compounds[0]["name"] == "methyl - cinnamate"
    assert compounds[0]["symbol"] == "1a"
    assert compounds[0]["registry_name"] == "Methyl-cinnamate"
    assert compounds[0]["registry_match_status"] == "verified"
    assert "resolution_source" not in compounds[0]
    assert stats["resolved"] == 0
    assert stats["verified"] == 1


def test_registry_converts_string_symbol_to_compound_dict():
    compounds, _stats = resolve_compounds(["1a"], {"1a": "methyl cinnamate"})

    assert compounds[0] == {
        "name": "methyl cinnamate",
        "symbol": "1a",
        "resolution_source": "name_registry",
        "resolution_method": "same_paper_symbol",
    }


def test_registry_leaves_unmatched_label_only_compound_unchanged():
    compounds, stats = resolve_compounds(
        [{"name": "1z", "symbol": "1z"}],
        {"1a": "methyl cinnamate"},
    )

    assert compounds[0] == {"name": "1z", "symbol": "1z"}
    assert stats["resolved"] == 0
    assert stats["verified"] == 0
    assert stats["conflicts"] == 0


def test_registry_resolution_stats_count_all_statuses():
    compounds, stats = resolve_compounds(
        [
            {"name": "1a", "symbol": "1a"},
            {"name": "methyl - cinnamate", "symbol": "1b"},
            {"name": "different full name", "symbol": "1c"},
        ],
        {
            "1a": "methyl cinnamate",
            "1b": "Methyl-cinnamate",
            "1c": "registry full name",
        },
    )

    assert compounds[0]["name"] == "methyl cinnamate"
    assert compounds[1]["registry_match_status"] == "verified"
    assert compounds[2]["registry_match_status"] == "conflict"
    assert stats == {"resolved": 1, "verified": 1, "conflicts": 1}


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(self.response))
                )
            ]
        )


def test_stage2_prompt_does_not_inject_registry_mapping():
    extractor = make_extractor()
    completions = FakeCompletions([])
    extractor.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    extractor.extract_model = "fake-model"
    extractor.enable_stage2_audit = False

    extractor.stage2_extract(
        "Reaction of 1a gave 2a.",
        "chunk 1",
        registry={"1a": "methyl cinnamate"},
        gp_texts={},
    )

    user_prompt = completions.calls[0]["messages"][1]["content"]
    assert "KNOWN SYMBOL-TO-NAME MAPPINGS" not in user_prompt
    assert "methyl cinnamate" not in user_prompt
    assert "do not invent a full name" in user_prompt
