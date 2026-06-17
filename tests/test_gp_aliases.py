import sys
from pathlib import Path


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from batch_si_extractor import SIExtractor  # noqa: E402


def make_extractor():
    extractor = SIExtractor.__new__(SIExtractor)
    extractor.last_gp_selection_debug = {}
    return extractor


def test_typical_procedure_matches_plural_aliases():
    extractor = make_extractor()

    assert extractor._text_contains_gp_alias(
        "The product was prepared according to Procedures A.",
        "Typical Procedure A",
    )
    assert extractor._text_contains_gp_alias(
        "The product was prepared according to Typical Procedures A.",
        "Typical Procedure A",
    )
    assert extractor._text_contains_gp_alias(
        "The product was prepared according to General Procedures A.",
        "Typical Procedure A",
    )


def test_plural_alias_keeps_label_specificity():
    extractor = make_extractor()

    assert extractor._text_contains_gp_alias(
        "The product was prepared according to Procedures B.",
        "General Procedure B",
    )
    assert not extractor._text_contains_gp_alias(
        "The product was prepared according to Procedures A.",
        "Typical Procedure B",
    )


def test_select_gp_for_chunk_uses_plural_alias():
    extractor = make_extractor()
    gp_texts = {
        "Typical Procedure A": "Typical Procedure A: heat in toluene.",
        "Typical Procedure B": "Typical Procedure B: cool in THF.",
    }

    selected = extractor.select_gp_for_chunk(
        "Compound 1a was prepared according to Procedures A.",
        gp_texts,
    )

    assert list(selected) == ["Typical Procedure A"]
    assert extractor.last_gp_selection_debug == {
        "mode": "explicit_match",
        "selected_gp_keys": ["Typical Procedure A"],
    }


def test_select_gp_for_chunk_keeps_multiple_explicit_matches():
    extractor = make_extractor()
    gp_texts = {
        "Typical Procedure A": "Typical Procedure A: heat in toluene.",
        "Typical Procedure B": "Typical Procedure B: cool in THF.",
    }

    selected = extractor.select_gp_for_chunk(
        "Products 1a and 3a followed Procedures A and Procedures B, respectively.",
        gp_texts,
    )

    assert list(selected) == ["Typical Procedure A", "Typical Procedure B"]
    assert extractor.last_gp_selection_debug == {
        "mode": "explicit_match",
        "selected_gp_keys": ["Typical Procedure A", "Typical Procedure B"],
    }


def test_select_gp_for_chunk_without_reference_is_unchanged():
    extractor = make_extractor()
    gp_texts = {
        "Typical Procedure A": "Typical Procedure A: heat in toluene.",
        "Typical Procedure B": "Typical Procedure B: cool in THF.",
    }

    selected = extractor.select_gp_for_chunk(
        "Compound 1a was obtained as a white solid in 53% yield.",
        gp_texts,
    )

    assert selected == {}
    assert extractor.last_gp_selection_debug == {
        "mode": "no_reference",
        "selected_gp_keys": [],
    }
