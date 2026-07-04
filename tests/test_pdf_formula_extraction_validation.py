import sys
from pathlib import Path


EXTRACT_DIR = Path(__file__).resolve().parents[1]
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from validate_pdf_formula_extraction import (  # noqa: E402
    extract_reference_formula_candidates,
    find_broken_formula_hits,
    is_reference_checkable_page,
)


def test_broken_formula_detector_finds_known_subscript_failures():
    text = """Na CO was added.
MgSO, filtered, and concentrated.
NaHCO solution was used.
NiCl (DME) was charged.
CH Cl was the solvent.
Et O was used for extraction.
"""

    hits = find_broken_formula_hits(text, page_number=7)
    patterns = {hit["pattern"] for hit in hits}

    assert {hit["page"] for hit in hits} == {7}
    assert patterns == {
        "alkali_carbonate_gap",
        "magnesium_sulfate_missing_subscript",
        "bicarbonate_missing_subscript",
        "nickel_chloride_missing_subscript",
        "chlorinated_solvent_gap",
        "ether_gap",
    }


def test_broken_formula_detector_accepts_complete_ascii_formulas():
    text = (
        "Na2CO3, K2SO4, MgSO4, NaHCO3, NiCl2(DME), CH2Cl2, "
        "CDCl3, Et2O, and NaCl were used."
    )

    assert find_broken_formula_hits(text, page_number=1) == []


def test_reference_formula_candidates_normalize_interstitial_spaces():
    text = "Na 2CO 3, CH 2Cl 2, CDCl 3, and Na 2SO 4 were used."

    assert extract_reference_formula_candidates(text) >= {
        "Na2CO3",
        "CH2Cl2",
        "CDCl3",
        "Na2SO4",
    }


def test_reference_formula_candidates_reject_concatenated_spectral_numbers():
    text = "CDCl37.39 HCHBHDHDHC33 S133UPC2"

    assert "CDCl37" not in extract_reference_formula_candidates(text)
    assert "HCHBHDHDHC33" not in extract_reference_formula_candidates(text)
    assert "S133UPC2" not in extract_reference_formula_candidates(text)


def test_reference_cross_check_skips_analytical_only_pages():
    assert not is_reference_checkable_page("NMR Spectra\nCDCl3")
    assert not is_reference_checkable_page("Crystal structure determination\nCCDC 123")
    assert is_reference_checkable_page("General Procedure A: Na2CO3 was added")
