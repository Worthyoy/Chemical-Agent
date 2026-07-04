"""Validate that PDF text extraction preserves chemical formula subscripts.

This is an offline corpus validation tool.  It deliberately does not import the
reaction extraction pipeline and never calls an LLM or the network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = SCRIPT_DIR.parent / "supporting_information"


@dataclass(frozen=True)
class BrokenFormulaPattern:
    name: str
    expression: str
    expected_example: str


BROKEN_FORMULA_PATTERNS = (
    BrokenFormulaPattern(
        "alkali_carbonate_gap",
        r"\b(?:Na|K|Li)\s+CO\b",
        "Na2CO3",
    ),
    BrokenFormulaPattern(
        "alkali_sulfate_gap",
        r"\b(?:Na|K|Li)\s+SO\b",
        "Na2SO4",
    ),
    BrokenFormulaPattern(
        "magnesium_sulfate_missing_subscript",
        r"\bMgSO(?=\s*(?:[,.;)]|$)|\s+(?:and|was|were|filtered|dried)\b)",
        "MgSO4",
    ),
    BrokenFormulaPattern(
        "bicarbonate_missing_subscript",
        r"\b(?:Na|K)HCO(?=\s*(?:[,.;)]|$)|\s+(?:solution|was|were)\b)",
        "NaHCO3",
    ),
    BrokenFormulaPattern(
        "nickel_chloride_missing_subscript",
        r"\bNiCl\s*(?=[([])",
        "NiCl2",
    ),
    BrokenFormulaPattern(
        "chlorinated_solvent_gap",
        r"\b(?:CH|CD)\s+Cl\b",
        "CH2Cl2 or CDCl3",
    ),
    BrokenFormulaPattern(
        "deuterated_chloroform_missing_subscript",
        r"\bCDCl(?=\s*(?:[,.;:)]|$))",
        "CDCl3",
    ),
    BrokenFormulaPattern(
        "ether_gap",
        r"\bEt\s+O\b",
        "Et2O",
    ),
)

COMPILED_BROKEN_PATTERNS = tuple(
    (pattern, re.compile(pattern.expression, re.IGNORECASE))
    for pattern in BROKEN_FORMULA_PATTERNS
)

# A conservative set used only to validate candidates recovered by pypdf.  D
# is included for deuterated solvents such as CDCl3.
ELEMENT_SYMBOLS = {
    "H", "D", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
    "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}

REFERENCE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z])(?:[A-Z][a-z]?\s*\d*\s*){2,}(?![a-z])"
)
FORMULA_COMPONENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

ANALYTICAL_ONLY_MARKERS = re.compile(
    r"(?:NMR\s+Spectra|NOE\s+spectrum|UPC2\s+traces|HPLC\s+traces|"
    r"Crystal\s+Structure|Crystal\s+data|Crystal\s+structure\s+determination|"
    r"CCDC\s+\d+|diffractometer)",
    re.IGNORECASE,
)

KNOWN_EXPECTATIONS = {
    "enantioselective-nickel-catalyzed-mizoroki-heck-cyclizations-to-generate-quaternary-stereocenters.pdf": (
        "Na2CO3",
        "NiCl2(Pn-Bu3)2",
        "Na2SO4",
        "NaHCO3",
        "CH2Cl2",
        "CDCl3",
    ),
}


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def extract_reference_formula_candidates(text: str) -> set[str]:
    """Extract conservative, digit-bearing formulas from reference text."""
    candidates: set[str] = set()
    for match in REFERENCE_TOKEN_RE.finditer(text or ""):
        compact = compact_whitespace(match.group())
        if not any(character.isdigit() for character in compact):
            continue
        components = FORMULA_COMPONENT_RE.findall(compact)
        if len(components) < 2:
            continue
        rebuilt = "".join(symbol + count for symbol, count in components)
        if rebuilt != compact:
            continue
        counts = [int(count) for _, count in components if count]
        # pypdf can concatenate axis values or peak labels into a formula on
        # spectra pages (for example CDCl37.39...).  Large counts are skipped
        # because this check is a conservative cross-check, not a formula
        # discovery pipeline.
        if counts and max(counts) > 32:
            continue
        if all(symbol in ELEMENT_SYMBOLS for symbol, _ in components):
            candidates.add(compact)
    return candidates


def is_reference_checkable_page(text: str) -> bool:
    """Exclude analytical-only pages that the reaction pipeline also ignores."""
    return not bool(ANALYTICAL_ONLY_MARKERS.search(text or ""))


def find_broken_formula_hits(text: str, page_number: int) -> list[dict]:
    hits: list[dict] = []
    for line_number, line in enumerate((text or "").splitlines(), start=1):
        for pattern, compiled in COMPILED_BROKEN_PATTERNS:
            for match in compiled.finditer(line):
                hits.append(
                    {
                        "page": page_number,
                        "line": line_number,
                        "pattern": pattern.name,
                        "matched_text": match.group(),
                        "expected_example": pattern.expected_example,
                        "context": line.strip(),
                    }
                )
    return hits


def load_reference_reader(pdf_path: Path):
    try:
        from pypdf import PdfReader

        return PdfReader(str(pdf_path)), "pypdf"
    except ImportError:
        try:
            from PyPDF2 import PdfReader

            return PdfReader(str(pdf_path)), "PyPDF2"
        except ImportError as exc:
            raise RuntimeError("pypdf or PyPDF2 is required for reference checking") from exc


def validate_pdf(
    pdf_path: Path,
    *,
    x_tolerance: float,
    y_tolerance: float,
    check_reference: bool,
) -> tuple[dict, str]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required") from exc

    started = time.perf_counter()
    broken_hits: list[dict] = []
    missing_reference_formulas: list[dict] = []
    extracted_pages: list[str] = []
    empty_text_pages: list[int] = []
    image_only_pages: list[int] = []
    reference_skipped_pages: list[int] = []

    reference_reader = None
    reference_backend = None
    if check_reference:
        reference_reader, reference_backend = load_reference_reader(pdf_path)

    with pdfplumber.open(pdf_path) as document:
        page_count = len(document.pages)
        if reference_reader is not None and len(reference_reader.pages) != page_count:
            raise RuntimeError(
                f"page count mismatch: pdfplumber={page_count}, "
                f"{reference_backend}={len(reference_reader.pages)}"
            )

        for page_number, page in enumerate(document.pages, start=1):
            text = page.extract_text(
                x_tolerance=x_tolerance,
                y_tolerance=y_tolerance,
            ) or ""
            extracted_pages.append(text)
            if not text.strip():
                empty_text_pages.append(page_number)
                if page.images:
                    image_only_pages.append(page_number)
            broken_hits.extend(find_broken_formula_hits(text, page_number))

            if reference_reader is not None:
                reference_text = reference_reader.pages[page_number - 1].extract_text() or ""
                if not is_reference_checkable_page(text + "\n" + reference_text):
                    reference_skipped_pages.append(page_number)
                    continue
                tuned_compact = compact_whitespace(text)
                for formula in sorted(extract_reference_formula_candidates(reference_text)):
                    if formula not in tuned_compact:
                        missing_reference_formulas.append(
                            {
                                "page": page_number,
                                "formula": formula,
                            }
                        )

    combined_text = "\n".join(extracted_pages)
    combined_compact = compact_whitespace(combined_text)
    expected_formulas = KNOWN_EXPECTATIONS.get(pdf_path.name, ())
    missing_expected_formulas = [
        formula for formula in expected_formulas if formula not in combined_compact
    ]

    status = "passed"
    if broken_hits or missing_reference_formulas or missing_expected_formulas:
        status = "failed"
    if not any(page.strip() for page in extracted_pages):
        status = "unverifiable"

    report = {
        "pdf": str(pdf_path),
        "pdf_name": pdf_path.name,
        "status": status,
        "pages": len(extracted_pages),
        "nonempty_text_pages": len(extracted_pages) - len(empty_text_pages),
        "empty_text_pages": empty_text_pages,
        "image_only_pages": image_only_pages,
        "extracted_characters": len(combined_text),
        "broken_formula_hit_count": len(broken_hits),
        "broken_formula_hits": broken_hits,
        "reference_backend": reference_backend,
        "reference_skipped_pages": reference_skipped_pages,
        "missing_reference_formula_count": len(missing_reference_formulas),
        "missing_reference_formulas": missing_reference_formulas,
        "expected_formulas": list(expected_formulas),
        "missing_expected_formulas": missing_expected_formulas,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    rendered_text = "\n\n".join(
        f"--- Page {page_number} ---\n{text}"
        for page_number, text in enumerate(extracted_pages, start=1)
    )
    return report, rendered_text + "\n"


def write_outputs(output_dir: Path, documents: Iterable[tuple[dict, str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for report, text in documents:
        text_path = output_dir / f"{Path(report['pdf_name']).stem}.txt"
        text_path.write_text(text, encoding="utf-8")


def print_document_report(report: dict, max_details: int) -> None:
    print(
        f"[{report['status'].upper()}] {report['pdf_name']} | "
        f"pages={report['pages']} chars={report['extracted_characters']} "
        f"broken={report['broken_formula_hit_count']} "
        f"reference_missing={report['missing_reference_formula_count']} "
        f"seconds={report['elapsed_seconds']}"
    )
    details = report["broken_formula_hits"] + [
        {
            "page": row["page"],
            "pattern": "reference_formula_missing",
            "matched_text": row["formula"],
            "context": "formula present in reference extractor but absent from tuned text",
        }
        for row in report["missing_reference_formulas"]
    ]
    for detail in details[:max_details]:
        print(
            f"  page {detail['page']}: {detail['pattern']} "
            f"{detail['matched_text']!r} | {detail['context']}"
        )
    if len(details) > max_details:
        print(f"  ... {len(details) - max_details} additional findings omitted")
    if report["missing_expected_formulas"]:
        print("  missing expected: " + ", ".join(report["missing_expected_formulas"]))
    if report["image_only_pages"]:
        print(
            "  warning: image-only pages (not formula-verifiable): "
            + ", ".join(map(str, report["image_only_pages"]))
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate PDF chemical formula extraction with tuned pdfplumber tolerances."
    )
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--x-tolerance", type=float, default=3.0)
    parser.add_argument("--y-tolerance", type=float, default=5.0)
    parser.add_argument("--skip-reference-check", action="store_true")
    parser.add_argument("--max-details", type=int, default=20)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    pdf_dir = args.pdf_dir.resolve()
    if not pdf_dir.is_dir():
        print(f"ERROR: PDF directory does not exist: {pdf_dir}", file=sys.stderr)
        return 2
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"ERROR: no PDF files found in {pdf_dir}", file=sys.stderr)
        return 2

    documents: list[tuple[dict, str]] = []
    processing_errors: list[dict] = []
    for pdf_path in pdf_paths:
        try:
            report, text = validate_pdf(
                pdf_path,
                x_tolerance=args.x_tolerance,
                y_tolerance=args.y_tolerance,
                check_reference=not args.skip_reference_check,
            )
        except Exception as exc:
            processing_errors.append(
                {
                    "pdf": str(pdf_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[ERROR] {pdf_path.name}: {type(exc).__name__}: {exc}")
            continue
        documents.append((report, text))
        print_document_report(report, max(0, args.max_details))

    document_reports = [report for report, _ in documents]
    passed = (
        not processing_errors
        and len(document_reports) == len(pdf_paths)
        and all(report["status"] == "passed" for report in document_reports)
    )
    final_report = {
        "schema": "pdf_formula_extraction_validation_v1",
        "pdf_dir": str(pdf_dir),
        "configuration": {
            "x_tolerance": args.x_tolerance,
            "y_tolerance": args.y_tolerance,
            "reference_check": not args.skip_reference_check,
        },
        "summary": {
            "status": "passed" if passed else "failed",
            "pdf_count": len(pdf_paths),
            "processed_pdf_count": len(document_reports),
            "page_count": sum(report["pages"] for report in document_reports),
            "broken_formula_hit_count": sum(
                report["broken_formula_hit_count"] for report in document_reports
            ),
            "missing_reference_formula_count": sum(
                report["missing_reference_formula_count"] for report in document_reports
            ),
            "unverifiable_pdf_count": sum(
                report["status"] == "unverifiable" for report in document_reports
            ),
        },
        "processing_errors": processing_errors,
        "documents": document_reports,
    }

    if args.output_dir:
        output_dir = args.output_dir.resolve()
        write_outputs(output_dir, documents)
        report_path = output_dir / "formula_validation_report.json"
        report_path.write_text(
            json.dumps(final_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Report: {report_path}")

    print(json.dumps(final_report["summary"], ensure_ascii=False, indent=2))
    if processing_errors:
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
