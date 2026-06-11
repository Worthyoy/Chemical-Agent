import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from gp_extractor import GPExtractor


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = (
    SCRIPT_DIR.parents[2]
    / "反应代表文献"
    / "反应代表文献"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gp_debug_representative_si_existing_rules"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_file_stem(pdf_path: Path) -> str:
    digest = hashlib.md5(str(pdf_path.resolve()).encode("utf-8")).hexdigest()[:8]
    stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pdf_path.stem)
    stem = stem.strip("._") or "pdf"
    return f"{stem[:80]}_{digest}"


def find_si_pdfs(input_root: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in input_root.rglob("*")
            if path.is_file()
            and path.suffix.casefold() == ".pdf"
            and path.name.casefold().startswith("si")
        ),
        key=lambda path: str(path).casefold(),
    )


def reaction_folder_for(pdf_path: Path, input_root: Path) -> str:
    try:
        relative = pdf_path.relative_to(input_root)
    except ValueError:
        return ""
    return relative.parts[0] if len(relative.parts) > 1 else ""


def make_page_cache_record(pdf_path: Path, input_root: Path, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "source": str(pdf_path),
        "reaction_folder": reaction_folder_for(pdf_path, input_root),
        "pdf_name": pdf_path.name,
        "total_pages_with_text": len(pages),
        "pages": pages,
    }


def extract_one(
    extractor: GPExtractor,
    pdf_path: Path,
    input_root: Path,
    page_cache_path: Path,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "source": str(pdf_path),
        "reaction_folder": reaction_folder_for(pdf_path, input_root),
        "pdf_name": pdf_path.name,
        "page_cache_path": str(page_cache_path),
        "total_pages_with_text": 0,
        "gp_count": 0,
        "gp_keys": [],
        "gp_texts": {},
        "gp_metadata": [],
        "error": None,
    }
    try:
        pages = extractor.extract_text_by_pages(str(pdf_path))
        write_json(page_cache_path, make_page_cache_record(pdf_path, input_root, pages))
        gp_texts = extractor.extract_general_procedure_texts(pages)
        record["total_pages_with_text"] = len(pages)
        record["gp_count"] = len(gp_texts)
        record["gp_keys"] = list(gp_texts.keys())
        record["gp_texts"] = gp_texts
        record["gp_metadata"] = [
            {
                key: value
                for key, value in gp_record.items()
                if key not in {"raw_text", "final_text"}
            }
            for gp_record in getattr(extractor, "last_gp_records", [])
        ]
    except Exception as exc:
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    return record


def make_summary(input_root: Path, output_dir: Path, total_pdfs: int, files: List[Dict[str, Any]]) -> Dict[str, Any]:
    processed = sum(1 for item in files if item.get("error") is None)
    failed = len(files) - processed
    total_gp = sum(int(item.get("gp_count") or 0) for item in files)
    files_with_gp = sum(1 for item in files if int(item.get("gp_count") or 0) > 0)

    return {
        "created_at": datetime.now().isoformat(),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "rule_set": "existing_regex_only",
        "total_pdfs": total_pdfs,
        "attempted": len(files),
        "remaining": total_pdfs - len(files),
        "processed": processed,
        "failed": failed,
        "files_with_gp": files_with_gp,
        "files_without_gp": processed - files_with_gp,
        "total_gp": total_gp,
        "files": files,
    }


def build_summary(input_root: Path, output_dir: Path, write_per_file: bool, overwrite: bool) -> Dict[str, Any]:
    extractor = GPExtractor(regex_only=True)
    pdfs = find_si_pdfs(input_root)
    files: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "all_si_gp_results.partial.json"

    print(f"Input root: {input_root}")
    print(f"Output dir: {output_dir}")
    print(f"Found SI PDFs: {len(pdfs)}")

    per_file_dir = output_dir / "per_file"
    page_cache_dir = output_dir / "page_cache"
    for index, pdf_path in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] {pdf_path.name}", flush=True)
        safe_stem = safe_file_stem(pdf_path)
        per_file_path = per_file_dir / f"{safe_stem}.json"
        page_cache_path = page_cache_dir / f"{safe_stem}.pages.json"
        if write_per_file and per_file_path.exists() and page_cache_path.exists() and not overwrite:
            try:
                record = json.loads(per_file_path.read_text(encoding="utf-8"))
                record["page_cache_path"] = str(page_cache_path)
                print("  cache: per-file JSON", flush=True)
            except Exception:
                record = extract_one(extractor, pdf_path, input_root, page_cache_path)
        else:
            if write_per_file and per_file_path.exists() and not page_cache_path.exists() and not overwrite:
                print("  cache miss: page cache", flush=True)
            record = extract_one(extractor, pdf_path, input_root, page_cache_path)
        files.append(record)
        if write_per_file:
            write_json(per_file_path, record)
        write_json(checkpoint_path, make_summary(input_root, output_dir, len(pdfs), files))

    return make_summary(input_root, output_dir, len(pdfs), files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract General Procedure text from representative SI PDFs using existing regex-only rules."
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Root folder containing reaction subfolders. Defaults to the representative literature folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where JSON outputs will be written.",
    )
    parser.add_argument(
        "--no-per-file",
        action="store_true",
        help="Only write all_si_gp_results.json; skip per-file JSON outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess PDFs even when a per-file JSON result already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    summary = build_summary(
        input_root=input_root,
        output_dir=output_dir,
        write_per_file=not args.no_per_file,
        overwrite=args.overwrite,
    )
    output_path = output_dir / "all_si_gp_results.json"
    write_json(output_path, summary)

    print("Done.")
    print(f"Processed: {summary['processed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Files with GP: {summary['files_with_gp']}")
    print(f"Total GP: {summary['total_gp']}")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
