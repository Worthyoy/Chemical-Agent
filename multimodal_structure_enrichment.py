import json
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


STRUCTURE_PROMPT = """You are an expert chemistry structure-name parser.

For each substrate identifier, extract scaffold and substituents for KG construction.
The identifier can be an IUPAC/common chemical name or a SMILES string.

Rules:
- Use only the provided identifier.
- Do not invent compounds.
- If the identifier is only a symbol/code such as 1a, L1, S3, or unknown, set parseable=false.
- For parseable substrates, scaffold is the core ring system, parent chain, or main functional-group backbone.
- Substituents should include locants/stereochemistry when present.

Return ONLY a JSON array:
[
  {"identifier": "<input identifier>", "parseable": true, "scaffold": "benzene", "substituents": ["4-fluoro"]},
  {"identifier": "<input identifier>", "parseable": false, "scaffold": null, "substituents": []}
]

Substrate identifiers:
{identifiers_block}"""


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "none", "null", "not specified", "not reported", "unknown"}:
        return ""
    return re.sub(r"\s+", " ", text)


def looks_like_symbol(value: str) -> bool:
    text = clean_text(value)
    if not text:
        return True
    return bool(re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?[A-Za-z0-9']{0,3}|[A-Za-z]{1,3}\d{0,3}", text))


def looks_like_smiles(value: str) -> bool:
    text = clean_text(value)
    if not text or " " in text:
        return False
    if re.search(r"[/\\=#\[\]@+]", text):
        return True
    if re.search(r"[cnops]\d", text):
        return True
    if re.search(r"\d.*[cnops]|[cnops].*\d", text):
        return True
    if len(text) >= 3 and re.fullmatch(r"[BCNOFPSIclbrH0-9().-]+", text):
        return True
    return False


def is_generic_identifier(value: str) -> bool:
    text = clean_text(value).casefold()
    if not text:
        return True
    exact = {
        "compound",
        "substrate",
        "product",
        "starting material",
        "unknown",
        "reagent",
        "catalyst",
        "ligand",
        "additive",
    }
    if text in exact:
        return True
    if len(text) < 28 and any(word in text for word in ("derivative", "substituted", "series")):
        return True
    return False


def substrate_identifier_with_source(compound: Dict[str, Any], modality: str) -> Tuple[str, str]:
    if modality == "image":
        name = clean_text(compound.get("name"))
        candidates = [
            ("smiles", clean_text(compound.get("smiles"))),
            ("name", name if looks_like_smiles(name) else ""),
            ("resolved_smiles", clean_text(compound.get("resolved_smiles"))),
            ("iupac_name", clean_text(compound.get("iupac_name"))),
            ("resolved_iupac_name", clean_text(compound.get("resolved_iupac_name"))),
            ("resolved_name_normalized", clean_text(compound.get("resolved_name_normalized"))),
            ("resolved_name", clean_text(compound.get("resolved_name"))),
            ("name", name if not looks_like_smiles(name) else ""),
            ("label", clean_text(compound.get("label"))),
            ("symbol", clean_text(compound.get("symbol"))),
        ]
    else:
        candidates = [
            ("iupac_name", clean_text(compound.get("iupac_name"))),
            ("resolved_iupac_name", clean_text(compound.get("resolved_iupac_name"))),
            ("resolved_name_normalized", clean_text(compound.get("resolved_name_normalized"))),
            ("resolved_name", clean_text(compound.get("resolved_name"))),
            ("name", clean_text(compound.get("name"))),
        ]
    return next(((value, source) for source, value in candidates if value), ("", ""))


def substrate_identifier(compound: Dict[str, Any], modality: str) -> str:
    identifier, _ = substrate_identifier_with_source(compound, modality)
    return identifier


def cache_key(identifier: str) -> str:
    return re.sub(r"\s+", " ", clean_text(identifier)).casefold()


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    write_json(path, cache)


def parse_llm_json(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    if text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("LLM response is not a JSON array")
    return [item for item in data if isinstance(item, dict)]


def call_structure_llm(
    identifiers: List[str],
    client: Any,
    model: str,
    max_retries: int = 3,
) -> Dict[str, Dict[str, Any]]:
    if not identifiers:
        return {}
    block = "\n".join(f"{idx}. {identifier}" for idx, identifier in enumerate(identifiers, start=1))
    prompt = build_structure_prompt(block)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Do not include markdown or explanatory text.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            rows = parse_llm_json(response.choices[0].message.content or "")
            result = {}
            for row in rows:
                identifier = clean_text(row.get("identifier"))
                if not identifier:
                    continue
                substituents = row.get("substituents") or []
                if not isinstance(substituents, list):
                    substituents = []
                result[cache_key(identifier)] = {
                    "parseable": bool(row.get("parseable")),
                    "scaffold": clean_text(row.get("scaffold")) or None,
                    "substituents": [clean_text(item) for item in substituents if clean_text(item)],
                    "structure_parse_source": "llm",
                    "structure_parse_status": "parsed",
                }
            return result
        except Exception as exc:
            if attempt == max_retries - 1:
                return {
                    cache_key(identifier): {
                        "parseable": False,
                        "scaffold": None,
                        "substituents": [],
                        "structure_parse_source": "llm",
                        "structure_parse_status": "error",
                        "structure_parse_error": str(exc),
                    }
                    for identifier in identifiers
                }
            time.sleep(2**attempt)
    return {}


def build_structure_prompt(identifiers_block: str) -> str:
    """Build the LLM prompt without treating example JSON braces as placeholders."""
    return STRUCTURE_PROMPT.replace("{identifiers_block}", identifiers_block)


def collect_needed_identifiers(payloads: List[Tuple[Path, Dict[str, Any], str]], cache: Dict[str, Dict[str, Any]]) -> List[str]:
    needed = {}
    for _, payload, modality in payloads:
        for reaction in payload.get("reactions") or []:
            if not isinstance(reaction, dict):
                continue
            for substrate in reaction.get("substrates") or []:
                if not isinstance(substrate, dict):
                    continue
                if clean_text(substrate.get("scaffold")):
                    continue
                identifier, _ = substrate_identifier_with_source(substrate, modality)
                key = cache_key(identifier)
                if not identifier or key in cache:
                    continue
                if is_generic_identifier(identifier) or (looks_like_symbol(identifier) and not looks_like_smiles(identifier)):
                    cache[key] = {
                        "parseable": False,
                        "scaffold": None,
                        "substituents": [],
                        "structure_parse_source": "prefilter",
                        "structure_parse_status": "skipped_symbol_or_generic",
                    }
                    continue
                needed[key] = identifier
    return sorted(needed.values(), key=str.casefold)


def apply_enrichment(payload: Dict[str, Any], modality: str, cache: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, int]]:
    enriched = deepcopy(payload)
    stats = {
        "substrates_seen": 0,
        "substrates_with_existing_structure": 0,
        "substrates_enriched": 0,
        "substrates_unparseable": 0,
        "substrates_missing_identifier": 0,
    }
    for reaction in enriched.get("reactions") or []:
        if not isinstance(reaction, dict):
            continue
        for substrate in reaction.get("substrates") or []:
            if not isinstance(substrate, dict):
                continue
            stats["substrates_seen"] += 1
            if clean_text(substrate.get("scaffold")):
                stats["substrates_with_existing_structure"] += 1
                continue
            identifier, source = substrate_identifier_with_source(substrate, modality)
            if not identifier:
                stats["substrates_missing_identifier"] += 1
                continue
            parsed = cache.get(cache_key(identifier)) or {}
            substrate["structure_parse_identifier"] = identifier
            substrate["structure_parse_modality"] = modality
            substrate["structure_parse_source"] = source
            if parsed.get("parseable") and clean_text(parsed.get("scaffold")):
                substrate["parseable"] = True
                substrate["scaffold"] = clean_text(parsed.get("scaffold"))
                substrate["substituents"] = [
                    clean_text(item) for item in parsed.get("substituents") or [] if clean_text(item)
                ]
                substrate["structure_parse_status"] = "parsed"
                stats["substrates_enriched"] += 1
            else:
                substrate["structure_parse_status"] = clean_text(parsed.get("structure_parse_status")) or "not_found"
                substrate.setdefault("parseable", False)
                substrate.setdefault("substituents", [])
                stats["substrates_unparseable"] += 1
    return enriched, stats


def enrich_multimodal_structure(
    text_reaction_paths: Iterable[Path | str],
    chemeagle_reaction_paths: Iterable[Path | str],
    output_dir: Path | str,
    cache_path: Path | str,
    client: Any,
    model: str,
    batch_size: int = 30,
    report_path: Path | str | None = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    cache_path = Path(cache_path)
    report_path = Path(report_path) if report_path is not None else output_dir.parent / "multimodal_structure_enrichment_report.json"
    cache = load_cache(cache_path)
    text_paths = [Path(path) for path in text_reaction_paths]
    image_paths = [Path(path) for path in chemeagle_reaction_paths]
    payloads: List[Tuple[Path, Dict[str, Any], str]] = []

    for path in text_paths:
        data = read_json(path)
        if isinstance(data, dict):
            payloads.append((path, data, "text"))
    for path in image_paths:
        data = read_json(path)
        if isinstance(data, dict):
            payloads.append((path, data, "image"))

    needed = collect_needed_identifiers(payloads, cache)
    for start in range(0, len(needed), batch_size):
        batch = needed[start : start + batch_size]
        cache.update(call_structure_llm(batch, client=client, model=model))
        save_cache(cache_path, cache)

    text_output_dir = output_dir / "text"
    image_output_dir = output_dir / "chemeagle"
    text_outputs: List[str] = []
    image_outputs: List[str] = []
    file_reports = []
    total_stats = {
        "identifiers_requested": len(needed),
        "substrates_seen": 0,
        "substrates_with_existing_structure": 0,
        "substrates_enriched": 0,
        "substrates_unparseable": 0,
        "substrates_missing_identifier": 0,
    }

    for path, payload, modality in payloads:
        enriched, stats = apply_enrichment(payload, modality, cache)
        target_dir = text_output_dir if modality == "text" else image_output_dir
        out_path = target_dir / path.name
        write_json(out_path, enriched)
        if modality == "text":
            text_outputs.append(str(out_path))
        else:
            image_outputs.append(str(out_path))
        for key in total_stats:
            if key in stats:
                total_stats[key] += stats[key]
        file_reports.append(
            {
                "input": str(path),
                "output": str(out_path),
                "modality": modality,
                **stats,
            }
        )

    report = {
        "status": "processed",
        "created_at": datetime.now().isoformat(),
        "text_inputs": [str(path) for path in text_paths],
        "chemeagle_inputs": [str(path) for path in image_paths],
        "text_outputs": text_outputs,
        "chemeagle_outputs": image_outputs,
        "cache_path": str(cache_path),
        "model": model,
        "batch_size": batch_size,
        "new_unique_parseable_substrate_identifiers": len(needed),
        "llm_batches": (len(needed) + batch_size - 1) // batch_size if batch_size else 0,
        **total_stats,
        "files": file_reports,
    }
    write_json(report_path, report)
    save_cache(cache_path, cache)
    return report
