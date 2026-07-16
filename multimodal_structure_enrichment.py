import json
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


STRUCTURE_PROMPT = """You are an expert chemistry structure-name parser.

For each {role_label} identifier, extract scaffold and substituents for KG construction.
The identifier can be an IUPAC/common chemical name or a SMILES string.

Rules:
- Use only the provided identifier.
- Do not invent compounds.
- If the identifier is only a symbol/code such as 1a, L1, S3, or unknown, set parseable=false.
- For parseable identifiers, scaffold is the core ring system, parent chain, or main functional-group backbone.
- {substituent_rule}

Return ONLY a JSON array:
[
  {"identifier": "<input identifier>", "parseable": true, "scaffold": "benzene", "substituents": ["4-fluoro"]},
  {"identifier": "<input identifier>", "parseable": false, "scaffold": null, "substituents": []}
]

{role_title} identifiers:
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


def normalize_cache_row(row: Dict[str, Any]) -> Dict[str, Any]:
    substituents = row.get("substituents") or []
    if not isinstance(substituents, list):
        substituents = []
    normalized = {
        "parseable": bool(row.get("parseable")),
        "scaffold": clean_text(row.get("scaffold")) or None,
        "substituents": [clean_text(item) for item in substituents if clean_text(item)],
    }
    for key in (
        "structure_parse_source",
        "structure_parse_status",
        "structure_parse_scope",
        "structure_parse_error",
    ):
        if row.get(key) not in (None, ""):
            normalized[key] = row.get(key)
    return normalized


def load_single_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        cache_key(key): normalize_cache_row(value if isinstance(value, dict) else {})
        for key, value in data.items()
        if cache_key(key)
    }


def legacy_cache_paths(path: Path) -> List[Path]:
    return [
        path.parent / "multimodal_structure_parse_cache.json",
        path.parent / "substrate_parse_cache.json",
    ]


def load_cache(path: Path, include_legacy: bool = True) -> Dict[str, Dict[str, Any]]:
    sources = [path]
    if include_legacy:
        sources.extend(candidate for candidate in legacy_cache_paths(path) if candidate != path)
    merged: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        for key, row in load_single_cache(source).items():
            existing = merged.get(key)
            if existing and existing.get("scaffold"):
                continue
            merged[key] = row
    return merged


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
    role: str = "substrate",
    max_retries: int = 3,
) -> Dict[str, Dict[str, Any]]:
    if not identifiers:
        return {}
    block = "\n".join(f"{idx}. {identifier}" for idx, identifier in enumerate(identifiers, start=1))
    prompt = build_structure_prompt(block, role=role)
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
                    "substituents": (
                        [clean_text(item) for item in substituents if clean_text(item)]
                        if role == "substrate"
                        else []
                    ),
                    "structure_parse_source": "llm",
                    "structure_parse_status": "parsed",
                    "structure_parse_scope": (
                        "substrate_full" if role == "substrate" else "product_scaffold"
                    ),
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
                        "structure_parse_scope": (
                            "substrate_full" if role == "substrate" else "product_scaffold"
                        ),
                        "structure_parse_error": str(exc),
                    }
                    for identifier in identifiers
                }
            time.sleep(2**attempt)
    return {}


def build_structure_prompt(identifiers_block: str, role: str = "substrate") -> str:
    """Build the LLM prompt without treating example JSON braces as placeholders."""
    if role == "product":
        replacements = {
            "{role_label}": "product",
            "{role_title}": "Product",
            "{substituent_rule}": "Products only need scaffold/class for benchmark context; return substituents as an empty list.",
            "{identifiers_block}": identifiers_block,
        }
    else:
        replacements = {
            "{role_label}": "substrate",
            "{role_title}": "Substrate",
            "{substituent_rule}": "Substituents should include locants/stereochemistry when present.",
            "{identifiers_block}": identifiers_block,
        }
    prompt = STRUCTURE_PROMPT
    for old, new in replacements.items():
        prompt = prompt.replace(old, new)
    return prompt


def identifier_with_source(compound: Dict[str, Any], modality: str) -> Tuple[str, str]:
    return substrate_identifier_with_source(compound, modality)


def cache_entry_satisfies_role(row: Dict[str, Any], role: str) -> bool:
    if not row:
        return False
    if not row.get("parseable"):
        return True
    if not clean_text(row.get("scaffold")):
        return False
    if role == "product":
        return True
    return row.get("structure_parse_scope") != "product_scaffold"


def collect_needed_identifiers(
    payloads: List[Tuple[Path, Dict[str, Any], str]],
    cache: Dict[str, Dict[str, Any]],
    roles: Iterable[str],
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    needed = {"substrates": {}, "products": {}}
    stats = {
        "cache_hits": 0,
        "skipped_existing_structure": 0,
        "skipped_missing_identifier": 0,
    }
    role_to_payload_key = {"substrates": "substrates", "products": "products"}
    for _, payload, modality in payloads:
        for reaction in payload.get("reactions") or []:
            if not isinstance(reaction, dict):
                continue
            for role in roles:
                payload_key = role_to_payload_key.get(role)
                if not payload_key:
                    continue
                for compound in reaction.get(payload_key) or []:
                    if not isinstance(compound, dict):
                        continue
                    if clean_text(compound.get("scaffold")):
                        stats["skipped_existing_structure"] += 1
                        continue
                    identifier, _ = identifier_with_source(compound, modality)
                    if not identifier:
                        stats["skipped_missing_identifier"] += 1
                        continue
                    key = cache_key(identifier)
                    cache_row = cache.get(key)
                    if cache_entry_satisfies_role(cache_row or {}, "product" if role == "products" else "substrate"):
                        stats["cache_hits"] += 1
                        continue
                    if is_generic_identifier(identifier) or (
                        looks_like_symbol(identifier) and not looks_like_smiles(identifier)
                    ):
                        cache[key] = {
                            "parseable": False,
                            "scaffold": None,
                            "substituents": [],
                            "structure_parse_source": "prefilter",
                            "structure_parse_status": "skipped_symbol_or_generic",
                            "structure_parse_scope": "prefilter",
                        }
                        continue
                    if role == "substrates":
                        needed["substrates"][key] = identifier
                    elif key not in needed["substrates"]:
                        needed["products"][key] = identifier
    return (
        {
            role: sorted(values.values(), key=str.casefold)
            for role, values in needed.items()
        },
        stats,
    )


def apply_enrichment(
    payload: Dict[str, Any],
    modality: str,
    cache: Dict[str, Dict[str, Any]],
    roles: Iterable[str],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    enriched = deepcopy(payload)
    stats = {
        "substrates_seen": 0,
        "substrates_with_existing_structure": 0,
        "substrates_enriched": 0,
        "substrates_unparseable": 0,
        "substrates_missing_identifier": 0,
        "products_seen": 0,
        "products_with_existing_scaffold": 0,
        "products_enriched": 0,
        "products_unparseable": 0,
        "products_missing_identifier": 0,
    }
    for reaction in enriched.get("reactions") or []:
        if not isinstance(reaction, dict):
            continue
        if "substrates" in roles:
            for substrate in reaction.get("substrates") or []:
                if not isinstance(substrate, dict):
                    continue
                stats["substrates_seen"] += 1
                if clean_text(substrate.get("scaffold")):
                    stats["substrates_with_existing_structure"] += 1
                    continue
                identifier, source = identifier_with_source(substrate, modality)
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
        if "products" in roles:
            for product in reaction.get("products") or []:
                if not isinstance(product, dict):
                    continue
                stats["products_seen"] += 1
                if clean_text(product.get("scaffold")):
                    stats["products_with_existing_scaffold"] += 1
                    continue
                identifier, source = identifier_with_source(product, modality)
                if not identifier:
                    stats["products_missing_identifier"] += 1
                    continue
                parsed = cache.get(cache_key(identifier)) or {}
                product["structure_parse_identifier"] = identifier
                product["structure_parse_modality"] = modality
                product["structure_parse_source"] = source
                if parsed.get("parseable") and clean_text(parsed.get("scaffold")):
                    product["parseable"] = True
                    product["scaffold"] = clean_text(parsed.get("scaffold"))
                    product.setdefault("substituents", [])
                    product["structure_parse_status"] = "parsed"
                    stats["products_enriched"] += 1
                else:
                    product["structure_parse_status"] = clean_text(parsed.get("structure_parse_status")) or "not_found"
                    product.setdefault("parseable", False)
                    product.setdefault("substituents", [])
                    stats["products_unparseable"] += 1
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
    roles: Iterable[str] = ("substrates", "products"),
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    cache_path = Path(cache_path)
    report_path = Path(report_path) if report_path is not None else output_dir.parent / "multimodal_structure_enrichment_report.json"
    roles = tuple(roles)
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

    needed_by_role, pre_stats = collect_needed_identifiers(payloads, cache, roles)
    substrate_needed = needed_by_role.get("substrates", [])
    product_needed = needed_by_role.get("products", [])
    for role_name, identifiers in (
        ("substrate", substrate_needed),
        ("product", product_needed),
    ):
        for start in range(0, len(identifiers), batch_size):
            batch = identifiers[start : start + batch_size]
            cache.update(call_structure_llm(batch, client=client, model=model, role=role_name))
        save_cache(cache_path, cache)

    text_output_dir = output_dir / "text"
    image_output_dir = output_dir / "chemeagle"
    text_outputs: List[str] = []
    image_outputs: List[str] = []
    file_reports = []
    total_stats = {
        "substrates_seen": 0,
        "substrates_with_existing_structure": 0,
        "substrates_enriched": 0,
        "substrates_unparseable": 0,
        "substrates_missing_identifier": 0,
        "products_seen": 0,
        "products_with_existing_scaffold": 0,
        "products_enriched": 0,
        "products_unparseable": 0,
        "products_missing_identifier": 0,
        **pre_stats,
    }

    for path, payload, modality in payloads:
        enriched, stats = apply_enrichment(payload, modality, cache, roles)
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
        "roles_processed": list(roles),
        "new_unique_parseable_substrate_identifiers": len(substrate_needed),
        "substrate_llm_identifiers": len(substrate_needed),
        "product_scaffold_llm_identifiers": len(product_needed),
        "identifiers_requested": len(substrate_needed) + len(product_needed),
        "llm_batches": (
            ((len(substrate_needed) + batch_size - 1) // batch_size)
            + ((len(product_needed) + batch_size - 1) // batch_size)
            if batch_size
            else 0
        ),
        **total_stats,
        "files": file_reports,
    }
    write_json(report_path, report)
    save_cache(cache_path, cache)
    return report
