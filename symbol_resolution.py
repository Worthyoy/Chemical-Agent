"""Deterministic cross-modal symbol-only resolution helpers."""

import copy
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TEXT_COMPOUND_FIELDS = (
    "substrates",
    "products",
    "intermediates",
    "catalysts",
    "other_components",
    "additives",
    "reagents",
)
IMAGE_COMPOUND_FIELDS = ("reactants", "products")
IMAGE_COMPOUND_ROLES = {
    "reagent",
    "reactant",
    "substrate",
    "catalyst",
    "ligand",
    "precatalyst",
    "base",
    "additive",
}
INVALID_IDENTITY_VALUES = {"", "none", "null", "not specified", "unknown", "n/a"}
UNCERTAIN_MARKERS = ("maybe wrong", "please check", "not sure", "uncertain")
GENERIC_NAMES = {
    "compound",
    "substrate",
    "product",
    "reagent",
    "ligand",
    "catalyst",
    "additive",
    "base",
    "unknown",
    "not specified",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(clean_text(item) for item in value if clean_text(item))
    text = str(value).strip()
    if text.casefold() in INVALID_IDENTITY_VALUES:
        return ""
    return re.sub(r"\s+", " ", text)


def normalize_chemical_name(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u02b9": "\u2032",
        "\u2018": "\u2032",
        "\u2019": "\u2032",
        "\u201b": "\u2032",
        "\u02ba": "\u2033",
        "\u201c": "\u2033",
        "\u201d": "\u2033",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\uff0d": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return clean_text(text)


def normalize_symbol(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).casefold())


def strip_amounts(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"\([^)]*(?:mol\s*%|equiv|mmol|mol|m|mg|g|ml|yield|ee|er|dr|isolated)[^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mol\s*%|equiv|mmol|mol|m|mg|g|ml|%)\b", "", text, flags=re.I)
    return clean_text(text.strip(" ,;"))


def main_token(value: Any) -> str:
    text = strip_amounts(value)
    if not text:
        return ""
    return clean_text(re.split(r"[,;/]", text, maxsplit=1)[0])


def valid_smiles(value: Any) -> str:
    text = clean_text(value)
    if text.casefold() in INVALID_IDENTITY_VALUES or text.casefold() == "<invalid>":
        return ""
    return text


def explicit_name(value: Any, symbol: str = "") -> str:
    name = clean_text(value)
    if not name:
        return ""
    if symbol and normalize_symbol(name) == normalize_symbol(symbol):
        return ""
    if name.casefold() in GENERIC_NAMES:
        return ""
    return name


def has_uncertain_marker(value: Any) -> bool:
    text = clean_text(value).casefold()
    return any(marker in text for marker in UNCERTAIN_MARKERS)


def canonical_paper_key(value: Any) -> str:
    text = clean_text(value)
    stem = Path(text).stem if text else ""
    stem = re.sub(r"_si$", "", stem, flags=re.I)
    stem = re.sub(r"_[0-9a-f]{8}$", "", stem, flags=re.I)
    return stem.casefold()


def identity_key(identity: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        clean_text(identity.get("iupac_name") or identity.get("resolved_iupac_name")).casefold(),
        valid_smiles(identity.get("smiles") or identity.get("resolved_smiles")).casefold(),
        clean_text(identity.get("name") or identity.get("resolved_name")).casefold(),
    )


def text_entity_identity(item: Dict[str, Any]) -> Dict[str, str]:
    symbol = clean_text(item.get("symbol"))
    return {
        "name": explicit_name(item.get("name"), symbol),
        "smiles": valid_smiles(item.get("smiles")),
        "iupac_name": clean_text(item.get("iupac_name")),
        "symbol": symbol,
    }


def image_entity_identity(item: Dict[str, Any]) -> Dict[str, str]:
    symbol = image_symbol(item)
    return {
        "name": explicit_name(item.get("name"), symbol),
        "smiles": valid_smiles(item.get("smiles")),
        "iupac_name": clean_text(item.get("iupac_name")),
        "symbol": symbol,
    }


def image_symbol(item: Dict[str, Any]) -> str:
    return clean_text(item.get("label") or item.get("symbol") or main_token(item.get("text")))


def has_reliable_identity(identity: Dict[str, str]) -> bool:
    return bool(identity.get("iupac_name") or identity.get("smiles") or identity.get("name"))


def is_symbol_only_text_entity(item: Dict[str, Any]) -> bool:
    symbol = clean_text(item.get("symbol"))
    if not symbol:
        return False
    identity = text_entity_identity(item)
    return not has_reliable_identity(identity)


def is_symbol_only_image_entity(item: Dict[str, Any]) -> bool:
    symbol = image_symbol(item)
    if not symbol:
        return False
    if has_uncertain_marker(item.get("label")) or has_uncertain_marker(item.get("text")):
        return False
    identity = image_entity_identity(item)
    return not has_reliable_identity(identity)


def is_compound_like_image_entity(field: str, item: Dict[str, Any]) -> bool:
    if field in IMAGE_COMPOUND_FIELDS:
        return True
    role = clean_text(item.get("refined_role") or item.get("role")).casefold()
    return field in {"conditions", "additional_info"} and role in IMAGE_COMPOUND_ROLES


def iter_text_compounds(payload: Dict[str, Any]):
    for reaction_index, reaction in enumerate(payload.get("reactions") or []):
        if not isinstance(reaction, dict):
            continue
        reaction_id = clean_text(reaction.get("id")) or f"reaction_{reaction_index + 1}"
        for field in TEXT_COMPOUND_FIELDS:
            values = reaction.get(field) or []
            if not isinstance(values, list):
                values = [values]
            for item_index, item in enumerate(values):
                if isinstance(item, dict):
                    yield reaction, reaction_id, field, item_index, item


def iter_image_entities(payload: Any):
    records = payload if isinstance(payload, list) else [payload]
    for record_index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("status") == "error":
            continue
        image_name = clean_text(record.get("image_name") or record.get("image") or record.get("image_path"))
        for reaction_index, reaction in enumerate(record.get("reactions") or []):
            if not isinstance(reaction, dict):
                continue
            reaction_id = clean_text(reaction.get("reaction_id") or reaction.get("id")) or f"reaction_{reaction_index + 1}"
            for field in ("reactants", "products", "conditions", "additional_info"):
                values = reaction.get(field) or []
                if not isinstance(values, list):
                    values = [values]
                for item_index, item in enumerate(values):
                    if isinstance(item, dict):
                        yield record, image_name, reaction, reaction_id, field, item_index, item


def unique_symbol_map(entries: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    unique = {}
    conflicts = {}
    for symbol_key, rows in entries.items():
        identities = {identity_key(row["identity"]) for row in rows}
        if len(identities) == 1:
            unique[symbol_key] = rows[0]
        else:
            conflicts[symbol_key] = rows
    return unique, conflicts


def build_text_symbol_map(payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for symbol, name in (payload.get("name_registry") or {}).items():
        identity = {
            "name": explicit_name(name, symbol),
            "smiles": "",
            "iupac_name": "",
            "symbol": clean_text(symbol),
        }
        symbol_key = normalize_symbol(symbol)
        if symbol_key and has_reliable_identity(identity):
            entries[symbol_key].append({"identity": identity, "source": "name_registry", "raw": {symbol: name}})

    for _, reaction_id, field, item_index, item in iter_text_compounds(payload):
        identity = text_entity_identity(item)
        symbol_key = normalize_symbol(identity.get("symbol"))
        if symbol_key and has_reliable_identity(identity):
            entries[symbol_key].append(
                {
                    "identity": identity,
                    "source": "text_reaction",
                    "reaction_id": reaction_id,
                    "field": field,
                    "item_index": item_index,
                    "raw": copy.deepcopy(item),
                }
            )
    return unique_symbol_map(entries)


def build_image_symbol_map(payload: Any) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    entries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, image_name, _, reaction_id, field, item_index, item in iter_image_entities(payload):
        identity = image_entity_identity(item)
        symbol_key = normalize_symbol(identity.get("symbol"))
        if symbol_key and has_reliable_identity(identity):
            entries[symbol_key].append(
                {
                    "identity": identity,
                    "source": "image_reaction",
                    "image_name": image_name,
                    "reaction_id": reaction_id,
                    "field": field,
                    "item_index": item_index,
                    "raw": copy.deepcopy(item),
                }
            )
    return unique_symbol_map(entries)


def add_resolution_fields(target: Dict[str, Any], source: Dict[str, Any], source_name: str, confidence: str) -> None:
    identity = source["identity"]
    if identity.get("name"):
        target["resolved_name"] = identity["name"]
        normalized_name = normalize_chemical_name(identity["name"])
        if normalized_name:
            target["resolved_name_normalized"] = normalized_name
    if identity.get("smiles"):
        target["resolved_smiles"] = identity["smiles"]
    if identity.get("iupac_name"):
        target["resolved_iupac_name"] = identity["iupac_name"]
    target["resolution_source"] = source_name
    target["resolution_method"] = "same_paper_symbol"
    target["resolution_confidence"] = confidence
    target["resolution_evidence"] = {
        "source": source.get("source"),
        "reaction_id": source.get("reaction_id"),
        "field": source.get("field"),
        "item_index": source.get("item_index"),
        "image_name": source.get("image_name"),
    }


def has_resolution_fields(item: Dict[str, Any]) -> bool:
    return any(
        clean_text(item.get(key))
        for key in ("resolved_name", "resolved_name_normalized", "resolved_smiles", "resolved_iupac_name")
    )


def validate_symbol_resolution(resolved_text: Dict[str, Any], resolved_image: Any, report: Dict[str, Any]) -> Dict[str, Any]:
    candidates = report.get("candidates") or []
    ambiguous_text_symbols = {
        normalize_symbol(row.get("symbol"))
        for row in candidates
        if isinstance(row, dict) and row.get("reason") == "ambiguous_text_symbol_in_same_paper"
    }
    ambiguous_image_symbols = {
        normalize_symbol(row.get("symbol"))
        for row in candidates
        if isinstance(row, dict) and row.get("reason") == "ambiguous_image_symbol_in_same_paper"
    }
    invalid_resolutions: List[Dict[str, Any]] = []
    text_resolved_count = 0
    image_resolved_count = 0

    for _, reaction_id, field, item_index, item in iter_text_compounds(resolved_text):
        if not has_resolution_fields(item):
            continue
        text_resolved_count += 1
        symbol = clean_text(item.get("symbol"))
        symbol_key = normalize_symbol(symbol)
        reasons = []
        if not is_symbol_only_text_entity(item):
            original_name = clean_text(item.get("name"))
            if original_name and normalize_symbol(original_name) != symbol_key:
                reasons.append("text_entity_already_had_name")
        if symbol_key in ambiguous_image_symbols:
            reasons.append("resolved_from_ambiguous_image_symbol")
        if reasons:
            invalid_resolutions.append(
                {
                    "side": "text",
                    "reaction_id": reaction_id,
                    "field": field,
                    "item_index": item_index,
                    "symbol": symbol,
                    "reasons": reasons,
                    "item": copy.deepcopy(item),
                }
            )

    for _, image_name, _, reaction_id, field, item_index, item in iter_image_entities(resolved_image):
        if not has_resolution_fields(item):
            continue
        image_resolved_count += 1
        symbol = image_symbol(item)
        symbol_key = normalize_symbol(symbol)
        reasons = []
        if valid_smiles(item.get("smiles")):
            reasons.append("image_entity_already_had_valid_smiles")
        if clean_text(item.get("iupac_name")):
            reasons.append("image_entity_already_had_iupac_name")
        if not is_compound_like_image_entity(field, item):
            reasons.append("non_compound_image_condition_resolved")
        if has_uncertain_marker(item.get("label")) or has_uncertain_marker(item.get("text")):
            reasons.append("uncertain_image_label_resolved")
        if symbol_key in ambiguous_text_symbols:
            reasons.append("resolved_from_ambiguous_text_symbol")
        if reasons:
            invalid_resolutions.append(
                {
                    "side": "image",
                    "image_name": image_name,
                    "reaction_id": reaction_id,
                    "field": field,
                    "item_index": item_index,
                    "symbol": symbol,
                    "reasons": reasons,
                    "item": copy.deepcopy(item),
                }
            )

    return {
        "status": "success" if not invalid_resolutions else "failed",
        "created_at": datetime.now().isoformat(),
        "resolved_count": text_resolved_count + image_resolved_count,
        "text_resolved_count": text_resolved_count,
        "image_resolved_count": image_resolved_count,
        "candidate_count": report.get("candidate_count", len(candidates)),
        "invalid_resolution_count": len(invalid_resolutions),
        "invalid_resolutions": invalid_resolutions,
        "checks": {
            "reject_existing_image_smiles": True,
            "reject_non_compound_conditions": True,
            "reject_uncertain_image_labels": True,
            "reject_ambiguous_symbols": True,
        },
    }


def candidate(reason: str, symbol: str, target: Dict[str, Any], candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "reason": reason,
        "symbol": symbol,
        "target": copy.deepcopy(target),
        "candidates": copy.deepcopy(candidates or []),
    }


def resolve_cross_modal_symbols(text_payload: Dict[str, Any], image_payload: Any) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:
    resolved_text = copy.deepcopy(text_payload)
    resolved_image = copy.deepcopy(image_payload)

    text_metadata = text_payload.get("metadata") if isinstance(text_payload.get("metadata"), dict) else {}
    text_key = canonical_paper_key(
        text_payload.get("paper_key")
        or text_metadata.get("paper_key")
        or text_payload.get("source")
        or ""
    )
    image_records = resolved_image if isinstance(resolved_image, list) else [resolved_image]
    image_keys = {
        canonical_paper_key(record.get("paper_key") or record.get("pdf_name") or "")
        for record in image_records
        if isinstance(record, dict)
    }
    same_paper = not text_key or not image_keys or text_key in image_keys

    text_map, text_conflicts = build_text_symbol_map(resolved_text)
    image_map, image_conflicts = build_image_symbol_map(resolved_image)
    candidates = []
    text_resolved = 0
    image_resolved = 0

    if same_paper:
        for _, image_name, _, reaction_id, field, item_index, item in iter_image_entities(resolved_image):
            symbol = image_symbol(item)
            symbol_key = normalize_symbol(symbol)
            if not symbol_key:
                continue
            if has_uncertain_marker(item.get("label")) or has_uncertain_marker(item.get("text")):
                candidates.append(candidate("uncertain_image_label", symbol, item))
                continue
            if not is_compound_like_image_entity(field, item) or not is_symbol_only_image_entity(item):
                continue
            if symbol_key in text_map:
                add_resolution_fields(item, text_map[symbol_key], "text_symbol_map", "medium")
                image_resolved += 1
            elif symbol_key in text_conflicts:
                candidates.append(candidate("ambiguous_text_symbol_in_same_paper", symbol, item, text_conflicts[symbol_key]))

        for _, reaction_id, field, item_index, item in iter_text_compounds(resolved_text):
            symbol = clean_text(item.get("symbol"))
            symbol_key = normalize_symbol(symbol)
            if not symbol_key or not is_symbol_only_text_entity(item):
                continue
            if symbol_key in image_map:
                confidence = "medium" if image_map[symbol_key]["identity"].get("smiles") or image_map[symbol_key]["identity"].get("iupac_name") else "low"
                add_resolution_fields(item, image_map[symbol_key], "image_symbol_map", confidence)
                text_resolved += 1
            elif symbol_key in image_conflicts:
                candidates.append(candidate("ambiguous_image_symbol_in_same_paper", symbol, item, image_conflicts[symbol_key]))

    report = {
        "status": "success",
        "created_at": datetime.now().isoformat(),
        "same_paper": same_paper,
        "text_paper_key": text_key,
        "image_paper_keys": sorted(key for key in image_keys if key),
        "text_symbols_unique": len(text_map),
        "image_symbols_unique": len(image_map),
        "text_symbol_conflicts": len(text_conflicts),
        "image_symbol_conflicts": len(image_conflicts),
        "text_entities_resolved": text_resolved,
        "image_entities_resolved": image_resolved,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    report["validation"] = validate_symbol_resolution(resolved_text, resolved_image, report)
    return resolved_text, resolved_image, report


def write_resolution_outputs(
    *,
    text_output_path: Path,
    image_output_path: Path,
    report_path: Path,
    candidates_path: Path,
    resolved_text: Dict[str, Any],
    resolved_image: Any,
    report: Dict[str, Any],
    validation_path: Optional[Path] = None,
) -> None:
    text_output_path.parent.mkdir(parents=True, exist_ok=True)
    image_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    if validation_path is not None:
        validation_path.parent.mkdir(parents=True, exist_ok=True)
    text_output_path.write_text(json.dumps(resolved_text, ensure_ascii=False, indent=2), encoding="utf-8")
    image_output_path.write_text(json.dumps(resolved_image, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps({k: v for k, v in report.items() if k != "candidates"}, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates_path.write_text(json.dumps(report.get("candidates", []), ensure_ascii=False, indent=2), encoding="utf-8")
    if validation_path is not None:
        validation_path.write_text(json.dumps(report.get("validation", {}), ensure_ascii=False, indent=2), encoding="utf-8")
