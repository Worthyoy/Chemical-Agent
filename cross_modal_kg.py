import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


TEXT_COMPOUND_FIELDS = ("substrates", "products", "catalysts", "additives", "reagents")
CONDITION_ROLES = {
    "solvent",
    "temperature",
    "time",
    "concentration",
    "volume",
    "atmosphere",
    "light",
    "light source",
    "light_source",
    "wavelength",
}
COMPOUND_LIKE_ROLES = {
    "reagent",
    "reagents",
    "catalyst",
    "catalysts",
    "base",
    "additive",
    "additives",
    "ligand",
    "precatalyst",
}
TARGET_ROLES = {"yield", "ee", "er", "dr"}
UNCERTAIN_TEXT_MARKERS = ("maybe wrong", "please check", "not sure", "uncertain")
GENERIC_NAMES = {"compound", "substrate", "product", "reagent", "ligand", "catalyst", "additive", "base", "unknown"}

EDGE_FIELDNAMES = [
    "x_name",
    "x_type",
    "relationship",
    "y_name",
    "y_type",
    "pdf_name",
    "yield",
    "ee",
    "er",
    "dr",
    "source_modality",
    "source_image",
    "text_reaction_id",
    "image_reaction_id",
    "match_method",
    "confidence",
    "evidence_type",
    "text_role",
    "image_role",
    "original_role",
    "refined_role",
    "role_source",
    "role_confidence",
    "resolution_source",
    "resolution_method",
    "resolution_confidence",
    "text_iupac_name",
    "image_smiles",
    "image_iupac_name",
]


@dataclass
class Entity:
    entity_id: str
    name: str
    source_modality: str
    source_paper: str
    reaction_id: str
    role: str
    raw: Dict[str, Any]
    symbol: str = ""
    iupac_name: str = ""
    smiles: str = ""
    source_image: str = ""


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(clean_text(item) for item in value if clean_text(item))
    text = str(value).strip()
    if text.casefold() in {"", "none", "null", "not specified"}:
        return ""
    return re.sub(r"\s+", " ", text)


def normalize_iupac_name(value: Any) -> str:
    text = clean_text(value).casefold()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_symbol(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).casefold())


def strip_amounts(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"\([^)]*(?:mol\s*%|equiv|mmol|mol|m|mg|g|ml|yield|ee|er|dr|isolated)[^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mol\s*%|equiv|mmol|mol|m|mg|g|ml|%)\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:the label maybe wrong|please check the image again)\b", "", text, flags=re.I)
    return clean_text(text.strip(" ,;"))


def main_token(value: Any) -> str:
    text = strip_amounts(clean_text(value))
    if not text:
        return ""
    text = re.split(r"[,;/]", text, maxsplit=1)[0]
    return clean_text(text)


def is_uncertain_text(value: Any) -> bool:
    text = clean_text(value).casefold()
    return any(marker in text for marker in UNCERTAIN_TEXT_MARKERS)


def paper_name_from_source(source: Any, fallback: str) -> str:
    raw = clean_text(source) or fallback
    return Path(raw).stem.replace("_si", "")


def looks_like_smiles(value: Any) -> bool:
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


def text_compound_display_name(item: Dict[str, Any]) -> str:
    return (
        clean_text(item.get("iupac_name"))
        or reliable_name(item)
        or clean_text(item.get("resolved_iupac_name"))
        or clean_text(item.get("resolved_name_normalized"))
        or clean_text(item.get("resolved_name"))
        or clean_text(item.get("symbol"))
        or clean_text(item.get("label"))
        or clean_text(item.get("smiles"))
        or clean_text(item.get("resolved_smiles"))
        or strip_amounts(clean_text(item.get("text")))
    )


def image_compound_display_name(item: Dict[str, Any]) -> str:
    raw_name = reliable_name(item)
    name_as_smiles = raw_name if looks_like_smiles(raw_name) else ""
    name_as_text = "" if name_as_smiles else raw_name
    return (
        clean_text(item.get("smiles"))
        or name_as_smiles
        or clean_text(item.get("resolved_smiles"))
        or clean_text(item.get("iupac_name"))
        or clean_text(item.get("resolved_iupac_name"))
        or clean_text(item.get("resolved_name_normalized"))
        or clean_text(item.get("resolved_name"))
        or name_as_text
        or strip_amounts(clean_text(item.get("text")))
        or clean_text(item.get("label"))
        or clean_text(item.get("symbol"))
    )


def compound_display_name(item: Dict[str, Any]) -> str:
    return text_compound_display_name(item)


def reliable_name(item: Dict[str, Any]) -> str:
    name = clean_text(item.get("name"))
    if not name:
        return ""
    symbol = clean_text(item.get("symbol") or item.get("label"))
    if symbol and normalize_symbol(name) == normalize_symbol(symbol):
        return ""
    if name.casefold() in GENERIC_NAMES:
        return ""
    return name


def clean_role(role: Any) -> str:
    return clean_text(role).casefold().replace("_", " ")


def condition_original_role(condition: Dict[str, Any]) -> str:
    return clean_role(condition.get("original_role") or condition.get("role"))


def condition_refined_role(condition: Dict[str, Any]) -> str:
    return clean_role(condition.get("refined_role"))


def effective_condition_role(condition: Dict[str, Any]) -> str:
    refined = condition_refined_role(condition)
    original = condition_original_role(condition)
    confidence = clean_text(condition.get("role_confidence")).casefold()
    if refined and refined != "unknown" and confidence != "low":
        return refined
    if original:
        return original
    return "reagent"


def normalize_entity(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return item
    if item is None:
        return {}
    return {"name": str(item)}


def target_values(reaction: Dict[str, Any]) -> Dict[str, str]:
    targets = reaction.get("targets") or {}
    if not isinstance(targets, dict):
        targets = {}
    return {
        "yield": clean_text(targets.get("yield")),
        "ee": clean_text(targets.get("ee")),
        "er": clean_text(targets.get("er")),
        "dr": clean_text(targets.get("dr")),
    }


def make_triple(
    x_name: str,
    x_type: str,
    relationship: str,
    y_name: str,
    y_type: str,
    pdf_name: str,
    yield_value: str = "",
    ee: str = "",
    er: str = "",
    dr: str = "",
    source_modality: str = "",
    source_image: str = "",
    text_reaction_id: str = "",
    image_reaction_id: str = "",
    match_method: str = "",
    confidence: str = "",
    evidence_type: str = "",
    text_role: str = "",
    image_role: str = "",
    original_role: str = "",
    refined_role: str = "",
    role_source: str = "",
    role_confidence: str = "",
    resolution_source: str = "",
    resolution_method: str = "",
    resolution_confidence: str = "",
    text_iupac_name: str = "",
    image_smiles: str = "",
    image_iupac_name: str = "",
) -> Dict[str, str]:
    return {
        "x_name": clean_text(x_name),
        "x_type": clean_text(x_type),
        "relationship": clean_text(relationship),
        "y_name": clean_text(y_name),
        "y_type": clean_text(y_type),
        "pdf_name": clean_text(pdf_name),
        "yield": clean_text(yield_value),
        "ee": clean_text(ee),
        "er": clean_text(er),
        "dr": clean_text(dr),
        "source_modality": clean_text(source_modality),
        "source_image": clean_text(source_image),
        "text_reaction_id": clean_text(text_reaction_id),
        "image_reaction_id": clean_text(image_reaction_id),
        "match_method": clean_text(match_method),
        "confidence": clean_text(confidence),
        "evidence_type": clean_text(evidence_type),
        "text_role": clean_text(text_role),
        "image_role": clean_text(image_role),
        "original_role": clean_text(original_role),
        "refined_role": clean_text(refined_role),
        "role_source": clean_text(role_source),
        "role_confidence": clean_text(role_confidence),
        "resolution_source": clean_text(resolution_source),
        "resolution_method": clean_text(resolution_method),
        "resolution_confidence": clean_text(resolution_confidence),
        "text_iupac_name": clean_text(text_iupac_name),
        "image_smiles": clean_text(image_smiles),
        "image_iupac_name": clean_text(image_iupac_name),
    }


def add_unique(triples: List[Dict[str, str]], seen: Set[Tuple[str, ...]], triple: Dict[str, str]) -> None:
    if not triple["x_name"] or not triple["y_name"]:
        return
    key = tuple(triple.get(field, "") for field in EDGE_FIELDNAMES)
    if key in seen:
        return
    seen.add(key)
    triples.append(triple)


def iter_text_reaction_payloads(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in paths:
        data = read_json(path)
        if isinstance(data, dict):
            yield path, data


def iter_chemeagle_records(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in paths:
        data = read_json(path)
        records = data if isinstance(data, list) else [data]
        for record in records:
            if isinstance(record, dict):
                yield path, record


def iter_image_reaction_payloads(paths: Iterable[Path]) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    for path in paths:
        data = read_json(path)
        if isinstance(data, dict) and isinstance(data.get("reactions"), list):
            yield path, data


def condition_summary_from_text(reaction: Dict[str, Any]) -> str:
    conditions = reaction.get("conditions") or {}
    if not isinstance(conditions, dict):
        return "not specified"
    parts = []
    role_labels = {
        "solvent": "solvent",
        "temperature": "temp",
        "time": "time",
        "concentration": "conc",
        "volume": "vol",
        "atmosphere": "atm",
        "light source": "light",
        "light_source": "light",
        "wavelength": "wl",
    }
    for role, label in role_labels.items():
        value = clean_text(conditions.get(role))
        if value:
            parts.append(f"{label}: {value}")
    return ", ".join(parts) if parts else "not specified"


def extract_reaction_types(reaction: Dict[str, Any]) -> List[str]:
    value = clean_text(reaction.get("reaction_type"))
    return [value or "unknown reaction"]


def build_text_reaction_triples(text_paths: Iterable[Path]) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    seen: Set[Tuple[str, ...]] = set()

    for path, data in iter_text_reaction_payloads(text_paths):
        source_paper = paper_name_from_source(data.get("source"), path.stem)
        reactions = data.get("reactions") or []
        for rxn_index, raw_rxn in enumerate(reactions, start=1):
            if not isinstance(raw_rxn, dict):
                continue
            reaction_id = clean_text(raw_rxn.get("id")) or f"text_reaction_{rxn_index}"
            reaction_node = f"TextReaction:{source_paper}:{reaction_id}"
            targets = target_values(raw_rxn)

            def add(
                relationship: str,
                y_name: str,
                y_type: str,
                text_role: str = "",
                entity: Optional[Dict[str, Any]] = None,
            ) -> None:
                add_unique(
                    triples,
                    seen,
                    make_triple(
                        reaction_node,
                        "Reaction",
                        relationship,
                        y_name,
                        y_type,
                        source_paper,
                        targets["yield"],
                        targets["ee"],
                        targets["er"],
                        targets["dr"],
                        source_modality="text",
                        text_reaction_id=reaction_id,
                        evidence_type="text_extraction",
                        text_role=text_role,
                        resolution_source=clean_text(entity.get("resolution_source")) if entity else "",
                        resolution_method=clean_text(entity.get("resolution_method")) if entity else "",
                        resolution_confidence=clean_text(entity.get("resolution_confidence")) if entity else "",
                    ),
                )

            add("REPORTED_IN", source_paper, "Paper")
            add("HAS_CONDITION", condition_summary_from_text(raw_rxn), "Condition", "condition")
            for reaction_type in extract_reaction_types(raw_rxn):
                add("OF_TYPE", reaction_type, "ReactionType")

            field_specs = {
                "substrates": ("USES_SUBSTRATE", "Substrate", "substrate"),
                "products": ("PRODUCES", "Product", "product"),
                "catalysts": ("USES_CATALYST", "Catalyst", "catalyst"),
                "additives": ("USES_ADDITIVE", "Additive", "additive"),
                "reagents": ("USES_REAGENT", "Reagent", "reagent"),
            }
            for field, (relationship, node_type, role) in field_specs.items():
                values = raw_rxn.get(field) or []
                if not isinstance(values, list):
                    values = [values]
                for item in values:
                    comp = normalize_entity(item)
                    name = text_compound_display_name(comp) or "unknown"
                    add(relationship, name, node_type, role, comp)
                    if field != "substrates":
                        continue
                    scaffold = clean_text(comp.get("scaffold"))
                    if scaffold:
                        add_unique(
                            triples,
                            seen,
                            make_triple(
                                name,
                                "Substrate",
                                "HAS_SCAFFOLD",
                                scaffold,
                                "SubstrateScaffold",
                                source_paper,
                                targets["yield"],
                                targets["ee"],
                                targets["er"],
                                targets["dr"],
                                source_modality="text",
                                text_reaction_id=reaction_id,
                                evidence_type="text_extraction",
                                text_role="substrate",
                                resolution_source=clean_text(comp.get("resolution_source")),
                                resolution_method=clean_text(comp.get("resolution_method")),
                                resolution_confidence=clean_text(comp.get("resolution_confidence")),
                            ),
                        )
                    for substituent in comp.get("substituents") or []:
                        substituent_name = clean_text(substituent)
                        if substituent_name:
                            add_unique(
                                triples,
                                seen,
                                make_triple(
                                    name,
                                    "Substrate",
                                    "HAS_SUBSTITUENT",
                                    substituent_name,
                                    "Substituent",
                                    source_paper,
                                    targets["yield"],
                                    targets["ee"],
                                    targets["er"],
                                    targets["dr"],
                                    source_modality="text",
                                    text_reaction_id=reaction_id,
                                    evidence_type="text_extraction",
                                    text_role="substrate",
                                    resolution_source=clean_text(comp.get("resolution_source")),
                                    resolution_method=clean_text(comp.get("resolution_method")),
                                    resolution_confidence=clean_text(comp.get("resolution_confidence")),
                                ),
                            )
    return triples


def extract_text_compounds(paths: Iterable[Path]) -> List[Entity]:
    entities: List[Entity] = []
    for path, data in iter_text_reaction_payloads(paths):
        source_paper = paper_name_from_source(data.get("source"), path.stem)
        for rxn_index, rxn in enumerate(data.get("reactions") or [], start=1):
            if not isinstance(rxn, dict):
                continue
            reaction_id = clean_text(rxn.get("id")) or f"text_reaction_{rxn_index}"
            for field in TEXT_COMPOUND_FIELDS:
                values = rxn.get(field) or []
                if not isinstance(values, list):
                    values = [values]
                for comp_index, item in enumerate(values, start=1):
                    comp = normalize_entity(item)
                    name = text_compound_display_name(comp)
                    if not name:
                        continue
                    entities.append(
                        Entity(
                            entity_id=f"text:{source_paper}:{reaction_id}:{field}:{comp_index}",
                            name=name,
                            iupac_name=clean_text(comp.get("iupac_name")) or reliable_name(comp) or clean_text(comp.get("resolved_iupac_name")) or name,
                            symbol=clean_text(comp.get("symbol")),
                            smiles=clean_text(comp.get("smiles")) or clean_text(comp.get("resolved_smiles")),
                            source_modality="text",
                            source_paper=source_paper,
                            reaction_id=reaction_id,
                            role=field.rstrip("s"),
                            raw=comp,
                        )
                    )
    return entities


def build_unique_text_symbol_map(text_entities: List[Entity]) -> Dict[str, Dict[str, Entity]]:
    candidates: Dict[str, Dict[str, List[Entity]]] = {}
    for entity in text_entities:
        symbol_key = normalize_symbol(entity.symbol)
        name_key = normalize_iupac_name(entity.name)
        if not symbol_key or not name_key:
            continue
        candidates.setdefault(entity.source_paper, {}).setdefault(symbol_key, []).append(entity)

    result: Dict[str, Dict[str, Entity]] = {}
    for paper, symbols in candidates.items():
        result[paper] = {}
        for symbol_key, entities in symbols.items():
            normalized_names = {normalize_iupac_name(entity.name) for entity in entities if normalize_iupac_name(entity.name)}
            if len(normalized_names) == 1:
                result[paper][symbol_key] = entities[0]
    return result


def image_record_is_usable(record: Dict[str, Any]) -> bool:
    if record.get("status") == "error":
        return False
    reactions = record.get("reactions")
    return isinstance(reactions, list) and bool(reactions)


def image_targets_from_conditions(conditions: List[Dict[str, Any]]) -> Dict[str, str]:
    targets = {"yield": "", "ee": "", "er": "", "dr": ""}
    for condition in conditions:
        role = effective_condition_role(condition)
        if role in targets and not targets[role]:
            targets[role] = clean_text(condition.get("text") or condition.get("value") or condition.get("label"))
    return targets


def split_image_conditions(conditions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    compound_like = []
    condition_like = []
    targets = {"yield": "", "ee": "", "er": "", "dr": ""}
    for condition in conditions:
        if not isinstance(condition, dict):
            continue
        role = effective_condition_role(condition)
        if role in TARGET_ROLES:
            if role in targets and not targets[role]:
                targets[role] = clean_text(condition.get("text") or condition.get("value") or condition.get("label"))
            continue
        if role in COMPOUND_LIKE_ROLES:
            compound_like.append(condition)
        elif role in CONDITION_ROLES:
            condition_like.append(condition)
        elif image_compound_display_name(condition):
            compound_like.append(condition)
    return compound_like, condition_like, targets


def image_condition_summary(condition_like: List[Dict[str, Any]]) -> str:
    parts = []
    for condition in condition_like:
        role = effective_condition_role(condition) or "condition"
        value = clean_text(condition.get("text") or condition.get("value") or condition.get("label") or condition.get("iupac_name"))
        if value:
            parts.append(f"{role}: {value}")
    return ", ".join(parts) if parts else "not specified"


def image_role_to_edge(role: str) -> Tuple[str, str]:
    normalized = clean_role(role)
    if normalized in {"catalyst", "catalysts", "ligand", "precatalyst"}:
        return "USES_CATALYST", "Catalyst"
    if normalized in {"base", "additive", "additives"}:
        return "USES_ADDITIVE", "Additive"
    return "USES_REAGENT", "Reagent"


def build_image_reaction_triples(chemeagle_paths: Iterable[Path]) -> List[Dict[str, str]]:
    triples: List[Dict[str, str]] = []
    seen: Set[Tuple[str, ...]] = set()

    for path, data in iter_image_reaction_payloads(chemeagle_paths):
        default_paper = paper_name_from_source(data.get("source"), path.stem)
        for rxn_index, rxn in enumerate(data.get("reactions") or [], start=1):
            if not isinstance(rxn, dict):
                continue
            source_paper = paper_name_from_source(rxn.get("source_paper") or data.get("source"), default_paper)
            source_image = clean_text(rxn.get("source_image") or rxn.get("image_name") or rxn.get("image"))
            image_node = source_image or f"{source_paper}:unknown_image"
            image_reaction_id = clean_text(rxn.get("id") or rxn.get("reaction_id")) or f"image_reaction_{rxn_index}"
            reaction_node = f"ImageReaction:{source_paper}:{image_node}:{image_reaction_id}"
            targets = target_values(rxn)

            def add(
                relationship: str,
                y_name: str,
                y_type: str,
                image_role: str = "",
                entity: Optional[Dict[str, Any]] = None,
            ) -> None:
                add_unique(
                    triples,
                    seen,
                    make_triple(
                        reaction_node,
                        "Reaction",
                        relationship,
                        y_name,
                        y_type,
                        source_paper,
                        targets["yield"],
                        targets["ee"],
                        targets["er"],
                        targets["dr"],
                        source_modality="image",
                        source_image=source_image,
                        image_reaction_id=image_reaction_id,
                        evidence_type="chemeagle_extraction",
                        image_role=image_role,
                        original_role=condition_original_role(entity) if entity else "",
                        refined_role=condition_refined_role(entity) if entity else "",
                        role_source=clean_text(entity.get("role_source")) if entity else "",
                        role_confidence=clean_text(entity.get("role_confidence")) if entity else "",
                        resolution_source=clean_text(entity.get("resolution_source")) if entity else "",
                        resolution_method=clean_text(entity.get("resolution_method")) if entity else "",
                        resolution_confidence=clean_text(entity.get("resolution_confidence")) if entity else "",
                    ),
                )

            add("REPORTED_IN", source_paper, "Paper")
            add("EXTRACTED_FROM_IMAGE", image_node, "Image")
            add("HAS_CONDITION", condition_summary_from_text(rxn), "Condition", "condition")
            for reaction_type in extract_reaction_types(rxn):
                add("OF_TYPE", reaction_type, "ReactionType")

            field_specs = {
                "substrates": ("USES_SUBSTRATE", "Substrate", "substrate"),
                "products": ("PRODUCES", "Product", "product"),
                "catalysts": ("USES_CATALYST", "Catalyst", "catalyst"),
                "additives": ("USES_ADDITIVE", "Additive", "additive"),
                "reagents": ("USES_REAGENT", "Reagent", "reagent"),
            }
            for field, (relationship, node_type, role) in field_specs.items():
                values = rxn.get(field) or []
                if not isinstance(values, list):
                    values = [values]
                for item in values:
                    comp = normalize_entity(item)
                    name = image_compound_display_name(comp) or "unknown"
                    add(relationship, name, node_type, role, comp)
                    if field != "substrates":
                        continue
                    scaffold = clean_text(comp.get("scaffold"))
                    if scaffold:
                        add_unique(
                            triples,
                            seen,
                            make_triple(
                                name,
                                "Substrate",
                                "HAS_SCAFFOLD",
                                scaffold,
                                "SubstrateScaffold",
                                source_paper,
                                targets["yield"],
                                targets["ee"],
                                targets["er"],
                                targets["dr"],
                                source_modality="image",
                                source_image=source_image,
                                image_reaction_id=image_reaction_id,
                                evidence_type="chemeagle_extraction",
                                image_role="substrate",
                                resolution_source=clean_text(comp.get("resolution_source")),
                                resolution_method=clean_text(comp.get("resolution_method")),
                                resolution_confidence=clean_text(comp.get("resolution_confidence")),
                            ),
                        )
                    for substituent in comp.get("substituents") or []:
                        substituent_name = clean_text(substituent)
                        if substituent_name:
                            add_unique(
                                triples,
                                seen,
                                make_triple(
                                    name,
                                    "Substrate",
                                    "HAS_SUBSTITUENT",
                                    substituent_name,
                                    "Substituent",
                                    source_paper,
                                    targets["yield"],
                                    targets["ee"],
                                    targets["er"],
                                    targets["dr"],
                                    source_modality="image",
                                    source_image=source_image,
                                    image_reaction_id=image_reaction_id,
                                    evidence_type="chemeagle_extraction",
                                    image_role="substrate",
                                    resolution_source=clean_text(comp.get("resolution_source")),
                                    resolution_method=clean_text(comp.get("resolution_method")),
                                    resolution_confidence=clean_text(comp.get("resolution_confidence")),
                                ),
                            )
    return triples


def make_image_compound(
    source_paper: str,
    source_image: str,
    reaction_id: str,
    role: str,
    raw: Dict[str, Any],
    index: int,
) -> Entity:
    iupac = clean_text(raw.get("iupac_name"))
    resolved_iupac = clean_text(raw.get("resolved_iupac_name"))
    smiles = clean_text(raw.get("smiles"))
    resolved_smiles = clean_text(raw.get("resolved_smiles"))
    raw_name = reliable_name(raw)
    if not smiles and looks_like_smiles(raw_name):
        smiles = raw_name
    label = clean_text(raw.get("label") or raw.get("symbol"))
    name = image_compound_display_name(raw)
    return Entity(
        entity_id=f"image:{source_paper}:{source_image}:{reaction_id}:{role}:{index}",
        name=name,
        iupac_name=iupac or resolved_iupac,
        smiles=smiles or resolved_smiles,
        symbol=label,
        source_modality="image",
        source_paper=source_paper,
        source_image=source_image,
        reaction_id=reaction_id,
        role=role,
        raw=raw,
    )


def extract_chemeagle_compounds(paths: Iterable[Path]) -> List[Entity]:
    entities: List[Entity] = []
    for path, data in iter_image_reaction_payloads(paths):
        default_paper = paper_name_from_source(data.get("source"), path.stem)
        for rxn_index, rxn in enumerate(data.get("reactions") or [], start=1):
            if not isinstance(rxn, dict):
                continue
            source_paper = paper_name_from_source(rxn.get("source_paper") or data.get("source"), default_paper)
            source_image = clean_text(rxn.get("source_image") or rxn.get("image_name") or rxn.get("image"))
            reaction_id = clean_text(rxn.get("id") or rxn.get("reaction_id")) or f"image_reaction_{rxn_index}"
            field_roles = {
                "substrates": "substrate",
                "products": "product",
                "catalysts": "catalyst",
                "additives": "additive",
                "reagents": "reagent",
            }
            for field, role in field_roles.items():
                for comp_index, comp in enumerate(rxn.get(field) or [], start=1):
                    if isinstance(comp, dict):
                        entities.append(make_image_compound(source_paper, source_image, reaction_id, role, comp, comp_index))
    return [entity for entity in entities if entity.name or entity.iupac_name or entity.symbol or entity.smiles]


def make_alignment(
    text: Entity,
    image: Entity,
    relation: str,
    method: str,
    confidence: str,
    reason: str = "",
    evidence_type: str = "cross_modal_alignment",
) -> Dict[str, Any]:
    return {
        "relationship": relation,
        "match_method": method,
        "confidence": confidence,
        "reason": reason,
        "evidence_type": evidence_type,
        "source_paper": text.source_paper or image.source_paper,
        "text_entity": entity_to_dict(text),
        "image_entity": entity_to_dict(image),
    }


def entity_to_dict(entity: Entity) -> Dict[str, Any]:
    return {
        "entity_id": entity.entity_id,
        "name": entity.name,
        "iupac_name": entity.iupac_name,
        "smiles": entity.smiles,
        "symbol": entity.symbol,
        "role": entity.role,
        "source_modality": entity.source_modality,
        "source_paper": entity.source_paper,
        "reaction_id": entity.reaction_id,
        "source_image": entity.source_image,
        "raw": entity.raw,
    }


def align_compounds(text_entities: List[Entity], image_entities: List[Entity]) -> List[Dict[str, Any]]:
    text_by_iupac: Dict[str, List[Entity]] = {}
    for entity in text_entities:
        key = normalize_iupac_name(entity.iupac_name)
        if key:
            text_by_iupac.setdefault(key, []).append(entity)

    alignments = []
    seen = set()
    for image in image_entities:
        key = normalize_iupac_name(image.iupac_name)
        if not key:
            continue
        for text in text_by_iupac.get(key, []):
            method = "smiles_to_iupac_exact_match" if image.smiles else "iupac_name"
            dedupe_key = (text.entity_id, image.entity_id, method)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            alignments.append(make_alignment(text, image, "SAME_AS", method, "high"))
    return alignments


def build_symbol_resolution_edges(text_entities: List[Entity], image_entities: List[Entity]) -> List[Dict[str, Any]]:
    symbol_map = build_unique_text_symbol_map(text_entities)
    alignments = []
    seen = set()

    for image in image_entities:
        if image.iupac_name or image.smiles:
            continue
        if is_uncertain_text(image.raw.get("label")) or is_uncertain_text(image.raw.get("text")):
            continue
        symbol_candidates = [
            ("label_symbol_resolution", image.symbol),
            ("text_token_symbol_resolution", main_token(image.raw.get("text"))),
        ]
        for method, value in symbol_candidates:
            symbol_key = normalize_symbol(value)
            if not symbol_key:
                continue
            text = symbol_map.get(image.source_paper, {}).get(symbol_key)
            if not text:
                continue
            dedupe_key = (text.entity_id, image.entity_id, method)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            alignments.append(
                make_alignment(
                    text,
                    image,
                    "NAME_RESOLVED_BY",
                    method,
                    "medium",
                    reason="Same-paper unique text symbol mapped to ChemEagle label/text token",
                    evidence_type="same_paper_symbol_resolution",
                )
            )
            break
    return alignments


def find_candidate_matches(
    text_entities: List[Entity],
    image_entities: List[Entity],
    exclude_image_entity_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    exclude_image_entity_ids = exclude_image_entity_ids or set()
    text_by_symbol: Dict[str, List[Entity]] = {}
    text_by_name: Dict[str, List[Entity]] = {}
    for entity in text_entities:
        symbol_key = normalize_symbol(entity.symbol)
        name_key = normalize_symbol(entity.name)
        if symbol_key:
            text_by_symbol.setdefault(symbol_key, []).append(entity)
        if name_key:
            text_by_name.setdefault(name_key, []).append(entity)

    candidates = []
    seen = set()
    for image in image_entities:
        if image.entity_id in exclude_image_entity_ids or image.iupac_name:
            continue
        checks = [
            ("label_symbol_candidate", image.symbol, text_by_symbol, "ChemEagle label matched text symbol"),
            ("text_symbol_candidate", main_token(image.raw.get("text")), text_by_symbol, "ChemEagle text token matched text symbol"),
            ("text_name_candidate", strip_amounts(clean_text(image.raw.get("text"))), text_by_name, "ChemEagle text matched text name"),
        ]
        for method, value, lookup, reason in checks:
            key = normalize_symbol(value)
            if not key:
                continue
            for text in lookup.get(key, []):
                dedupe_key = (text.entity_id, image.entity_id, method)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                confidence = "very_low" if is_uncertain_text(image.raw.get("label")) or is_uncertain_text(image.raw.get("text")) else "low"
                candidates.append(make_alignment(text, image, "CANDIDATE_SAME_AS", method, confidence, reason=reason, evidence_type="candidate_only"))
    return candidates


def alignment_to_triple(alignment: Dict[str, Any]) -> Dict[str, str]:
    text = alignment["text_entity"]
    image = alignment["image_entity"]
    relationship = alignment["relationship"]
    if relationship == "NAME_RESOLVED_BY":
        x_name = image.get("smiles") or image.get("name") or image.get("symbol")
        x_type = "ImageCompound"
        y_name = text.get("iupac_name") or text.get("name")
        y_type = "TextCompound"
    else:
        x_name = text.get("iupac_name") or text.get("name")
        x_type = "TextCompound"
        y_name = image.get("smiles") or image.get("name") or image.get("iupac_name")
        y_type = "ImageCompound"
    return make_triple(
        x_name=x_name,
        x_type=x_type,
        relationship=relationship,
        y_name=y_name,
        y_type=y_type,
        pdf_name=alignment.get("source_paper", ""),
        source_modality="text+image",
        source_image=image.get("source_image", ""),
        text_reaction_id=text.get("reaction_id", ""),
        image_reaction_id=image.get("reaction_id", ""),
        match_method=alignment.get("match_method", ""),
        confidence=alignment.get("confidence", ""),
        evidence_type=alignment.get("evidence_type", "cross_modal_alignment"),
        text_role=text.get("role", ""),
        image_role=image.get("role", ""),
        text_iupac_name=text.get("iupac_name", ""),
        image_smiles=image.get("smiles", ""),
        image_iupac_name=image.get("iupac_name", ""),
    )


def write_unified_multimodal_kg(path: Path, triples: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EDGE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(triples)


def relationship_counts(triples: List[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for triple in triples:
        rel = triple.get("relationship", "")
        counts[rel] = counts.get(rel, 0) + 1
    return counts


def build_cross_modal_kg(
    text_reaction_paths: Iterable[Path | str],
    chemeagle_raw_iupac_paths: Iterable[Path | str],
    output_dir: Path | str,
) -> Dict[str, Any]:
    text_paths = [Path(path) for path in text_reaction_paths]
    image_paths = [Path(path) for path in chemeagle_raw_iupac_paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_triples = build_text_reaction_triples(text_paths)
    image_triples = build_image_reaction_triples(image_paths)
    text_compounds = extract_text_compounds(text_paths)
    image_compounds = extract_chemeagle_compounds(image_paths)

    same_as_alignments = align_compounds(text_compounds, image_compounds)
    symbol_resolution_alignments = build_symbol_resolution_edges(text_compounds, image_compounds)
    excluded_candidate_image_ids = {
        alignment["image_entity"]["entity_id"]
        for alignment in same_as_alignments + symbol_resolution_alignments
        if alignment.get("image_entity")
    }
    candidates = find_candidate_matches(text_compounds, image_compounds, exclude_image_entity_ids=excluded_candidate_image_ids)
    strong_alignments = same_as_alignments + symbol_resolution_alignments

    triples = text_triples + image_triples
    seen = {tuple(triple.get(field, "") for field in EDGE_FIELDNAMES) for triple in triples}
    for alignment in strong_alignments:
        add_unique(triples, seen, alignment_to_triple(alignment))

    alignments_path = output_dir / "cross_modal_alignments.json"
    candidates_path = output_dir / "cross_modal_alignment_candidates.json"
    reaction_candidates_path = output_dir / "reaction_alignment_candidates.json"
    unified_kg_path = output_dir / "kg_triples_unified_multimodal.csv"

    write_json(
        alignments_path,
        {
            "created_at": datetime.now().isoformat(),
            "text_inputs": [str(path) for path in text_paths],
            "chemeagle_inputs": [str(path) for path in image_paths],
            "total_alignments": len(strong_alignments),
            "same_as_alignments": len(same_as_alignments),
            "symbol_resolution_alignments": len(symbol_resolution_alignments),
            "alignments": strong_alignments,
            "notes": {
                "same_as_policy": "Only exact normalized IUPAC-name matches create SAME_AS. Image nodes display SMILES when available, with image_iupac_name retained as evidence.",
                "symbol_resolution_policy": "Symbol-only completion creates NAME_RESOLVED_BY evidence, not SAME_AS.",
                "condition_policy": "SAME_CONDITION_AS is intentionally not emitted in the v1 main KG.",
                "target_policy": "yield/ee/er/dr are edge attributes; KG inputs are filtered reactions only.",
            },
        },
    )
    write_json(
        candidates_path,
        {
            "created_at": datetime.now().isoformat(),
            "total_candidates": len(candidates),
            "candidates": candidates,
        },
    )
    write_json(
        reaction_candidates_path,
        {
            "created_at": datetime.now().isoformat(),
            "total_candidates": 0,
            "candidates": [],
            "notes": "Reaction-level alignment is intentionally candidate-only; automatic merging is disabled.",
        },
    )
    write_unified_multimodal_kg(unified_kg_path, triples)

    counts = relationship_counts(triples)
    return {
        "status": "processed",
        "text_inputs": [str(path) for path in text_paths],
        "chemeagle_inputs": [str(path) for path in image_paths],
        "cross_modal_alignments": str(alignments_path),
        "cross_modal_alignment_candidates": str(candidates_path),
        "reaction_alignment_candidates": str(reaction_candidates_path),
        "kg_triples_unified_multimodal": str(unified_kg_path),
        "kg_triples_multimodal": str(unified_kg_path),
        "text_compounds": len(text_compounds),
        "image_compounds": len(image_compounds),
        "text_reaction_triples": len(text_triples),
        "image_reaction_triples": len(image_triples),
        "same_as_alignments": len(same_as_alignments),
        "symbol_resolution_alignments": len(symbol_resolution_alignments),
        "total_alignments": len(strong_alignments),
        "candidate_alignments": len(candidates),
        "total_kg_triples": len(triples),
        "relationship_counts": counts,
    }
