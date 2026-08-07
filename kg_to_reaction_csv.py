"""Convert unified KG edge CSV into one-row-per-reaction summary CSV."""

from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List


COL_PAPER = "文献"
COL_LOCATION = "反应位置"
COL_SUBSTRATE = "底物"
COL_PRODUCT = "产物"
COL_INTERMEDIATE = "中间体"
COL_CATALYST = "催化剂"
COL_LIGAND = "配体"
COL_SOLVENT = "溶剂"
COL_OTHER_COMPONENT = "其它组分"
COL_CONDITIONS = "条件（温度、时间、气氛、光源、波长等）"
COL_YIELD = "产率"
COL_INTRINSIC_EXTRACTION_ERROR = "直接从文献中抓取的提取错误"
COL_PROCEDURAL_FIDELITY_ERROR = "实验过程与操作顺序错误"

OUTPUT_FIELDNAMES = [
    COL_PAPER,
    COL_LOCATION,
    COL_SUBSTRATE,
    COL_PRODUCT,
    COL_INTERMEDIATE,
    COL_CATALYST,
    COL_LIGAND,
    COL_SOLVENT,
    COL_OTHER_COMPONENT,
    COL_CONDITIONS,
    COL_YIELD,
    "dr",
    "ee",
    "er",
    COL_INTRINSIC_EXTRACTION_ERROR,
    COL_PROCEDURAL_FIDELITY_ERROR,
]


ROLE_CONFIG = {
    "USES_SUBSTRATE": (COL_SUBSTRATE, "substrate_amount"),
    "PRODUCES": (COL_PRODUCT, "product_amount"),
    "USES_CATALYST": (COL_CATALYST, "catalyst_amount"),
    "USES_LIGAND": (COL_LIGAND, "ligand_amount"),
    "USES_SOLVENT": (COL_SOLVENT, "solvent_amount"),
    "USES_OTHER_COMPONENT": (COL_OTHER_COMPONENT, "other_component_amount"),
    # Legacy KG relationships are accepted and folded into the new summary column.
    "USES_ADDITIVE": (COL_OTHER_COMPONENT, "additive_amount"),
    "USES_REAGENT": (COL_OTHER_COMPONENT, "reagent_amount"),
}


IGNORED_RELATIONSHIPS = {
    "REPORTED_IN",
    "OF_TYPE",
    "HAS_SCAFFOLD",
    "HAS_SUBSTITUENT",
    "EXTRACTED_FROM_IMAGE",
    "SAME_AS",
    "NAME_RESOLVED_BY",
}

SOLVENT_NAME_LABELS = {"solvent"}
SOLVENT_AMOUNT_LABELS = {
    "solvent amount",
    "vol",
    "volume",
}
SOLVENT_CONDITION_LABELS = SOLVENT_NAME_LABELS | SOLVENT_AMOUNT_LABELS


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "none", "null", "not specified"}:
        return ""
    return re.sub(r"\s+", " ", text)


def reaction_location(reaction_id: str) -> str:
    """Return the local reaction id from TextReaction/ImageReaction node ids."""
    text = clean_text(reaction_id)
    if not text:
        return ""
    return text.rsplit(":", 1)[-1]


def parse_source_pages(value: str) -> List[int]:
    """Parse KG source_pages strings such as '24' or '117, 118'."""
    pages = []
    seen = set()
    for match in re.findall(r"\d+", clean_text(value)):
        page = int(match)
        if page > 0 and page not in seen:
            seen.add(page)
            pages.append(page)
    return pages


def render_reaction_location(local_id: str, pages: Iterable[int]) -> str:
    unique_pages = sorted({int(page) for page in pages if int(page) > 0})
    if not unique_pages:
        return local_id
    return f"{local_id}; PDF pages: {', '.join(str(page) for page in unique_pages)}"


def normalize_step(step: str) -> str:
    text = clean_text(step)
    match = re.fullmatch(r"(?i)(?:step\s*)?(\d+)", text)
    return match.group(1) if match else ""


def add_unique(values: List[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def formatted_compound(row: Dict[str, str], amount_column: str = "") -> str:
    name = clean_text(row.get("y_name"))
    if not name:
        return ""
    amount = clean_text(row.get(amount_column)) if amount_column else ""
    return f"{name} ({amount})" if amount else name


def add_grouped_value(grouped: Dict[str, List[str]], step: str, value: str) -> None:
    key = normalize_step(step)
    grouped.setdefault(key, [])
    add_unique(grouped[key], value)


def render_grouped_values(grouped: Dict[str, List[str]]) -> str:
    plain = grouped.get("", [])
    parts = list(plain)
    for step in sorted((item for item in grouped if item), key=lambda item: int(item)):
        values = grouped.get(step, [])
        if values:
            parts.append(f"step{step}: {'; '.join(values)}")
    return "; ".join(parts)


def first_nonempty(rows: Iterable[Dict[str, str]], field: str) -> str:
    for row in rows:
        value = clean_text(row.get(field))
        if value:
            return value
    return ""


def split_condition_parts(condition_text: str) -> List[str]:
    """Split combined KG condition text while preserving single condition values."""
    text = clean_text(condition_text)
    if not text:
        return []
    return [
        part.strip()
        for part in re.split(
            r";\s*|,\s*(?=(?:solvent|solvent_amount|solvent amount|vol|volume|temp|time|atm|light|wl|conc)\s*:)",
            text,
            flags=re.IGNORECASE,
        )
        if part.strip()
    ]


def condition_label(part: str) -> str:
    match = re.match(r"\s*([^:：]+)\s*[:：]", part)
    return clean_text(match.group(1)).casefold().replace("_", " ") if match else ""


def is_solvent_condition(part: str) -> bool:
    return condition_label(part) in SOLVENT_CONDITION_LABELS


def condition_value(part: str) -> str:
    match = re.match(r"\s*[^:：]+\s*[:：]\s*(.*)", part)
    return clean_text(match.group(1)) if match else ""


def add_legacy_solvent_part(grouped: Dict[str, Dict[str, List[str]]], step: str, part: str) -> None:
    key = normalize_step(step)
    bucket = grouped.setdefault(key, {"names": [], "amounts": []})
    label = condition_label(part)
    value = condition_value(part)
    if not value:
        return
    target = bucket["names"] if label in SOLVENT_NAME_LABELS else bucket["amounts"]
    add_unique(target, value)


def merge_legacy_solvents(
    solvents: Dict[str, List[str]],
    legacy: Dict[str, Dict[str, List[str]]],
) -> None:
    """Pair legacy HAS_CONDITION solvent fields by step and stable source order."""
    for step, bucket in legacy.items():
        names = bucket.get("names", [])
        amounts = bucket.get("amounts", [])
        for index, name in enumerate(names):
            amount = amounts[index] if index < len(amounts) else ""
            value = f"{name} ({amount})" if amount else name
            add_grouped_value(solvents, step, value)
        for amount in amounts[len(names):]:
            add_grouped_value(solvents, step, f"solvent_amount: {amount}")


def new_reaction_record(row: Dict[str, str], reaction_id: str) -> Dict:
    return {
        "rows": [],
        "source_pages": [],
        COL_PAPER: clean_text(row.get("pdf_name")),
        COL_LOCATION: reaction_location(reaction_id),
        COL_SUBSTRATE: {},
        COL_PRODUCT: {},
        COL_INTERMEDIATE: {},
        COL_CATALYST: {},
        COL_LIGAND: {},
        COL_SOLVENT: {},
        "legacy_solvents": {},
        COL_OTHER_COMPONENT: {},
        COL_CONDITIONS: {},
    }


def convert_kg_rows(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    reactions: "OrderedDict[str, Dict]" = OrderedDict()
    for row in rows:
        if clean_text(row.get("source_modality")) == "text+image":
            continue
        reaction_id = clean_text(row.get("reaction_id"))
        if not reaction_id:
            continue
        rel = clean_text(row.get("relationship"))
        if rel in IGNORED_RELATIONSHIPS:
            continue

        reaction = reactions.setdefault(reaction_id, new_reaction_record(row, reaction_id))
        reaction["rows"].append(row)
        for page in parse_source_pages(row.get("source_pages", "")):
            if page not in reaction["source_pages"]:
                reaction["source_pages"].append(page)
        if not reaction[COL_PAPER]:
            reaction[COL_PAPER] = clean_text(row.get("pdf_name"))

        if rel in ROLE_CONFIG:
            field, amount_column = ROLE_CONFIG[rel]
            add_grouped_value(
                reaction[field],
                row.get("step", ""),
                formatted_compound(row, amount_column),
            )
        elif rel == "HAS_CONDITION":
            for part in split_condition_parts(row.get("y_name")):
                if is_solvent_condition(part):
                    add_legacy_solvent_part(
                        reaction["legacy_solvents"], row.get("step", ""), part
                    )
                else:
                    add_grouped_value(
                        reaction[COL_CONDITIONS], row.get("step", ""), part
                    )
        elif rel in {"PRODUCES_INTERMEDIATE", "USES_INTERMEDIATE"}:
            # Prefer the produced step when both produced/use edges exist.
            if rel == "USES_INTERMEDIATE":
                existing_values = {
                    value
                    for values in reaction[COL_INTERMEDIATE].values()
                    for value in values
                }
                if clean_text(row.get("y_name")) in existing_values:
                    continue
            add_grouped_value(
                reaction[COL_INTERMEDIATE],
                row.get("step", ""),
                clean_text(row.get("y_name")),
            )

    output = []
    for reaction in reactions.values():
        source_rows = reaction.pop("rows")
        source_pages = reaction.pop("source_pages")
        merge_legacy_solvents(
            reaction[COL_SOLVENT], reaction.pop("legacy_solvents")
        )
        output.append(
            {
                COL_PAPER: reaction[COL_PAPER],
                COL_LOCATION: render_reaction_location(reaction[COL_LOCATION], source_pages),
                COL_SUBSTRATE: render_grouped_values(reaction[COL_SUBSTRATE]),
                COL_PRODUCT: render_grouped_values(reaction[COL_PRODUCT]),
                COL_INTERMEDIATE: render_grouped_values(reaction[COL_INTERMEDIATE]),
                COL_CATALYST: render_grouped_values(reaction[COL_CATALYST]),
                COL_LIGAND: render_grouped_values(reaction[COL_LIGAND]),
                COL_SOLVENT: render_grouped_values(reaction[COL_SOLVENT]),
                COL_OTHER_COMPONENT: render_grouped_values(reaction[COL_OTHER_COMPONENT]),
                COL_CONDITIONS: render_grouped_values(reaction[COL_CONDITIONS]),
                COL_YIELD: first_nonempty(source_rows, "yield"),
                "dr": first_nonempty(source_rows, "dr"),
                "ee": first_nonempty(source_rows, "ee"),
                "er": first_nonempty(source_rows, "er"),
                COL_INTRINSIC_EXTRACTION_ERROR: "",
                COL_PROCEDURAL_FIDELITY_ERROR: "",
            }
        )
    return output


def convert_kg_csv(kg_path: Path | str, output_path: Path | str) -> Dict[str, int | str]:
    kg_path = Path(kg_path)
    output_path = Path(output_path)
    with kg_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = convert_kg_rows(rows)
    # Python's sort is stable, so reaction order within the same paper is retained.
    output_rows.sort(
        key=lambda row: clean_text(row.get(COL_PAPER)).casefold(),
        reverse=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "kg_path": str(kg_path),
        "output_path": str(output_path),
        "input_edges": len(rows),
        "output_reactions": len(output_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert unified KG triples CSV into reaction summary CSV."
    )
    parser.add_argument("--kg", required=True, help="Input kg_triples_unified_multimodal.csv")
    parser.add_argument("--output", required=True, help="Output reaction summary CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = convert_kg_csv(args.kg, args.output)
    print(
        f"Converted {result['input_edges']} KG edges into "
        f"{result['output_reactions']} reaction rows: {result['output_path']}"
    )


if __name__ == "__main__":
    main()
