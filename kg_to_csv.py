import json
import csv
from pathlib import Path

BASE = Path(__file__).parent / "filtered"
Q1_PATH = BASE / "Q1_substrate_to_condition.json"
Q2_PATH = BASE / "Q2_condition_to_substrate.json"
OUTPUT_PATH = BASE / "kg_triples.csv"


def clean_name(s) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        return ", ".join(str(x) for x in s if x)
    s = str(s)
    if not s or s in ("not specified", "null", "None", ""):
        return ""
    return s.strip()


def paper_short_name(full: str) -> str:
    return full[:60].strip()


def extract_reaction_types(rxn: dict) -> list[str]:
    value = clean_name((rxn or {}).get("reaction_type"))
    if not value:
        value = "unknown reaction"
    return [value]

def normalize_entity(item):
    if isinstance(item, dict):
        return item
    if item is None:
        return {}
    return {"name": str(item)}

def normalize_reaction(rxn, index: int):
    if not isinstance(rxn, dict):
        rxn = {}
    normalized = dict(rxn)
    normalized.setdefault("id", f"reaction_{index}")
    if not isinstance(normalized.get("conditions"), dict):
        normalized["conditions"] = {}
    if not isinstance(normalized.get("targets"), dict):
        normalized["targets"] = {}
    for field in (
        "substrates", "products", "catalysts", "ligands",
        "other_components", "additives", "reagents",
    ):
        values = normalized.get(field) or []
        if not isinstance(values, list):
            values = [values]
        normalized[field] = [normalize_entity(item) for item in values]
    return normalized


def build_kg_csv(q1_path=Q1_PATH, q2_path=Q2_PATH, output_path=OUTPUT_PATH):
    q1_path = Path(q1_path)
    q2_path = Path(q2_path)
    output_path = Path(output_path)

    print(f"Loading {q1_path} ...")
    with open(q1_path, encoding="utf-8") as f:
        q1 = json.load(f)
    print(f"  {len(q1)} reactions loaded from Q1")

    print(f"Loading {q2_path} ...")
    with open(q2_path, encoding="utf-8") as f:
        q2 = json.load(f)
    print(f"  {len(q2)} reactions loaded from Q2")

    seen_keys = set()
    reactions = []
    for idx, raw_rxn in enumerate(q1 + q2, start=1):
        rxn = normalize_reaction(raw_rxn, idx)
        key = (rxn.get("source_paper", ""), rxn["id"])
        if key not in seen_keys:
            seen_keys.add(key)
            reactions.append(rxn)
    print(f"  {len(reactions)} unique reactions after merge")

    triples = []
    seen_triples = set()

    def add(x_name, x_type, relationship, y_name, y_type, pdf_name, yield_val="", ee="", er="", dr=""):
        key = (x_name, relationship, y_name, pdf_name)
        if key in seen_triples:
            return
        seen_triples.add(key)
        triples.append({
            "x_name": x_name,
            "x_type": x_type,
            "relationship": relationship,
            "y_name": y_name,
            "y_type": y_type,
            "pdf_name": pdf_name,
            "yield": yield_val,
            "ee": ee,
            "er": er,
            "dr": dr,
        })

    for rxn in reactions:
        rxn_id = rxn["id"]
        src = rxn.get("source_paper", "unknown")
        p_short = paper_short_name(src)

        x_name = f"{p_short}: {rxn_id}"

        cond = rxn.get("conditions", {})
        targets = rxn.get("targets", {})

        cond_parts = []
        solvent = clean_name(cond.get("solvent"))
        if solvent:
            cond_parts.append(f"solvent: {solvent}")
        temp = clean_name(cond.get("temperature"))
        if temp:
            cond_parts.append(f"temp: {temp}")
        time = clean_name(cond.get("time"))
        if time:
            cond_parts.append(f"time: {time}")
        light = clean_name(cond.get("light source")) or clean_name(cond.get("light_source"))
        if light:
            cond_parts.append(f"light: {light}")
        wl = clean_name(cond.get("wavelength"))
        if wl:
            cond_parts.append(f"wl: {wl}")
        vol = clean_name(cond.get("volume"))
        if vol:
            cond_parts.append(f"vol: {vol}")
        cond_summary = ", ".join(cond_parts) if cond_parts else "not specified"

        yield_val = clean_name(targets.get("yield"))
        ee_raw = targets.get("ee")
        ee = clean_name(str(ee_raw)) if ee_raw is not None else ""
        if ee in ("null", "None"):
            ee = ""
        er = clean_name(targets.get("er"))
        dr = clean_name(targets.get("dr"))

        add(x_name, "Reaction", "REPORTED_IN", src, "Paper", src, yield_val, ee, er, dr)
        add(x_name, "Reaction", "HAS_CONDITION", cond_summary, "Condition", src, yield_val, ee, er, dr)

        for rt in extract_reaction_types(rxn):
            add(x_name, "Reaction", "OF_TYPE", rt, "ReactionType", src, yield_val, ee, er, dr)

        for sub in (rxn.get("substrates") or []):
            sub_name = clean_name(sub.get("name", ""))
            if not sub_name:
                sub_name = clean_name(sub.get("symbol", "unknown"))
            if not sub_name:
                sub_name = "unknown"

            add(x_name, "Reaction", "USES_SUBSTRATE", sub_name, "Substrate", src, yield_val, ee, er, dr)

            scaffold = clean_name(sub.get("scaffold"))
            if scaffold:
                add(sub_name, "Substrate", "HAS_SCAFFOLD", scaffold, "SubstrateScaffold", src, yield_val, ee, er, dr)

            for subst_name in (sub.get("substituents") or []):
                subst_clean = clean_name(subst_name)
                if subst_clean:
                    add(sub_name, "Substrate", "HAS_SUBSTITUENT", subst_clean, "Substituent", src, yield_val, ee, er, dr)

        for prod in (rxn.get("products") or []):
            prod_name = clean_name(prod.get("name", ""))
            if not prod_name:
                prod_name = clean_name(prod.get("symbol", "unknown"))
            if not prod_name:
                prod_name = "unknown"
            add(x_name, "Reaction", "PRODUCES", prod_name, "Product", src, yield_val, ee, er, dr)

        for cat in (rxn.get("catalysts") or []):
            cat_name = clean_name(cat.get("name", ""))
            if not cat_name:
                cat_name = clean_name(cat.get("symbol", "unknown"))
            if not cat_name:
                cat_name = "unknown"
            add(x_name, "Reaction", "USES_CATALYST", cat_name, "Catalyst", src, yield_val, ee, er, dr)

        for add_item in (rxn.get("additives") or []):
            add_name = clean_name(add_item.get("name", "unknown"))
            if not add_name:
                add_name = "unknown"
            add(x_name, "Reaction", "USES_ADDITIVE", add_name, "Additive", src, yield_val, ee, er, dr)

        for reag in (rxn.get("reagents") or []):
            reag_name = clean_name(reag.get("name", "unknown"))
            if not reag_name:
                reag_name = "unknown"
            add(x_name, "Reaction", "USES_REAGENT", reag_name, "Reagent", src, yield_val, ee, er, dr)

    fieldnames = ["x_name", "x_type", "relationship", "y_name", "y_type", "pdf_name", "yield", "ee", "er", "dr"]

    print(f"Writing {len(triples)} triples to {output_path} ...")
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(triples)

    rel_counts = {}
    for t in triples:
        rel_counts[t["relationship"]] = rel_counts.get(t["relationship"], 0) + 1

    print(f"\nDone! {len(triples)} total triples:")
    for rel, cnt in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"  {rel}: {cnt}")
    return {
        "q1_path": str(q1_path),
        "q2_path": str(q2_path),
        "output_path": str(output_path),
        "total_reactions": len(reactions),
        "total_triples": len(triples),
        "relationship_counts": rel_counts,
    }


def main():
    build_kg_csv()


if __name__ == "__main__":
    main()
