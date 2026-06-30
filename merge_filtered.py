import json
import re
from datetime import datetime
from pathlib import Path


def should_filter(name):
    # Compatibility hook: merge no longer hard-filters symbolic names such as
    # 1a/2b/5g or placeholder names. Downstream benchmark/KG stages decide
    # whether a reaction is usable.
    return False


def is_empty_target_value(value):
    if value is None:
        return True
    return str(value).strip().casefold() in {"", "null", "not reported"}


def normalize_targets(reaction):
    if "targets" not in reaction or not isinstance(reaction.get("targets"), dict):
        reaction["targets"] = {}

    for key in ("ee", "yield", "er", "dr"):
        if key in reaction["targets"] and is_empty_target_value(reaction["targets"][key]):
            reaction["targets"].pop(key)

    for key in ("ee", "yield", "er", "dr"):
        val = reaction.pop(key, None)
        if is_empty_target_value(val):
            continue
        if key not in reaction["targets"]:
            reaction["targets"][key] = val

    if not reaction["targets"]:
        reaction.pop("targets", None)


def convert_er_to_ee(targets):
    er = targets.get("er")
    if is_empty_target_value(er):
        targets.pop("er", None)
        return

    ee = targets.get("ee")
    if not is_empty_target_value(ee):
        return

    match = re.match(r"^\s*([\d.]+)\s*[:/]\s*([\d.]+)\s*$", str(er))
    if not match:
        return

    major = float(match.group(1))
    minor = float(match.group(2))
    total = major + minor
    if total == 0:
        return

    ee_value = abs(major - minor) / total * 100
    targets["ee"] = f"{ee_value:.0f}%" if ee_value == int(ee_value) else f"{ee_value:.1f}%"


def _source_paper_from_payload(data, json_file):
    return Path(str(data.get("source") or json_file.stem)).stem.replace("_si", "")


def _normalize_reaction_for_merge(reaction, source_paper):
    reaction = dict(reaction)
    reaction["source_paper"] = source_paper
    reaction.setdefault("source_modality", "text")
    reaction.pop("source", None)
    reaction.pop("extracted_at", None)
    normalize_targets(reaction)
    if "targets" in reaction:
        convert_er_to_ee(reaction["targets"])
    return reaction


def merge_filtered_reactions_from_files(filtered_paths, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_reactions = []
    total_files = 0
    skipped_non_dict_reactions = 0

    for json_file in [Path(p) for p in filtered_paths]:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        source_paper = _source_paper_from_payload(data, json_file)

        for reaction in data.get("reactions", []):
            if not isinstance(reaction, dict):
                skipped_non_dict_reactions += 1
                continue
            all_reactions.append(_normalize_reaction_for_merge(reaction, source_paper))

        total_files += 1

    output = {
        "merged_at": datetime.now().isoformat(),
        "total_files": total_files,
        "total_reactions": len(all_reactions),
        "filtered_out_reactions": 0,
        "merge_skipped_reactions": skipped_non_dict_reactions,
        "input_files": [str(Path(p)) for p in filtered_paths],
        "reactions": all_reactions,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Merged {total_files} filtered files into {output_path}")
    print(f"Total reactions: {len(all_reactions)}")
    print("Filtered out reactions: 0")
    print(f"Skipped non-dict reactions: {skipped_non_dict_reactions}")
    return output


def merge_filtered_reactions(filtered_dir="filtered", output_path=None):
    filtered_dir = Path(filtered_dir)
    output_path = Path(output_path) if output_path else filtered_dir / "merged_filtered_reactions.json"

    input_paths = []
    for json_file in filtered_dir.glob("*.json"):
        if json_file.name == "merged_filtered_reactions.json":
            continue
        if json_file.name == "workflow_report.json":
            continue
        if json_file.name.startswith(("Q1_", "Q2_")):
            continue
        input_paths.append(json_file)

    return merge_filtered_reactions_from_files(input_paths, output_path)


if __name__ == "__main__":
    merge_filtered_reactions()
