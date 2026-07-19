import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

EXTRACT_DIR = Path(__file__).resolve().parent.parent
if str(EXTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(EXTRACT_DIR))

from merge_filtered import merge_filtered_reactions_from_files


def renormalize(merged_path: Path, input_dirs: list[Path]) -> dict:
    merged_path = merged_path.resolve()
    previous = json.loads(merged_path.read_text(encoding="utf-8"))
    candidates = {}
    for input_dir in input_dirs:
        for path in input_dir.resolve().glob("*.json"):
            if path.name in candidates:
                raise ValueError(f"Duplicate local input basename: {path.name}")
            candidates[path.name] = path

    ordered_inputs = []
    missing = []
    for historical_path in previous.get("input_files", []):
        basename = Path(historical_path).name
        local_path = candidates.get(basename)
        if local_path is None:
            missing.append(basename)
        else:
            ordered_inputs.append(local_path)
    if missing:
        raise ValueError(f"Missing {len(missing)} local merge inputs: {missing[:10]}")
    if len(ordered_inputs) != int(previous.get("total_files", -1)):
        raise ValueError("Local input count does not match previous merge metadata")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = merged_path.with_name(
        f"{merged_path.stem}_backup_pre_er_normalization_{run_id}{merged_path.suffix}"
    )
    shutil.copy2(merged_path, backup_path)
    result = merge_filtered_reactions_from_files(ordered_inputs, merged_path)
    if result.get("total_reactions") != previous.get("total_reactions"):
        shutil.copy2(backup_path, merged_path)
        raise ValueError("Reaction count changed; restored merged backup")
    return {
        "merged_path": str(merged_path),
        "backup_path": str(backup_path),
        "total_files": result.get("total_files"),
        "total_reactions": result.get("total_reactions"),
        "er_to_ee_policy": result.get("er_to_ee_policy"),
        "er_to_ee_conversion": result.get("er_to_ee_conversion"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path, action="append")
    args = parser.parse_args()
    print(json.dumps(renormalize(args.merged, args.input_dir), ensure_ascii=False, indent=2))
