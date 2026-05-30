import json
import re
from pathlib import Path
from typing import Dict, List


class ReactionFilter:
    INVALID_YIELD_PATTERNS = [
        r'^null$', r'^not specified$', r'^not reported$', r'^determined by chiral HPLC$',
        r'^N/A$', r'^$', r'^/$'
    ]
    INVALID_EE_PATTERNS = INVALID_YIELD_PATTERNS
    INVALID_ER_PATTERNS = [
        r'^null$', r'^not specified$', r'^not reported$', r'^N/A$', r'^$', r'^/$',
        r'^determined by chiral HPLC$'
    ]
    INVALID_DR_PATTERNS = INVALID_ER_PATTERNS

    VALID_NUMBER_PATTERN = r'^[<>≥≤~]?\s*\d+(?:\.\d+)?\s*%?(?:\s*(?:yield|isolated|ee))?'
    VALID_RATIO_PATTERN = r'^[<>≥≤~]?\s*\d+(?:\.\d+)?\s*[:/]\s*[<>≥≤~]?\s*\d+(?:\.\d+)?(?:\s*(?:er|dr))?'

    INVALID_NAME_PATTERNS = [
        r'^target\s+product$',
        r'^desired\s+product$',
        r'^white\s+product$',
        r'^not\s+specified$',
        r'^colorless\s+oil$',
        r'^product$',
        r'^starting\s+material$',
        r'^compound$',
        r'^material$',
        r'^mixture$',
        r'^oil$',
        r'^solid$',
        r'^liquid$',
        r'^crude$',
    ]

    BATCH_SIZE = 15

    def is_invalid_chemical_name(self, name: str) -> bool:
        """检查名称是否明显不是化学名称"""
        if not name:
            return False
        name_lower = name.strip().lower()
        for pattern in self.INVALID_NAME_PATTERNS:
            if re.match(pattern, name_lower, re.IGNORECASE):
                return True
        return False

    def compound_name(self, item: Dict) -> str:
        if not isinstance(item, dict):
            return ""
        for key in (
            "iupac_name",
            "resolved_iupac_name",
            "name",
            "resolved_name_normalized",
            "resolved_name",
            "smiles",
            "resolved_smiles",
            "symbol",
        ):
            value = item.get(key)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str and value_str.lower() not in {"none", "null", "unknown", "not specified", "<invalid>"}:
                return value_str
        return ""

    def __init__(self, batch_size: int = 15):
        self.BATCH_SIZE = batch_size
        self.last_stats = {}

    def is_valid_number_value(self, value, invalid_patterns) -> bool:
        if value is None:
            return False
        value_str = str(value).strip().lower()
        if not value_str:
            return False
        for pattern in invalid_patterns:
            if re.match(pattern, value_str, re.IGNORECASE):
                return False
        if re.match(self.VALID_NUMBER_PATTERN, value_str, re.IGNORECASE):
            return True
        if re.match(self.VALID_RATIO_PATTERN, value_str, re.IGNORECASE):
            return True
        return False

    def is_negative_value(self, value) -> bool:
        if value is None:
            return False
        value_str = str(value).strip()
        if not value_str:
            return False
        # 检查是否以负号开头
        return value_str.startswith('-')

    def check_numeric_criteria(self, reaction: Dict) -> bool:
        targets = reaction.get('targets', {})
        
        yield_val = targets.get('yield') if targets else None
        ee_val = targets.get('ee') if targets else None
        er_val = targets.get('er') if targets else None
        dr_val = targets.get('dr') if targets else None
        
        if yield_val is None and 'yield' in reaction:
            yield_val = reaction.get('yield')
        if ee_val is None and 'ee' in reaction:
            ee_val = reaction.get('ee')
        if er_val is None and 'er' in reaction:
            er_val = reaction.get('er')
        if dr_val is None and 'dr' in reaction:
            dr_val = reaction.get('dr')

        # 如果yield或ee为负数，直接返回False
        if self.is_negative_value(yield_val) or self.is_negative_value(ee_val):
            return False

        yield_valid = self.is_valid_number_value(yield_val, self.INVALID_YIELD_PATTERNS)
        ee_valid = self.is_valid_number_value(ee_val, self.INVALID_EE_PATTERNS)
        er_valid = self.is_valid_number_value(er_val, self.INVALID_ER_PATTERNS)
        dr_valid = self.is_valid_number_value(dr_val, self.INVALID_DR_PATTERNS)

        all_invalid = not yield_valid and not ee_valid and not er_valid and not dr_valid
        return not all_invalid

    def filter_reactions(self, reactions: List[Dict]) -> List[Dict]:
        if not reactions:
            return []

        print(f"  Step 0: Basic filtering...")
        self.last_stats = {
            "input_reactions": len(reactions),
            "basic_failed": 0,
            "target_failed": 0,
            "retained": 0,
        }
        valid_indices = []
        for idx, reaction in enumerate(reactions, 1):
            r_id = reaction.get('id', 'N/A')
            substrates = reaction.get('substrates') or []
            products = reaction.get('products') or []
            
            # 检查1：空列表
            if not substrates or not products:
                print(f"    [{idx}] Failed: {r_id} (empty substrates or products)")
                self.last_stats["basic_failed"] += 1
                continue
            
            # 检查2：底物name全部为空 或 产物name全部为空
            all_sub_names_empty = all(self.compound_name(s) == '' for s in substrates)
            all_prod_names_empty = all(self.compound_name(p) == '' for p in products)
            if all_sub_names_empty or all_prod_names_empty:
                print(f"    [{idx}] Failed: {r_id} (all substrate names null or all product names null)")
                self.last_stats["basic_failed"] += 1
                continue

            # 检查3：底物和产物名称相同
            sub_names = set()
            prod_names = set()
            
            for s in substrates:
                name = self.compound_name(s).strip().lower()
                if name:
                    sub_names.add(name)
            
            for p in products:
                name = self.compound_name(p).strip().lower()
                if name:
                    prod_names.add(name)
            
            # 如果两者都有名称，检查是否相同
            if sub_names and prod_names and sub_names == prod_names:
                print(f"    [{idx}] Failed: {r_id} (substrates and products have identical names)")
                self.last_stats["basic_failed"] += 1
                continue
            
            # 检查4：过滤明显不是化学名称的条目
            invalid_names = []
            
            for s in substrates:
                name = self.compound_name(s)
                if self.is_invalid_chemical_name(name):
                    invalid_names.append(f"substrate: '{name}'")
            
            for p in products:
                name = self.compound_name(p)
                if self.is_invalid_chemical_name(name):
                    invalid_names.append(f"product: '{name}'")
            
            if invalid_names:
                print(f"    [{idx}] Failed: {r_id} (invalid chemical name: {', '.join(invalid_names)})")
                self.last_stats["basic_failed"] += 1
                continue
            
            valid_indices.append(idx - 1)
            print(f"    [{idx}] Passed: {r_id}")

        if not valid_indices:
            print(f"  No reactions passed basic filter")
            return []

        valid_reactions = [reactions[i] for i in valid_indices]
        print(f"  Passed basic filter: {len(valid_reactions)}/{len(reactions)}")

        print(f"  Step 1: Numeric criteria filtering...")
        numeric_valid_indices = []
        for idx, reaction in enumerate(valid_reactions, 1):
            if self.check_numeric_criteria(reaction):
                numeric_valid_indices.append(idx - 1)
                print(f"    [{idx}] Passed numeric filter: {reaction.get('id', 'N/A')}")
            else:
                print(f"    [{idx}] Failed numeric filter: {reaction.get('id', 'N/A')}")
                self.last_stats["target_failed"] += 1

        if not numeric_valid_indices:
            print(f"  No reactions passed numeric filter")
            return []

        numeric_valid_reactions = [valid_reactions[i] for i in numeric_valid_indices]
        self.last_stats["retained"] = len(numeric_valid_reactions)
        print(f"  Passed numeric filter: {len(numeric_valid_reactions)}/{len(valid_reactions)}")

        return numeric_valid_reactions

    def filter_file(self, input_path: str, output_path: str) -> int:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        reactions = data.get('reactions', [])
        if not reactions:
            print(f"  No reactions found in: {input_path}")
            output_data = {
                "source": data.get('source', input_path),
                "extracted_at": data.get('extracted_at'),
                "total_reactions": 0,
                "filtered_reactions": 0,
                "reactions": []
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            return 0

        print(f"  Total reactions to filter: {len(reactions)}")
        filtered = self.filter_reactions(reactions)

        output_data = {
            "source": data.get('source', input_path),
            "extracted_at": data.get('extracted_at'),
            "total_reactions": len(reactions),
            "filtered_reactions": len(filtered),
            "filter_stats": self.last_stats,
            "reactions": filtered
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"  Final result: {len(filtered)}/{len(reactions)} reactions retained")
        return len(filtered)

def filter_reaction_file(input_path, output_path, overwrite: bool = False) -> Dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "input": str(input_path),
            "output": str(output_path),
            "reactions": None,
        }

    filter_engine = ReactionFilter(batch_size=15)
    count = filter_engine.filter_file(str(input_path), str(output_path))
    return {
        "status": "processed",
        "input": str(input_path),
        "output": str(output_path),
        "reactions": count,
        "filter_stats": filter_engine.last_stats,
    }

def run_filter(input_dir="output", output_dir="filtered", overwrite: bool = False) -> Dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return {"processed": 0, "skipped": 0, "total_filtered": 0, "files": {}}

    filter_engine = ReactionFilter(batch_size=15)

    total_filtered = 0
    skipped_files = 0
    file_results = {}
    for json_file in json_files:
        output_path = output_dir / json_file.name

        if output_path.exists() and not overwrite:
            skipped_files += 1
            print(f"\nSkipping file (already processed): {json_file.name}")
            file_results[json_file.name] = {"status": "skipped", "output": str(output_path)}
            continue

        print(f"\nProcessing file: {json_file.name}")
        count = filter_engine.filter_file(str(json_file), str(output_path))
        total_filtered += count
        file_results[json_file.name] = {
            "status": "processed",
            "output": str(output_path),
            "reactions": count,
        }

    processed_files = len(json_files) - skipped_files
    print(f"\n===== COMPLETE =====")
    print(f"Files processed: {processed_files}")
    print(f"Files skipped (already processed): {skipped_files}")
    print(f"Total reactions retained: {total_filtered}")
    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "processed": processed_files,
        "skipped": skipped_files,
        "total_filtered": total_filtered,
        "files": file_results,
    }


def main():
    input_dir = Path("output")
    output_dir = Path("filtered")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    filter_engine = ReactionFilter(batch_size=15)

    total_filtered = 0
    skipped_files = 0
    for json_file in json_files:
        output_path = output_dir / json_file.name

        # 如果 filtered 中已有同名结果文件，则跳过避免重复处理
        if output_path.exists():
            skipped_files += 1
            print(f"\nSkipping file (already processed): {json_file.name}")
            continue

        print(f"\nProcessing file: {json_file.name}")
        count = filter_engine.filter_file(str(json_file), str(output_path))
        total_filtered += count

    processed_files = len(json_files) - skipped_files
    print(f"\n===== COMPLETE =====")
    print(f"Files processed: {processed_files}")
    print(f"Files skipped (already processed): {skipped_files}")
    print(f"Total reactions retained: {total_filtered}")


if __name__ == "__main__":
    main()
