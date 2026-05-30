import json
import re
from pathlib import Path
from datetime import datetime

def should_filter(name):
    if not name:
        return False
    name = name.strip()
    # 过滤 "unknown"
    if name.lower() == "unknown":
        return True
    # 过滤 1-20 + 小写字母的编号，如 1a, 13j, 20z
    if re.match(r'^(?:[1-9]|1[0-9]|20)[a-z]+$', name):
        return True
    return False

def normalize_targets(reaction):
    if 'targets' not in reaction:
        reaction['targets'] = {}
    # 清理 targets 中已有的无效值
    for key in ('ee', 'yield', 'er', 'dr'):
        if key in reaction['targets']:
            val = reaction['targets'][key]
            if val is None or val == '' or val == 'null' or val == 'not reported':
                reaction['targets'].pop(key)
    # 将顶层字段移入 targets
    for key in ('ee', 'yield', 'er', 'dr'):
        val = reaction.pop(key, None)
        if val is None or val == '' or val == 'null' or val == 'not reported':
            continue
        if key not in reaction['targets']:
            reaction['targets'][key] = val
    if not reaction['targets']:
        reaction.pop('targets', None)

def convert_er_to_ee(targets):
    er = targets.get('er')
    if er is None or er == 'null' or er == '':
        targets.pop('er', None)
        return
    ee = targets.get('ee')
    if ee is not None and ee != 'null' and ee != '':
        targets.pop('er', None)
        return
    match = re.match(r'^\s*([\d.]+)\s*[:/]\s*([\d.]+)\s*$', str(er))
    if not match:
        targets.pop('er', None)
        return
    major = float(match.group(1))
    minor = float(match.group(2))
    total = major + minor
    if total == 0:
        targets.pop('er', None)
        return
    ee_value = abs(major - minor) / total * 100
    targets['ee'] = f"{ee_value:.0f}%" if ee_value == int(ee_value) else f"{ee_value:.1f}%"
    targets.pop('er', None)

def merge_filtered_reactions_from_files(filtered_paths, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_reactions = []
    total_files = 0
    filtered_count = 0

    for json_file in [Path(p) for p in filtered_paths]:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            continue

        source_paper = Path(str(data.get('source') or json_file.stem)).stem.replace("_si", "")

        for reaction in data.get('reactions', []):
            if not isinstance(reaction, dict):
                continue

            skip = False
            for sub in reaction.get('substrates', []) or []:
                if isinstance(sub, dict) and should_filter(sub.get('name', '')):
                    skip = True
                    break
            if not skip:
                for prod in reaction.get('products', []) or []:
                    if isinstance(prod, dict) and should_filter(prod.get('name', '')):
                        skip = True
                        break

            if skip:
                filtered_count += 1
                continue

            reaction = dict(reaction)
            reaction['source_paper'] = source_paper
            reaction.setdefault('source_modality', 'text')
            reaction.pop('source', None)
            reaction.pop('extracted_at', None)
            normalize_targets(reaction)
            if 'targets' in reaction:
                convert_er_to_ee(reaction['targets'])

            all_reactions.append(reaction)

        total_files += 1

    output = {
        "merged_at": datetime.now().isoformat(),
        "total_files": total_files,
        "total_reactions": len(all_reactions),
        "filtered_out_reactions": filtered_count,
        "input_files": [str(Path(p)) for p in filtered_paths],
        "reactions": all_reactions
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Merged {total_files} filtered files into {output_path}")
    print(f"Total reactions: {len(all_reactions)}")
    print(f"Filtered out reactions: {filtered_count}")
    return output

def merge_filtered_reactions(filtered_dir="filtered", output_path=None):
    filtered_dir = Path(filtered_dir)
    output_path = Path(output_path) if output_path else filtered_dir / "merged_filtered_reactions.json"
    
    all_reactions = []
    total_files = 0
    filtered_count = 0
    
    for json_file in filtered_dir.glob("*.json"):
        if json_file.name == "merged_filtered_reactions.json":
            continue
        if json_file.name == "workflow_report.json":
            continue
        if json_file.name.startswith(("Q1_", "Q2_")):
            continue
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 跳过非字典类型的文件（如 Q1, Q2 等特殊文件）
        if not isinstance(data, dict):
            continue
        
        # 提取文件名作为source_paper（去掉_si.json后缀）
        source_paper = json_file.stem.replace("_si", "")
        
        for reaction in data.get('reactions', []):
            # 检查底物和产物name是否需要过滤
            skip = False
            for sub in reaction.get('substrates', []):
                if should_filter(sub.get('name', '')):
                    skip = True
                    break
            if not skip:
                for prod in reaction.get('products', []):
                    if should_filter(prod.get('name', '')):
                        skip = True
                        break
            
            if skip:
                filtered_count += 1
                continue
            
            # 只保留source_paper，移除其他字段
            reaction['source_paper'] = source_paper
            reaction.setdefault('source_modality', 'text')
            # 删除不需要的字段（如果存在）
            reaction.pop('source', None)
            reaction.pop('extracted_at', None)
            
            # 将顶层 ee/yield/er/dr 归入 targets
            normalize_targets(reaction)

            # 将er转为ee（如果有er且没有ee）
            if 'targets' in reaction:
                convert_er_to_ee(reaction['targets'])
            
            all_reactions.append(reaction)
        
        total_files += 1
    
    # 输出结果
    output = {
        "merged_at": datetime.now().isoformat(),
        "total_files": total_files,
        "total_reactions": len(all_reactions),
        "reactions": all_reactions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成!")
    print(f"文件数: {total_files}")
    print(f"总反应数: {len(all_reactions)}")
    print(f"过滤掉的反应数: {filtered_count}")
    return output


if __name__ == "__main__":
    merge_filtered_reactions()
