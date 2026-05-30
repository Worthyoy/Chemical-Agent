import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv, dotenv_values
    DOTENV_PATH = Path(__file__).resolve().parent / ".env"
    load_dotenv(DOTENV_PATH)
    load_dotenv()
except ImportError:
    dotenv_values = None
    DOTENV_PATH = Path(__file__).resolve().parent / ".env"

from openai import OpenAI


# ============================================================
# 正则快速预过滤（不需要 API 调用）
# ============================================================

def is_symbol_name(name: str) -> bool:
    """是否为符号名称（如 1a, 6b, 13j, 20z）"""
    if not name:
        return False
    return bool(re.match(r'^[0-9][a-zA-Z0-9]{0,4}$', name.strip()))


def is_obvious_generic(name: str) -> bool:
    """是否为明显泛化名称（不需要 LLM 即可判断）"""
    if not name:
        return False
    n = name.strip().lower()
    exact = {
        'ketone derivative', 'aldehyde derivative', 'amine derivative',
        'acid derivative', 'alcohol derivative', 'ester derivative',
        'amide derivative', 'nitrile derivative', 'olefin derivative',
        'alkene derivative', 'alkyne derivative', 'vinyl derivative',
        'aryl derivative', 'heteroaryl derivative', 'phenol derivative',
        'compound', 'product', 'material', 'starting material',
        'unknown', 'not specified', 'not reported', 'n/a', 'null',
    }
    if n in exact:
        return True
    generic_keywords = [
        'derivative', 'substituted', 'compound', 'starting material',
        'product', 'material', 'mixture',
    ]
    if len(n) < 20:
        for kw in generic_keywords:
            if kw in n:
                return True
    return False


def quick_classify(name: str) -> Optional[str]:
    """
    快速分类：返回 'Q2' 表示确定非具体IUPAC，None 表示需要进一步判断
    """
    if is_symbol_name(name):
        return 'Q2'
    if is_obvious_generic(name):
        return 'Q2'
    return None


def has_valid_condition(reaction: dict) -> bool:
    """检查反应是否至少有一个有效的反应条件"""
    conditions = reaction.get('conditions', {})
    if not conditions:
        return False
    for key, value in conditions.items():
        if value and isinstance(value, str) and value.strip().lower() not in ('not specified', 'null', 'n/a', ''):
            return True
    return False


# ============================================================
# LLM 批量解析
# ============================================================

LLM_PROMPT_TEMPLATE = """You are an expert chemistry IUPAC name parser.
For each compound name, determine if it is a **specific IUPAC chemical name** that can be parsed into a scaffold and substituents, or a **generic/symbol name** that cannot.

For parseable specific names, extract:
- scaffold: the core ring system or parent structure name (e.g. "aziridine", "cyclobutane", "benzene", "pyridine", "styrene", "indole").
  For acyclic compounds, use the main functional group backbone (e.g. "ketene", "alkene").
  Keep it simple — just the ring/system name, no locants.
- substituents: list of all substituent/functional groups attached to the scaffold.
  Each substituent should include locant numbers and stereochemistry where present.

For NON-parseable names (generic like "ketone derivative", or symbols like "1a"), set parseable=false.

Return ONLY a JSON array:
[
  {{"name": "<original name>", "parseable": true/false, "scaffold": "<name>" or null, "substituents": ["<sub1>", ...] or []}},
  ...
]

Compound names to parse:
{names_block}"""


def build_names_block(names: List[str]) -> str:
    lines = []
    for i, name in enumerate(names, 1):
        lines.append(f"{i}. {name}")
    return "\n".join(lines)


def call_llm_parse(names: List[str], client: OpenAI, model: str = "gpt-4o",
                   temperature: float = 0.0, max_retries: int = 3) -> Dict[str, dict]:
    """
    调用 LLM 批量解析化合物名称
    返回 {name: {"parseable": bool, "scaffold": str|None, "substituents": list}}
    """
    if not names:
        return {}

    names_block = build_names_block(names)
    prompt = LLM_PROMPT_TEMPLATE.format(names_block=names_block)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a chemistry IUPAC name parser. Always respond with valid JSON array only. No markdown, no explanation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=8000,
            )
            raw = response.choices[0].message.content.strip()
            result = _parse_llm_json(raw)
            return result
        except Exception as e:
            print(f"    [LLM] attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {}


def _parse_llm_json(raw: str) -> Dict[str, dict]:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _try_fix_json(raw)

    if not isinstance(data, list):
        raise ValueError("LLM response is not a JSON array")

    result = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        parseable = item.get("parseable", False)
        scaffold = item.get("scaffold")
        substituents = item.get("substituents", [])
        if not isinstance(substituents, list):
            substituents = []
        result[name] = {
            "parseable": bool(parseable),
            "scaffold": scaffold if scaffold else None,
            "substituents": substituents
        }
    return result


def _try_fix_json(raw: str) -> list:
    """尝试修复被截断或不完整的 JSON 数组"""
    # 先尝试补全未闭合的引号
    fixed = _fix_unterminated_strings(raw)

    # 找到 JSON 数组的开始
    start = fixed.find('[')
    if start == -1:
        return _extract_json_objects(raw)

    # 找到最后一个完整的对象结束位置
    array_content = fixed[start:]

    # 尝试在 "}]" 处截断
    last_complete = array_content.rfind('}]')
    if last_complete != -1:
        candidate = array_content[:last_complete + 2] + ']'
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 尝试在 "}," 处截断（最后一个对象不完整）
    last_obj_end = array_content.rfind('},')
    if last_obj_end != -1:
        candidate = array_content[:last_obj_end + 1] + ']'
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 回退到逐个提取
    return _extract_json_objects(raw)


def _fix_unterminated_strings(raw: str) -> str:
    """修复未闭合的字符串引号"""
    result = []
    in_string = False
    escape_next = False

    for ch in raw:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\':
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        result.append(ch)

    # 如果字符串未闭合，补上引号
    if in_string:
        result.append('"')

    return ''.join(result)


def _extract_json_objects(raw: str) -> list:
    """从文本中逐个提取 JSON 对象"""
    results = []
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for i, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(raw[start:i+1])
                    results.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    return results


def batch_llm_parse(names: List[str], client: OpenAI, model: str = "gpt-4o",
                    batch_size: int = 50) -> Dict[str, dict]:
    all_results = {}
    total_batches = (len(names) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(names), batch_size):
        batch = names[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        print(f"    [LLM] batch {batch_num}/{total_batches} ({len(batch)} names)")
        result = call_llm_parse(batch, client, model=model)
        all_results.update(result)
        if batch_num < total_batches:
            time.sleep(1)

    return all_results


# ============================================================
# 主流程
# ============================================================

def classify_and_enrich_reactions(reactions: List[dict], client: OpenAI,
                                   model: str = "gpt-4o",
                                   batch_size: int = 50,
                                   cache_path: Optional[Path] = None
                                   ) -> Tuple[List[dict], List[dict]]:
    """
    Q1: 有有效反应条件的反应（用于：给定底物名称 -> 预测最佳条件）
    Q2: Q1 的子集，至少一个底物可解析为骨架+取代基（用于：给定骨架+条件 -> 预测最佳取代基）

    1. 收集唯一底物名和产物名
    2. 正则快速过滤
    3. LLM 批量解析
    4. 充实底物和产物信息
    5. 分类：Q1 = 有有效条件，Q2 = Q1 中底物可解析的子集
    """

    # --- Step 1: 收集唯一名称 ---
    print("[Step 1] Collecting unique substrate and product names...")
    unique_names = set()
    for reaction in reactions:
        for substrate in reaction.get('substrates', []):
            name = substrate.get('name', '').strip()
            if name:
                unique_names.add(name)
        for product in reaction.get('products', []):
            name = product.get('name', '').strip()
            if name:
                unique_names.add(name)
    print(f"  Unique names (substrates + products): {len(unique_names)}")

    # --- Step 2: 加载缓存 ---
    parse_cache = {}
    if cache_path and cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            parse_cache = json.load(f)
        print(f"  Loaded cache: {len(parse_cache)} entries")

    # --- Step 3: 正则快速过滤 ---
    print("[Step 2] Regex quick filter...")
    regex_filtered = set()
    need_llm = []
    cached = set(parse_cache.keys())

    for name in sorted(unique_names):
        if name in cached:
            continue
        cat = quick_classify(name)
        if cat == 'Q2':
            regex_filtered.add(name)
            parse_cache[name] = {"parseable": False, "scaffold": None, "substituents": []}
        else:
            need_llm.append(name)

    print(f"  Regex filtered: {len(regex_filtered)}")
    print(f"  Need LLM: {len(need_llm)}")

    # --- Step 4: LLM 批量解析 ---
    print("[Step 3] LLM batch parse...")
    if need_llm:
        llm_results = batch_llm_parse(need_llm, client, model=model, batch_size=batch_size)
        for name in need_llm:
            if name in llm_results:
                info = llm_results[name]
                parse_cache[name] = {
                    "parseable": info.get("parseable", False),
                    "scaffold": info.get("scaffold"),
                    "substituents": info.get("substituents", []),
                }
            else:
                parse_cache[name] = {"parseable": False, "scaffold": None, "substituents": []}

    # 保存缓存
    if cache_path:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(parse_cache, f, ensure_ascii=False, indent=2)
        print(f"  Cache saved: {cache_path}")

    # --- Step 5: 充实信息并分类 ---
    print("[Step 4] Enriching compounds and classifying reactions...")
    Q1_reactions = []
    Q2_reactions = []

    def enrich_compound(compound: dict, cache: dict):
        name = compound.get('name', '').strip()
        info = cache.get(name, {})
        if info.get("parseable") and info.get("scaffold"):
            compound['scaffold'] = info['scaffold']
            compound['substituents'] = info.get('substituents', [])
            compound['parseable'] = True
        else:
            compound['scaffold'] = None
            compound['substituents'] = []
            compound['parseable'] = False

    for reaction in reactions:
        if not has_valid_condition(reaction):
            continue

        for substrate in reaction.get('substrates', []):
            enrich_compound(substrate, parse_cache)
        for product in reaction.get('products', []):
            enrich_compound(product, parse_cache)

        Q1_reactions.append(reaction)

        has_parseable_substrate = any(
            s.get('parseable') for s in reaction.get('substrates', [])
        )
        if has_parseable_substrate:
            Q2_reactions.append(reaction)

    return Q1_reactions, Q2_reactions


def split_reactions(input_path="filtered/merged_filtered_reactions.json",
                    output_dir="filtered",
                    cache_path=None,
                    model: str = "gpt-4o",
                    batch_size: int = 30,
                    api_key: Optional[str] = None,
                    base_url: str = "https://oneapi.xty.app/v1"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    cache_path = Path(cache_path) if cache_path else output_dir / "substrate_parse_cache.json"

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    reactions = data.get('reactions', [])
    print(f"Total reactions: {len(reactions)}")

    if not api_key:
        if dotenv_values is None:
            print("ERROR: python-dotenv is required. Install with: pip install python-dotenv")
            return
        env_values = dotenv_values(DOTENV_PATH)
        api_key = env_values.get("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: set OPENAI_API_KEY in .env")
        print(f"  .env path: {DOTENV_PATH}")
        return

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    Q1_reactions, Q2_reactions = classify_and_enrich_reactions(
        reactions, client, model=model, batch_size=batch_size, cache_path=cache_path
    )

    # 统计
    scaffold_types = {}
    Q1_substrate_count = 0
    Q1_product_count = 0
    product_scaffold_types = {}
    Q2_substrate_count = 0
    Q2_scaffold_types = {}

    for r in Q1_reactions:
        for s in r.get('substrates', []):
            if s.get('scaffold'):
                Q1_substrate_count += 1
                sc = s['scaffold'].lower()
                scaffold_types[sc] = scaffold_types.get(sc, 0) + 1
        for p in r.get('products', []):
            if p.get('scaffold'):
                Q1_product_count += 1
                sc = p['scaffold'].lower()
                product_scaffold_types[sc] = product_scaffold_types.get(sc, 0) + 1

    for r in Q2_reactions:
        for s in r.get('substrates', []):
            if s.get('scaffold'):
                Q2_substrate_count += 1
                sc = s['scaffold'].lower()
                Q2_scaffold_types[sc] = Q2_scaffold_types.get(sc, 0) + 1

    # 输出
    Q1_output = output_dir / "Q1_substrate_to_condition.json"
    with open(Q1_output, 'w', encoding='utf-8') as f:
        json.dump(Q1_reactions, f, ensure_ascii=False, indent=2)

    Q2_output = output_dir / "Q2_condition_to_substrate.json"
    with open(Q2_output, 'w', encoding='utf-8') as f:
        json.dump(Q2_reactions, f, ensure_ascii=False, indent=2)

    # 报告
    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Total reactions:        {len(reactions)}")
    print(f"Q1 reactions:           {len(Q1_reactions)}  (has valid reaction conditions)")
    print(f"Q2 reactions:           {len(Q2_reactions)}  (Q1 subset with parseable substrate scaffold+substituents)")
    print()
    print(f"Q1 substrate stats:")
    print(f"  Substrates with scaffold: {Q1_substrate_count}")
    print(f"  Top 10 scaffolds:")
    for sc, cnt in sorted(scaffold_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {sc}: {cnt}")
    print()
    print(f"Q1 product stats:")
    print(f"  Products with scaffold: {Q1_product_count}")
    print(f"  Top 10 scaffolds:")
    for sc, cnt in sorted(product_scaffold_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {sc}: {cnt}")
    print()
    print(f"Q2 substrate stats:")
    print(f"  Substrates with scaffold: {Q2_substrate_count}")
    print(f"  Top 10 scaffolds:")
    for sc, cnt in sorted(Q2_scaffold_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {sc}: {cnt}")
    print()
    print(f"Output files:")
    print(f"  Q1:    {Q1_output}")
    print(f"  Q2:    {Q2_output}")
    print(f"  Cache: {cache_path}")
    print("=" * 60)
    return {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "q1_output": str(Q1_output),
        "q2_output": str(Q2_output),
        "cache": str(cache_path),
        "total_reactions": len(reactions),
        "q1_reactions": len(Q1_reactions),
        "q2_reactions": len(Q2_reactions),
    }


if __name__ == "__main__":
    split_reactions()
