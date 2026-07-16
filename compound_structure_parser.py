import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from openai import OpenAI


LLM_PROMPT_TEMPLATE = """You are an expert chemistry IUPAC name parser.
For each compound name, determine if it is a specific IUPAC chemical name that can be parsed into a scaffold and substituents, or a generic/symbol name that cannot.

For parseable specific names, extract:
- scaffold: the core ring system or parent structure name (e.g. "aziridine", "cyclobutane", "benzene", "pyridine", "styrene", "indole").
  For acyclic compounds, use the main functional group backbone (e.g. "ketene", "alkene").
  Keep it simple: just the ring/system name, no locants.
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


def is_symbol_name(name: str) -> bool:
    if not name:
        return False
    return bool(re.match(r"^[0-9][a-zA-Z0-9]{0,4}$", name.strip()))


def is_obvious_generic(name: str) -> bool:
    if not name:
        return False
    n = name.strip().lower()
    exact = {
        "ketone derivative",
        "aldehyde derivative",
        "amine derivative",
        "acid derivative",
        "alcohol derivative",
        "ester derivative",
        "amide derivative",
        "nitrile derivative",
        "olefin derivative",
        "alkene derivative",
        "alkyne derivative",
        "vinyl derivative",
        "aryl derivative",
        "heteroaryl derivative",
        "phenol derivative",
        "compound",
        "product",
        "material",
        "starting material",
        "unknown",
        "not specified",
        "not reported",
        "n/a",
        "null",
    }
    if n in exact:
        return True
    generic_keywords = [
        "derivative",
        "substituted",
        "compound",
        "starting material",
        "product",
        "material",
        "mixture",
    ]
    return len(n) < 20 and any(keyword in n for keyword in generic_keywords)


def quick_classify(name: str) -> Optional[str]:
    if is_symbol_name(name) or is_obvious_generic(name):
        return "Q2"
    return None


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip())


def _cache_lookup_key(name: str) -> str:
    return _normalize_name(name).casefold()


def _normalize_parse_info(info: Optional[dict]) -> dict:
    info = info or {}
    scaffold = info.get("scaffold")
    if isinstance(scaffold, str):
        scaffold = scaffold.strip() or None
    substituents = info.get("substituents", [])
    if not isinstance(substituents, list):
        substituents = []
    parseable = bool(info.get("parseable", False))
    if scaffold and str(scaffold).lower() != "unknown":
        parseable = True
    return {
        "parseable": parseable,
        "scaffold": scaffold if scaffold else None,
        "substituents": substituents,
    }


def structure_from_existing(compound: dict) -> Optional[dict]:
    if not isinstance(compound, dict):
        return None
    scaffold = compound.get("scaffold")
    substituents = compound.get("substituents")
    parseable = compound.get("parseable")
    has_scaffold = bool(scaffold and str(scaffold).strip().lower() != "unknown")
    has_substituents = isinstance(substituents, list) and bool(substituents)
    if parseable is not None or has_scaffold or has_substituents:
        return _normalize_parse_info(
            {
                "parseable": parseable if parseable is not None else has_scaffold,
                "scaffold": scaffold,
                "substituents": substituents if isinstance(substituents, list) else [],
            }
        )
    return None


def _load_single_structure_cache(cache_path: Path) -> Dict[str, dict]:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}
    return {
        _normalize_name(name): _normalize_parse_info(info)
        for name, info in raw.items()
        if _normalize_name(name)
    }


def _legacy_structure_cache_paths(cache_path: Path) -> List[Path]:
    return [
        cache_path.parent / "multimodal_structure_parse_cache.json",
        cache_path.parent / "substrate_parse_cache.json",
    ]


def load_structure_cache(cache_path: Optional[Path], include_legacy: bool = True) -> Dict[str, dict]:
    if not cache_path:
        return {}
    cache_path = Path(cache_path)
    merged: Dict[str, dict] = {}
    sources = [cache_path]
    if include_legacy:
        sources.extend(path for path in _legacy_structure_cache_paths(cache_path) if path != cache_path)

    lookup: Dict[str, str] = {}
    for source in sources:
        for name, info in _load_single_structure_cache(source).items():
            key = _cache_lookup_key(name)
            existing_name = lookup.get(key)
            if existing_name:
                existing = merged.get(existing_name, {})
                if not existing.get("scaffold") and info.get("scaffold"):
                    merged[existing_name] = _normalize_parse_info(info)
                continue
            lookup[key] = name
            merged[name] = _normalize_parse_info(info)
    return merged


def write_structure_cache(cache_path: Optional[Path], cache: Dict[str, dict]) -> None:
    if not cache_path:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = {name: _normalize_parse_info(cache[name]) for name in sorted(cache, key=lambda x: x.casefold())}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)


def build_names_block(names: List[str]) -> str:
    return "\n".join(f"{idx}. {name}" for idx, name in enumerate(names, 1))


def call_llm_parse(
    names: List[str],
    client: OpenAI,
    model: str = "gpt-5-mini",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> Dict[str, dict]:
    if not names:
        return {}

    prompt = LLM_PROMPT_TEMPLATE.format(names_block=build_names_block(names))
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a chemistry IUPAC name parser. Always respond with valid JSON array only. No markdown, no explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            raw = response.choices[0].message.content.strip()
            return _parse_llm_json(raw)
        except Exception as exc:
            print(f"    [compound-structure] LLM attempt {attempt + 1} failed: {exc}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
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
        name = _normalize_name(item.get("name", ""))
        if not name:
            continue
        result[name] = _normalize_parse_info(item)
    return result


def _try_fix_json(raw: str) -> list:
    fixed = _fix_unterminated_strings(raw)
    start = fixed.find("[")
    if start == -1:
        return _extract_json_objects(raw)

    array_content = fixed[start:]
    last_complete = array_content.rfind("}]")
    if last_complete != -1:
        candidate = array_content[: last_complete + 2] + "]"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    last_obj_end = array_content.rfind("},")
    if last_obj_end != -1:
        candidate = array_content[: last_obj_end + 1] + "]"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return _extract_json_objects(raw)


def _fix_unterminated_strings(raw: str) -> str:
    result = []
    in_string = False
    escape_next = False
    for ch in raw:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        result.append(ch)
    if in_string:
        result.append('"')
    return "".join(result)


def _extract_json_objects(raw: str) -> list:
    results = []
    depth = 0
    start = None
    in_string = False
    escape_next = False

    for idx, ch in enumerate(raw):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = raw[start : idx + 1]
                try:
                    results.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass
                start = None
    return results


def batch_llm_parse(
    names: List[str],
    client: OpenAI,
    model: str = "gpt-5-mini",
    batch_size: int = 50,
) -> Dict[str, dict]:
    all_results = {}
    total_batches = (len(names) + batch_size - 1) // batch_size
    for batch_idx in range(0, len(names), batch_size):
        batch = names[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        print(f"    [compound-structure] LLM batch {batch_num}/{total_batches} ({len(batch)} names)")
        all_results.update(call_llm_parse(batch, client, model=model))
        if batch_num < total_batches:
            time.sleep(1)
    return all_results


def _iter_compounds(reactions: Sequence[dict], roles: Iterable[str]) -> Iterable[dict]:
    for reaction in reactions or []:
        if not isinstance(reaction, dict):
            continue
        for role in roles:
            for compound in reaction.get(role, []) or []:
                if isinstance(compound, dict):
                    yield compound


def collect_compound_names(reactions: Sequence[dict], roles: Iterable[str] = ("substrates", "products")) -> List[str]:
    names = {_normalize_name(compound.get("name", "")) for compound in _iter_compounds(reactions, roles)}
    return sorted((name for name in names if name), key=lambda x: x.casefold())


def collect_existing_structure_cache(
    reactions: Sequence[dict],
    roles: Iterable[str] = ("substrates", "products"),
) -> Dict[str, dict]:
    cache = {}
    for compound in _iter_compounds(reactions, roles):
        name = _normalize_name(compound.get("name", ""))
        if not name or name in cache:
            continue
        existing = structure_from_existing(compound)
        if existing is not None:
            cache[name] = existing
    return cache


def parse_compound_structures(
    names: List[str],
    cache_path: Optional[Path],
    client: OpenAI,
    model: str = "gpt-5-mini",
    batch_size: int = 50,
    existing_cache: Optional[Dict[str, dict]] = None,
    allow_llm: bool = True,
) -> Dict[str, dict]:
    names = sorted({_normalize_name(name) for name in names if _normalize_name(name)}, key=lambda x: x.casefold())
    parse_cache = load_structure_cache(cache_path)
    if existing_cache:
        for name, info in existing_cache.items():
            clean_name = _normalize_name(name)
            if clean_name:
                parse_cache[clean_name] = _normalize_parse_info(info)

    regex_filtered = []
    need_llm = []
    cache_lookup = {_cache_lookup_key(name): name for name in parse_cache}
    for name in names:
        if name in parse_cache:
            continue
        cached_name = cache_lookup.get(_cache_lookup_key(name))
        if cached_name:
            parse_cache[name] = _normalize_parse_info(parse_cache[cached_name])
            continue
        if quick_classify(name) == "Q2":
            parse_cache[name] = {"parseable": False, "scaffold": None, "substituents": []}
            regex_filtered.append(name)
        else:
            need_llm.append(name)

    print(f"  Structure cache entries: {len(parse_cache)}")
    print(f"  Existing/cache hits: {len([name for name in names if name in parse_cache])}")
    print(f"  Regex filtered: {len(regex_filtered)}")
    print(f"  Need LLM: {len(need_llm)}")

    if need_llm and allow_llm:
        llm_results = batch_llm_parse(need_llm, client, model=model, batch_size=batch_size)
        for name in need_llm:
            parse_cache[name] = _normalize_parse_info(llm_results.get(name))

    write_structure_cache(cache_path, parse_cache)
    return {name: parse_cache[name] for name in names if name in parse_cache}


def enrich_reactions_with_structure(
    reactions: List[dict],
    cache_path: Optional[Path],
    client: OpenAI,
    model: str = "gpt-5-mini",
    batch_size: int = 50,
    roles: Iterable[str] = ("substrates", "products"),
    allow_llm: bool = True,
) -> List[dict]:
    role_tuple = tuple(roles)
    names = collect_compound_names(reactions, role_tuple)
    existing_cache = collect_existing_structure_cache(reactions, role_tuple)
    print(f"[Structure parse] Unique compound names: {len(names)}")
    print(f"[Structure parse] Existing enriched names: {len(existing_cache)}")

    parse_cache = parse_compound_structures(
        names,
        cache_path=cache_path,
        client=client,
        model=model,
        batch_size=batch_size,
        existing_cache=existing_cache,
        allow_llm=allow_llm,
    )

    for compound in _iter_compounds(reactions, role_tuple):
        name = _normalize_name(compound.get("name", ""))
        info = parse_cache.get(name)
        if info is None:
            cached_name = {
                _cache_lookup_key(cache_name): cache_name
                for cache_name in parse_cache
            }.get(_cache_lookup_key(name))
            info = parse_cache.get(cached_name) if cached_name else None
        if info is None:
            compound.setdefault("parseable", False)
            compound.setdefault("substituents", [])
            continue
        compound["parseable"] = bool(info.get("parseable", False))
        compound["scaffold"] = info.get("scaffold")
        compound["substituents"] = info.get("substituents", [])

    return reactions
