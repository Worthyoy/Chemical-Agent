import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None


UNKNOWN_REACTION_TYPE = "unknown reaction"


CLASSIFY_SYSTEM_PROMPT = """You are a chemistry reaction-classification specialist.
Return valid JSON only."""


CLASSIFY_USER_PROMPT = """Assign the primary reaction type for each reaction.

Rules:
- Use a short, standard English main reaction class.
- Base the label on the paper title, section/procedure context when present, substrates, products, catalysts, ligands, other components, and conditions.
- Do not output mechanism details, catalyst names, optimization labels, or broad condition labels as the reaction type.
- Do not use "photocatalysis", "metal-catalyzed reaction", "optimization", or similar context labels as the main type unless the source explicitly defines the reaction that way.
- If the evidence is insufficient, output "unknown reaction".
- Do not choose from a provided list; create the best concise label from the evidence.

Return this JSON shape:
{{"reaction_types":[{{"index":0,"reaction_type":"short main type"}}]}}

Reactions:
{payload}
"""


NORMALIZE_SYSTEM_PROMPT = """You normalize chemistry reaction-type labels.
Return valid JSON only."""


NORMALIZE_USER_PROMPT = """Normalize these raw reaction-type labels.

Rules:
- Merge synonyms and near-synonyms into one concise, standard English main reaction type.
- Keep labels broad enough to be reusable across papers.
- Preserve "unknown reaction" exactly.
- Do not use a fixed taxonomy or candidate list; choose the best normalized label from the raw labels and examples.
- Return a mapping for every raw label.

Return this JSON shape:
{{"mapping":{{"raw label":"normalized label"}}}}

Raw labels with counts and examples:
{payload}
"""


def _clean_reaction_type(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none", "n/a", "not specified"}:
        return UNKNOWN_REACTION_TYPE
    return text


def _strip_json_fence(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    elif raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return raw


def _load_json_response(raw: str):
    raw = _strip_json_fence(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _reaction_summary(reaction: dict, index: int) -> dict:
    def names(field: str, limit: int = 8) -> List[str]:
        values = []
        for item in reaction.get(field, []) or []:
            if isinstance(item, dict):
                name = item.get("name") or item.get("symbol")
            else:
                name = item
            if name:
                values.append(str(name))
        return values[:limit]

    conditions = reaction.get("conditions") or {}
    targets = reaction.get("targets") or {}
    return {
        "index": index,
        "id": reaction.get("id"),
        "source_paper": reaction.get("source_paper"),
        "section": reaction.get("section") or reaction.get("chunk_label"),
        "substrates": names("substrates"),
        "products": names("products"),
        "intermediates": names("intermediates"),
        "catalysts": names("catalysts"),
        "ligands": names("ligands"),
        "other_components": names("other_components"),
        "conditions": {
            key: value
            for key, value in conditions.items()
            if value not in (None, "", "null")
        },
        "targets": {
            key: value for key, value in targets.items() if value not in (None, "", "null")
        },
        "step_count": reaction.get("step_count"),
        "raw_reaction_type": reaction.get("reaction_type"),
    }


def _chat_json(client, model: str, system_prompt: str, user_prompt: str, max_tokens: int = 8000):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )
    return _load_json_response(response.choices[0].message.content or "")


def _batched(items: List[dict], batch_size: int) -> Iterable[List[dict]]:
    batch_size = max(1, int(batch_size or 1))
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def classify_missing_reaction_types(
    reactions: List[dict],
    client,
    model: str,
    batch_size: int = 20,
) -> Dict[int, str]:
    missing = [
        index
        for index, reaction in enumerate(reactions)
        if _clean_reaction_type(reaction.get("reaction_type")) == UNKNOWN_REACTION_TYPE
    ]
    if not missing:
        return {}

    assignments: Dict[int, str] = {}
    summaries = [_reaction_summary(reactions[index], index) for index in missing]
    for batch_num, batch in enumerate(_batched(summaries, batch_size), start=1):
        print(f"  [Reaction Type] LLM classify batch {batch_num} ({len(batch)} reactions)")
        payload = json.dumps(batch, ensure_ascii=False, indent=2)
        data = _chat_json(
            client,
            model,
            CLASSIFY_SYSTEM_PROMPT,
            CLASSIFY_USER_PROMPT.format(payload=payload),
        )
        rows = data.get("reaction_types", []) if isinstance(data, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            index = row.get("index")
            if isinstance(index, int) and 0 <= index < len(reactions):
                assignments[index] = _clean_reaction_type(row.get("reaction_type"))
    return assignments


def normalize_reaction_type_labels(
    reactions: List[dict],
    client,
    model: str,
    examples_per_label: int = 3,
) -> Dict[str, str]:
    labels = [_clean_reaction_type(r.get("reaction_type")) for r in reactions]
    counts = Counter(labels)
    examples_by_label: Dict[str, List[dict]] = defaultdict(list)
    for index, reaction in enumerate(reactions):
        label = _clean_reaction_type(reaction.get("reaction_type"))
        if len(examples_by_label[label]) < examples_per_label:
            examples_by_label[label].append(_reaction_summary(reaction, index))

    payload = json.dumps(
        [
            {
                "raw_label": label,
                "count": counts[label],
                "examples": examples_by_label[label],
            }
            for label in sorted(counts, key=lambda x: (-counts[x], x.casefold()))
        ],
        ensure_ascii=False,
        indent=2,
    )
    print(f"  [Reaction Type] LLM normalize {len(counts)} raw labels")
    data = _chat_json(
        client,
        model,
        NORMALIZE_SYSTEM_PROMPT,
        NORMALIZE_USER_PROMPT.format(payload=payload),
    )
    mapping = data.get("mapping", {}) if isinstance(data, dict) else {}
    if not isinstance(mapping, dict):
        mapping = {}
    result = {}
    for raw_label in counts:
        normalized = _clean_reaction_type(mapping.get(raw_label, raw_label))
        result[raw_label] = normalized
    result[UNKNOWN_REACTION_TYPE] = UNKNOWN_REACTION_TYPE
    return result


def normalize_reaction_types(
    reactions: List[dict],
    model: str = "gpt-5-mini",
    api_key: Optional[str] = None,
    base_url: str = "https://hk.xty.app/v1",
    batch_size: int = 20,
) -> dict:
    if OpenAI is None:
        raise ImportError("openai package is required for LLM reaction type normalization")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    for reaction in reactions:
        reaction["reaction_type"] = _clean_reaction_type(reaction.get("reaction_type"))

    assignments = classify_missing_reaction_types(
        reactions,
        client,
        model=model,
        batch_size=batch_size,
    )
    for index, reaction_type in assignments.items():
        reactions[index]["reaction_type"] = reaction_type

    raw_counts = Counter(_clean_reaction_type(r.get("reaction_type")) for r in reactions)
    mapping = normalize_reaction_type_labels(reactions, client, model=model)

    for reaction in reactions:
        raw_type = _clean_reaction_type(reaction.get("reaction_type"))
        reaction["reaction_type"] = mapping.get(raw_type, raw_type)

    normalized_counts = Counter(_clean_reaction_type(r.get("reaction_type")) for r in reactions)
    return {
        "status": "success",
        "model": model,
        "total_reactions": len(reactions),
        "classified_missing": len(assignments),
        "raw_label_counts": dict(sorted(raw_counts.items())),
        "mapping": dict(sorted(mapping.items())),
        "normalized_label_counts": dict(sorted(normalized_counts.items())),
    }


def normalize_reaction_types_file(
    input_path,
    output_path=None,
    report_path=None,
    model: str = "gpt-5-mini",
    api_key: Optional[str] = None,
    base_url: str = "https://hk.xty.app/v1",
    batch_size: int = 20,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path
    report_path = Path(report_path) if report_path else output_path.with_name(
        output_path.stem + "_reaction_type_report.json"
    )

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reactions = data.get("reactions", []) if isinstance(data, dict) else []
    if not isinstance(reactions, list):
        reactions = []

    report = normalize_reaction_types(
        reactions,
        model=model,
        api_key=api_key,
        base_url=base_url,
        batch_size=batch_size,
    )
    data["reaction_type_normalization"] = report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        **report,
        "input": str(input_path),
        "output": str(output_path),
        "report_output": str(report_path),
    }
