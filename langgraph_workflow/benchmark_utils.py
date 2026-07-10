import json
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def parse_percentage(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        number = float(val)
        return number if 0.0 <= number <= 100.0 else None

    text = str(val).strip().replace("％", "%")
    if not text:
        return None

    percentage_matches = re.findall(
        r"(?<![\d.])(?:[<>≤≥~≈]\s*)?(\d+(?:\.\d+)?)\s*%",
        text,
    )
    if len(percentage_matches) == 1:
        number = float(percentage_matches[0])
        return number if 0.0 <= number <= 100.0 else None
    if len(percentage_matches) > 1:
        return None

    if re.fullmatch(r"\s*[<>≤≥~≈]?\s*\d+(?:\.\d+)?\s*", text):
        number_match = re.search(r"\d+(?:\.\d+)?", text)
        number = float(number_match.group(0)) if number_match else None
        if number is not None and 0.0 <= number <= 100.0:
            return number
    return None


UNKNOWN_REACTION_TYPE = "unknown reaction"


def extract_reaction_type(reaction_or_paper=None) -> str:
    if isinstance(reaction_or_paper, dict):
        value = reaction_or_paper.get("reaction_type")
    else:
        value = None
    value = str(value or "").strip()
    if not value or value.casefold() in {"null", "none", "n/a", "not specified"}:
        return UNKNOWN_REACTION_TYPE
    return value


def _is_unknown_reaction_type(reaction_type: str) -> bool:
    return str(reaction_type or "").strip().casefold() == UNKNOWN_REACTION_TYPE


def get_scaffold_combo(substrates: List[dict]) -> Optional[Tuple[str, ...]]:
    scaffolds = sorted(
        [
            s.get("scaffold", "")
            for s in substrates
            if s.get("scaffold") and s.get("scaffold") != "unknown"
        ]
    )
    return tuple(scaffolds) if scaffolds else None


def find_key_substrate_in_group(entries: List[dict]) -> Optional[dict]:
    all_substrates = []
    for entry in entries:
        for substrate in entry.get("substrates", []):
            all_substrates.append(substrate)

    for substrate in all_substrates:
        if substrate.get("substituents") and substrate.get("scaffold"):
            return substrate

    for substrate in all_substrates:
        if substrate.get("scaffold") and substrate.get("scaffold") != "unknown":
            return substrate

    return None


def get_conditions_key(conditions: Dict) -> Optional[str]:
    if not conditions:
        return None
    return json.dumps(conditions, sort_keys=True, ensure_ascii=False)


def format_conditions_en(conditions: Dict) -> str:
    if not conditions:
        return "not specified"
    parts = []
    for key in sorted(conditions.keys(), key=lambda x: str(x).casefold()):
        value = conditions.get(key)
        if value is None or value == "":
            continue
        label = str(key).replace("_", " ")
        if isinstance(value, list):
            step_parts = []
            for item in value:
                if not isinstance(item, dict) or item.get("value") in (None, ""):
                    continue
                step = item.get("step")
                prefix = f"Step {step}" if step not in (None, "") else "Step"
                step_parts.append(f"{prefix}: {item.get('value')}")
            if step_parts:
                parts.append(f"{label}: " + "; ".join(step_parts))
            continue
        parts.append(f"{label}: {value}")
    return "; ".join(parts) if parts else "not specified"


def _compound_public_fields(compound: dict) -> dict:
    public = {
        "name": compound.get("name", ""),
        "scaffold": compound.get("scaffold"),
        "substituents": compound.get("substituents", []),
    }
    if compound.get("symbol") is not None:
        public["symbol"] = compound.get("symbol")
    if compound.get("step") is not None:
        public["step"] = compound.get("step")
    return public


def _compound_identity(compound: dict) -> Tuple[str, str, Tuple[str, ...]]:
    return (
        _normalize_compound_name(compound.get("name", "")).casefold(),
        str(compound.get("scaffold") or "").casefold(),
        tuple(str(s).casefold() for s in compound.get("substituents", []) or []),
    )


def _substrates_for_scaffold(substrates: List[dict], scaffold: str) -> List[dict]:
    scaffold_key = str(scaffold or "").casefold()
    return [
        sub
        for sub in substrates or []
        if str(sub.get("scaffold") or "").casefold() == scaffold_key
    ]


def _fixed_substrate_key(substrates: List[dict], variable_scaffold: str) -> Tuple[Tuple[str, str, Tuple[str, ...]], ...]:
    fixed = [
        _compound_identity(sub)
        for sub in substrates or []
        if str(sub.get("scaffold") or "").casefold()
        != str(variable_scaffold or "").casefold()
    ]
    return tuple(sorted(fixed))


def _clean_substrates_except_scaffold(substrates: List[dict], variable_scaffold: str) -> List[dict]:
    fixed = []
    for sub in substrates or []:
        scaffold = sub.get("scaffold")
        if not scaffold or scaffold == "unknown":
            continue
        if str(scaffold).casefold() == str(variable_scaffold or "").casefold():
            continue
        fixed.append(_compound_public_fields(sub))
    return sorted(fixed, key=lambda x: str(x.get("name", "")).casefold())


def _product_scaffold_class(entries: List[dict]) -> str:
    scaffolds = set()
    for entry in entries:
        for product in entry.get("products", []) or []:
            scaffold = product.get("scaffold")
            if scaffold and scaffold != "unknown":
                scaffolds.add(str(scaffold))
    if len(scaffolds) == 1:
        return f"{next(iter(scaffolds))} product"
    return "corresponding reaction product"


def _option_sort_key(option: dict) -> Tuple[float, str]:
    return (
        -float(option["metadata_hidden"]["score"]),
        str(option["metadata_hidden"]["original_answer_index"]),
    )


def _shuffle_options(options: List[dict], seed: str) -> List[dict]:
    shuffled = list(options)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    for idx, option in enumerate(shuffled, start=1):
        option["option_id"] = f"O{idx}"
    return shuffled


def _build_q2_question_en(
    reaction_type: str,
    conditions: Dict,
    fixed_substrates: List[dict],
    variable_scaffold: str,
    product_class: str,
) -> str:
    fixed_text = "; ".join(
        f"{sub.get('name', '')} (scaffold: {sub.get('scaffold', 'unknown')})"
        for sub in fixed_substrates
    )
    if not fixed_text:
        fixed_text = "none"

    return "\n".join(
        [
            f"Reaction type: {reaction_type}.",
            f"Reaction conditions: {format_conditions_en(conditions)}.",
            f"Fixed substrate(s): {fixed_text}.",
            f"Variable substrate scaffold: {variable_scaffold}.",
            f"Product scaffold/class: {product_class}.",
            (
                "Question: Among substrates with the specified variable scaffold, "
                "which substituent pattern on that substrate gives the best combined ee and yield?"
            ),
        ]
    )


def _normalize_compound_name(name: str) -> str:
    return str(name or "").strip()


def _is_valid_compound_name(name: str) -> bool:
    n = _normalize_compound_name(name)
    if not n:
        return False
    return n.lower() not in {"not specified", "n/a", "null"}


def _substrate_combo_key(substrates: List[dict]) -> Optional[Tuple[str, ...]]:
    names = []
    for sub in substrates or []:
        sub_name = sub.get("name", "")
        if _is_valid_compound_name(sub_name):
            names.append(_normalize_compound_name(sub_name))
    if not names:
        return None

    # Order-insensitive key: sort normalized names.
    return tuple(sorted(names, key=lambda x: x.casefold()))


def _generate_q1_benchmark_legacy(q1_data: List[dict]) -> List[dict]:
    combo_data: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)

    for reaction in q1_data:
        substrates = reaction.get("substrates", [])
        combo_key = _substrate_combo_key(substrates)
        if not combo_key:
            continue

        substrate_info = []
        for sub in substrates:
            sub_name = sub.get("name", "")
            if _is_valid_compound_name(sub_name):
                substrate_info.append(
                    {
                        "name": _normalize_compound_name(sub_name),
                        "scaffold": sub.get("scaffold"),
                        "substituents": sub.get("substituents", []),
                    }
                )

        if not substrate_info:
            continue

        cond = reaction.get("conditions", {})
        targets = reaction.get("targets", {})

        ee = parse_percentage(targets.get("ee"))
        yield_val = parse_percentage(targets.get("yield"))

        combo_data[combo_key].append(
            {
                "substrates": substrate_info,
                "conditions": cond,
                "ee": ee,
                "yield": yield_val,
                "er": targets.get("er"),
                "reaction_id": reaction.get("id"),
                "source_paper": reaction.get("source_paper"),
            }
        )

    q1_benchmark: List[dict] = []

    for combo_key in sorted(combo_data.keys(), key=lambda k: " | ".join(k).casefold()):
        entries = combo_data[combo_key]
        if not entries:
            continue

        combo_str = " + ".join(combo_key)
        question = (
            f"For the substrate combination {combo_str}, "
            f"which reaction conditions give the best ee and yield?"
        )

        answers = []
        for entry in entries:
            score = 0
            count = 0
            if entry["ee"] is not None:
                score += entry["ee"]
                count += 1
            if entry["yield"] is not None:
                score += entry["yield"]
                count += 1
            score = score / count if count > 0 else 0

            answers.append(
                {
                    "substrates": entry["substrates"],
                    "conditions": entry["conditions"],
                    "ee": entry["ee"],
                    "yield": entry["yield"],
                    "er": entry["er"],
                    "score": score,
                    "reaction_id": entry.get("reaction_id"),
                    "source_paper": entry.get("source_paper"),
                }
            )

        answers.sort(key=lambda x: x["score"], reverse=True)

        source_papers = sorted(
            {a.get("source_paper") for a in answers if a.get("source_paper")},
            key=lambda x: str(x).casefold(),
        )

        q1_benchmark.append(
            {
                "id": f"Q1_{len(q1_benchmark) + 1}",
                "reaction_type": UNKNOWN_REACTION_TYPE,
                "substrate_combo": list(combo_key),
                "question": question,
                "source_papers": source_papers,
                "answers": answers,
            }
        )

    return q1_benchmark


def _score_from_targets(targets: Dict) -> Tuple[Optional[float], Optional[float], float, int]:
    ee = parse_percentage((targets or {}).get("ee"))
    yield_val = parse_percentage((targets or {}).get("yield"))
    score = 0.0
    count = 0
    if ee is not None:
        score += ee
        count += 1
    if yield_val is not None:
        score += yield_val
        count += 1
    return ee, yield_val, score / count if count else 0.0, count


def _public_condition_option(reaction: dict) -> dict:
    public = {"conditions": _clean_public_conditions(reaction.get("conditions") or {})}
    catalysts = _catalyst_names_public(reaction.get("catalysts", []))
    if catalysts:
        public["catalysts"] = catalysts
    ligands = _condition_components_public(reaction.get("ligands", []), include_amount=False)
    if ligands:
        public["ligands"] = ligands
    other_components = _condition_components_public(
        reaction.get("other_components", []), include_amount=True
    )
    if other_components:
        public["other_components"] = other_components
    for key in (
        "additives",
        "reagents",
        "electrodes",
        "atmosphere",
        "scale",
    ):
        value = reaction.get(key)
        if value not in (None, "", [], {}):
            public[key] = value
    return public


def _public_option_key(option: dict) -> str:
    return json.dumps(option, sort_keys=True, ensure_ascii=False)


def _canonical_grouping_text(value, *, mode: str = "condition") -> str:
    """Normalize display-only text into a conservative benchmark grouping key.

    This is intentionally less aggressive than chemical entity normalization: it
    removes punctuation/encoding noise that should not split Q2 groups, but it
    does not infer missing quantities or merge aliases.
    """
    text = str(value or "").strip().casefold()
    if not text:
        return ""

    replacements = {
        "渭": "u",
        "碌": "u",
        "μ": "u",
        "µ": "u",
        "掳": "deg",
        "°": "deg",
        "–": "-",
        "—": "-",
        "−": "-",
        "，": ",",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\bhours?\b", "h", text)
    text = re.sub(r"\bhrs?\b", "h", text)
    text = re.sub(r"\bminutes?\b", "min", text)
    text = re.sub(r"\bmins?\b", "min", text)
    text = re.sub(r"\bmicroliters?\b", "ul", text)
    text = re.sub(r"\bmicrolitres?\b", "ul", text)
    text = re.sub(r"\bu\s*l\b", "ul", text)
    text = re.sub(r"\bdeg\s*c\b", "degc", text)
    text = re.sub(r"\bmol\s*%\b", "mol%", text)

    if mode in {"condition", "amount"}:
        # For grouping, punctuation and parentheses around condition/amount
        # qualifiers should not distinguish otherwise identical text.
        text = re.sub(r"[()\[\]{}]", " ", text)
        text = re.sub(r"[,;:]", " ", text)
    else:
        # Names are normalized conservatively. Do not remove parentheses from
        # chemical names such as Pd2(dba)3 or (R,R)-QuinoxP*.
        text = re.sub(r"[,;:]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _canonical_grouping_value(value, *, mode: str = "condition"):
    if _is_empty_public_value(value):
        return None
    if isinstance(value, str):
        canonical = _canonical_grouping_text(value, mode=mode)
        return canonical or None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        items = [
            _canonical_grouping_value(item, mode=mode)
            for item in value
        ]
        items = [item for item in items if not _is_empty_public_value(item)]
        if not items:
            return None
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
        )
    if isinstance(value, dict):
        cleaned = {}
        for key, item_value in value.items():
            item_mode = "condition"
            if str(key) == "name":
                item_mode = "name"
            elif str(key) == "amount":
                item_mode = "amount"
            canonical = _canonical_grouping_value(item_value, mode=item_mode)
            if not _is_empty_public_value(canonical):
                cleaned[key] = canonical
        return cleaned or None
    return _canonical_grouping_text(value, mode=mode) or None


def _condition_grouping_signature(public_condition_option: dict) -> dict:
    """Return the canonical Q2 grouping signature for reaction conditions.

    The original public condition option is kept for display. This canonical
    version is used only for grouping/key generation, so harmless punctuation
    differences do not split benchmark questions.
    """
    canonical = _canonical_grouping_value(public_condition_option or {}, mode="condition")
    return canonical if isinstance(canonical, dict) else {}


def _substrate_combo_public(substrates: List[dict]) -> List[dict]:
    substrate_info = []
    for sub in substrates or []:
        sub_name = sub.get("name", "")
        if _is_valid_compound_name(sub_name):
            clean = _compound_public_fields(sub)
            clean["name"] = _normalize_compound_name(sub_name)
            substrate_info.append(clean)
    return sorted(substrate_info, key=lambda x: str(x.get("name", "")).casefold())


def _product_combo_public(products: List[dict]) -> List[dict]:
    product_info = []
    for product in products or []:
        name = product.get("name", "")
        if _is_valid_product_name(name, product.get("scaffold")):
            clean = {
                "name": _normalize_compound_name(name),
                "scaffold": product.get("scaffold"),
            }
            product_info.append(clean)
    return sorted(product_info, key=lambda x: str(x.get("name", "")).casefold())


def _is_valid_product_name(name: str, scaffold: Optional[str] = None) -> bool:
    n = _normalize_compound_name(name)
    if not _is_valid_compound_name(n):
        return False
    if scaffold and n.casefold() == str(scaffold).strip().casefold():
        return False
    lower = n.casefold()
    if lower in {"unknown product", "heterodimer"}:
        return False
    if re.fullmatch(r"(product|compound|adduct)\s*\w+", lower):
        return False
    if re.fullmatch(r"\w{1,3}", lower):
        return False
    if re.search(r"\b(solid|oil|liquid|foam|powder|crystal|residue)\b", lower):
        return False
    return True


def _catalyst_names_public(catalysts: List[dict]) -> List[dict]:
    return _condition_components_public(catalysts, include_amount=True)


def _condition_components_public(items: List[dict], *, include_amount: bool) -> List[dict]:
    names = []
    for item in items or []:
        name = item.get("name", "")
        if _is_valid_compound_name(name):
            public = {"name": _normalize_compound_name(name)}
            if include_amount and item.get("amount") not in (None, ""):
                public["amount"] = item.get("amount")
            if item.get("step") is not None:
                public["step"] = item.get("step")
            names.append(public)
    return sorted(
        names,
        key=lambda x: (
            str(x.get("name", "")).casefold(),
            str(x.get("amount", "")).casefold(),
            str(x.get("step", "")),
        ),
    )


def _is_empty_public_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not any(not _is_empty_public_value(item) for item in value)
    if isinstance(value, dict):
        return not any(not _is_empty_public_value(item) for item in value.values())
    return False


def _clean_public_conditions(conditions: Dict) -> Dict:
    if not isinstance(conditions, dict):
        return {}
    cleaned = {}
    for key, value in conditions.items():
        if _is_empty_public_value(value):
            continue
        if isinstance(value, list):
            items = []
            for item in value:
                if _is_empty_public_value(item):
                    continue
                if isinstance(item, dict):
                    if "value" in item and _is_empty_public_value(item.get("value")):
                        continue
                    clean_item = {
                        item_key: item_value
                        for item_key, item_value in item.items()
                        if not _is_empty_public_value(item_value)
                    }
                    if clean_item:
                        items.append(clean_item)
                else:
                    items.append(item)
            if items:
                cleaned[key] = items
            continue
        if isinstance(value, dict):
            clean_value = {
                item_key: item_value
                for item_key, item_value in value.items()
                if not _is_empty_public_value(item_value)
            }
            if clean_value:
                cleaned[key] = clean_value
            continue
        cleaned[key] = value
    return cleaned


def _public_combo_key(items: List[dict]) -> Optional[str]:
    if not items:
        return None
    return json.dumps(items, sort_keys=True, ensure_ascii=False)


def _q1_review_context(
    paper: str,
    reaction_type: str,
    combo_key: Tuple[str, ...],
    product_combo: Optional[List[dict]] = None,
) -> dict:
    context = {
        "source_paper": paper,
        "reaction_type": reaction_type,
        "substrate_combo": list(combo_key),
    }
    if product_combo is not None:
        context["product_combo"] = product_combo
    return context


def _format_named_scaffold(items: List[dict]) -> str:
    return "; ".join(
        f"{item.get('name', '')} (scaffold: {item.get('scaffold') or 'unknown'})"
        for item in items
    )


def _build_q1_question_en(
    reaction_type: str,
    substrates: List[dict],
    products: List[dict],
) -> str:
    substrate_text = "; ".join(
        (
            f"{sub.get('name', '')}"
            f" (scaffold: {sub.get('scaffold') or 'unknown'})"
        )
        for sub in substrates
    )
    return "\n".join(
        [
            f"Reaction type: {reaction_type}.",
            f"Substrate combination: {substrate_text}.",
            f"Product(s): {_format_named_scaffold(products)}.",
            (
                "Question: Which complete reaction conditions, including catalyst "
                "if applicable, are expected to give the best combined ee and yield?"
            ),
        ]
    )


def generate_q1_benchmark_package(q1_data: List[dict]) -> dict:
    paper_combo_data: Dict[Tuple[str, str, Tuple[str, ...], str], List[dict]] = defaultdict(list)
    review_set: List[dict] = []

    for reaction in q1_data:
        paper = reaction.get("source_paper", "")
        if not paper:
            continue
        reaction_type = extract_reaction_type(reaction)

        combo_key = _substrate_combo_key(reaction.get("substrates", []))
        if not combo_key:
            continue

        if _is_unknown_reaction_type(reaction_type):
            review_set.append(
                {
                    "reason": "unknown_reaction_type",
                    "reaction_id": reaction.get("id"),
                    **_q1_review_context(paper, reaction_type, combo_key),
                }
            )
            continue

        product_combo = _product_combo_public(reaction.get("products", []))
        product_key = _public_combo_key(product_combo)
        if not product_key:
            review_set.append(
                {
                    "reason": "missing_product",
                    "reaction_id": reaction.get("id"),
                    **_q1_review_context(paper, reaction_type, combo_key),
                }
            )
            continue

        paper_combo_data[(paper, reaction_type, combo_key, product_key)].append(reaction)

    q1_benchmark: List[dict] = []

    for (paper, reaction_type, combo_key, product_key), entries in sorted(
        paper_combo_data.items(),
        key=lambda item: (
            str(item[0][0]).casefold(),
            str(item[0][1]).casefold(),
            " | ".join(item[0][2]).casefold(),
            str(item[0][3]).casefold(),
        ),
    ):
        if len(entries) < 2:
            continue

        substrate_combo = _substrate_combo_public(entries[0].get("substrates", []))
        if not substrate_combo:
            continue
        product_combo = json.loads(product_key)

        options = []
        omitted_no_targets = []
        for original_answer_index, reaction in enumerate(entries):
            targets = reaction.get("targets", {}) or {}
            ee, yield_val, score, target_count = _score_from_targets(targets)
            if target_count == 0:
                omitted_no_targets.append(reaction.get("id"))
                continue

            options.append(
                {
                    "option_id": None,
                    **_public_condition_option(reaction),
                    "metadata_hidden": {
                        "original_answer_index": original_answer_index,
                        "reaction_id": reaction.get("id"),
                        "ee_raw": targets.get("ee"),
                        "yield_raw": targets.get("yield"),
                        "ee": ee,
                        "yield": yield_val,
                        "er": targets.get("er"),
                        "score": score,
                        "source_paper": reaction.get("source_paper"),
                    },
                }
            )

        if len(options) < 2:
            if omitted_no_targets:
                review_set.append(
                    {
                        "reason": "insufficient_scored_options",
                        **_q1_review_context(
                            paper,
                            reaction_type,
                            combo_key,
                            product_combo,
                        ),
                        "omitted_no_targets": omitted_no_targets,
                    }
                )
            continue

        by_public_key = defaultdict(list)
        for option in options:
            public_option = {
                key: value for key, value in option.items() if key != "metadata_hidden"
            }
            by_public_key[_public_option_key(public_option)].append(option)

        conflicting_duplicate_keys = []
        deduped_options = []
        for duplicate_options in by_public_key.values():
            scores = {
                duplicate["metadata_hidden"]["score"]
                for duplicate in duplicate_options
            }
            if len(duplicate_options) > 1 and len(scores) > 1:
                conflicting_duplicate_keys.append(
                    {
                        "reaction_ids": [
                            duplicate["metadata_hidden"]["reaction_id"]
                            for duplicate in duplicate_options
                        ],
                        "scores": sorted(scores, reverse=True),
                    }
                )
            else:
                deduped_options.append(duplicate_options[0])

        if conflicting_duplicate_keys:
            review_set.append(
                {
                    "reason": "duplicate_public_conditions_conflicting_results",
                    **_q1_review_context(
                        paper,
                        reaction_type,
                        combo_key,
                        product_combo,
                    ),
                    "details": conflicting_duplicate_keys,
                }
            )
            continue

        if len(deduped_options) < 2:
            review_set.append(
                {
                    "reason": "insufficient_distinct_conditions",
                    **_q1_review_context(
                        paper,
                        reaction_type,
                        combo_key,
                        product_combo,
                    ),
                    "option_count": len(deduped_options),
                }
            )
            continue

        ranked_options = sorted(deduped_options, key=_option_sort_key)
        max_score = ranked_options[0]["metadata_hidden"]["score"]

        shuffled_options = _shuffle_options(deduped_options, f"{paper}|{combo_key}|{product_key}")
        by_original_index = {
            option["metadata_hidden"]["original_answer_index"]: option
            for option in shuffled_options
        }
        gold_option_ids = [
            option["option_id"]
            for option in shuffled_options
            if option["metadata_hidden"]["score"] == max_score
        ]
        gold_ranked_option_ids = [
            by_original_index[option["metadata_hidden"]["original_answer_index"]][
                "option_id"
            ]
            for option in ranked_options
        ]

        public_options = [
            {key: value for key, value in option.items() if key != "metadata_hidden"}
            for option in shuffled_options
        ]

        q1_benchmark.append(
            {
                "id": f"Q1_{len(q1_benchmark) + 1}",
                "reaction_type": reaction_type,
                "substrate_combo": substrate_combo,
                "product_combo": product_combo,
                "question_en": _build_q1_question_en(
                    reaction_type,
                    substrate_combo,
                    product_combo,
                ),
                "source_paper": paper,
                "option_count": len(public_options),
                "options": public_options,
                "gold_option_ids": sorted(gold_option_ids),
                "gold_ranked_option_ids": gold_ranked_option_ids,
                "metric_eligibility": {
                    "top1": True,
                    "top3": len(public_options) >= 3,
                },
                "metadata_hidden": {
                    "option_results": [
                        {
                            "option_id": option["option_id"],
                            **option["metadata_hidden"],
                        }
                        for option in shuffled_options
                    ],
                    "has_top_score_tie": len(gold_option_ids) > 1,
                    "omitted_no_targets": omitted_no_targets,
                },
            }
        )

    option_count_distribution = {
        str(count): sum(1 for item in q1_benchmark if item["option_count"] == count)
        for count in sorted({item["option_count"] for item in q1_benchmark})
    }

    report = {
        "input_groups": len(paper_combo_data),
        "main_questions": len(q1_benchmark),
        "review_questions": len(review_set),
        "min_options": min((item["option_count"] for item in q1_benchmark), default=0),
        "max_options": max((item["option_count"] for item in q1_benchmark), default=0),
        "option_count_distribution": option_count_distribution,
        "top3_eligible_questions": sum(
            1 for item in q1_benchmark if item["metric_eligibility"]["top3"]
        ),
        "top3_ineligible_questions": sum(
            1 for item in q1_benchmark if not item["metric_eligibility"]["top3"]
        ),
        "top_score_tie_questions": sum(
            1 for item in q1_benchmark if item["metadata_hidden"]["has_top_score_tie"]
        ),
        "review_reasons": dict(
            sorted(
                {
                    reason: sum(1 for item in review_set if item["reason"] == reason)
                    for reason in {item["reason"] for item in review_set}
                }.items()
            )
        ),
    }

    return {
        "benchmark": q1_benchmark,
        "review_set": review_set,
        "report": report,
    }


def generate_q1_benchmark(q1_data: List[dict]) -> List[dict]:
    return generate_q1_benchmark_package(q1_data)["benchmark"]


def generate_q2_benchmark_package(q2_data: List[dict]) -> dict:
    paper_combo_data: Dict[Tuple[str, str, Tuple[str, ...], str], List[dict]] = defaultdict(list)
    review_set = []

    for reaction in q2_data:
        paper = reaction.get("source_paper", "")
        if not paper:
            continue
        reaction_type = extract_reaction_type(reaction)

        substrates = reaction.get("substrates", [])
        if not substrates:
            continue

        combo = get_scaffold_combo(substrates)
        if not combo:
            continue

        if _is_unknown_reaction_type(reaction_type):
            review_set.append(
                {
                    "reason": "unknown_reaction_type",
                    "reaction_id": reaction.get("id"),
                    "source_paper": paper,
                    "reaction_type": reaction_type,
                    "scaffold_combo": list(combo),
                }
            )
            continue

        condition_signature = _public_condition_option(reaction)
        condition_grouping_signature = _condition_grouping_signature(condition_signature)
        cond_key = _public_option_key(condition_grouping_signature)

        key = (paper, reaction_type, combo, cond_key)
        paper_combo_data[key].append(reaction)

    q2_benchmark = []

    for (paper, reaction_type, combo, cond_key), entries in sorted(
        paper_combo_data.items(),
        key=lambda item: (
            str(item[0][0]).casefold(),
            str(item[0][1]).casefold(),
            " | ".join(item[0][2]).casefold(),
            str(item[0][3]).casefold(),
        ),
    ):
        first_entry = entries[0]
        conditions = first_entry.get("conditions", {})
        condition_signature = _public_condition_option(first_entry)
        condition_grouping_signature = _condition_grouping_signature(condition_signature)

        for variable_scaffold in combo:
            variable_candidates = []
            bad_entries = []

            for entry_index, reaction in enumerate(entries):
                matches = _substrates_for_scaffold(
                    reaction.get("substrates", []), variable_scaffold
                )
                if len(matches) != 1:
                    bad_entries.append(
                        {
                            "reaction_id": reaction.get("id"),
                            "match_count": len(matches),
                        }
                    )
                    continue
                variable_candidates.append((entry_index, reaction, matches[0]))

            if bad_entries:
                review_set.append(
                    {
                        "reason": "variable_scaffold_not_unique",
                        "source_paper": paper,
                        "reaction_type": reaction_type,
                        "conditions": conditions,
                        "condition_signature": condition_signature,
                        "condition_grouping_signature": condition_grouping_signature,
                        "scaffold_combo": list(combo),
                        "variable_scaffold": variable_scaffold,
                        "details": bad_entries,
                    }
                )
                continue

            fixed_keys = {
                _fixed_substrate_key(reaction.get("substrates", []), variable_scaffold)
                for _, reaction, _ in variable_candidates
            }
            if len(fixed_keys) != 1:
                review_set.append(
                    {
                        "reason": "fixed_substrate_mismatch",
                        "source_paper": paper,
                        "reaction_type": reaction_type,
                        "conditions": conditions,
                        "condition_signature": condition_signature,
                        "condition_grouping_signature": condition_grouping_signature,
                        "scaffold_combo": list(combo),
                        "variable_scaffold": variable_scaffold,
                        "reaction_ids": [reaction.get("id") for _, reaction, _ in variable_candidates],
                    }
                )
                continue

            first_reaction = variable_candidates[0][1]
            fixed_substrates = _clean_substrates_except_scaffold(
                first_reaction.get("substrates", []), variable_scaffold
            )
            product_class = _product_scaffold_class(
                [reaction for _, reaction, _ in variable_candidates]
            )
            question_en = _build_q2_question_en(
                reaction_type,
                conditions,
                fixed_substrates,
                variable_scaffold,
                product_class,
            )

            options = []
            omitted_no_targets = []
            for original_answer_index, reaction, variable_substrate in variable_candidates:
                targets = reaction.get("targets", {})
                ee, yield_val, score, target_count = _score_from_targets(targets)
                er = targets.get("er")
                if target_count == 0 and er in (None, "", [], {}):
                    omitted_no_targets.append(
                        {
                            "reaction_id": reaction.get("id"),
                            "variable_substrate": _compound_public_fields(variable_substrate),
                        }
                    )
                    continue

                options.append(
                    {
                        "option_id": None,
                        "variable_substrate": _compound_public_fields(variable_substrate),
                        "metadata_hidden": {
                            "original_answer_index": original_answer_index,
                            "reaction_id": reaction.get("id"),
                            "ee_raw": targets.get("ee"),
                            "yield_raw": targets.get("yield"),
                            "ee": ee,
                            "yield": yield_val,
                            "er": er,
                            "score": score,
                            "source_paper": reaction.get("source_paper"),
                        },
                    }
                )

            if len(options) < 2:
                review_set.append(
                    {
                        "reason": "insufficient_scored_options",
                        "source_paper": paper,
                        "reaction_type": reaction_type,
                        "conditions": conditions,
                        "condition_signature": condition_signature,
                        "condition_grouping_signature": condition_grouping_signature,
                        "scaffold_combo": list(combo),
                        "variable_scaffold": variable_scaffold,
                        "scored_option_count": len(options),
                        "omitted_no_targets": omitted_no_targets,
                        "reaction_ids": [reaction.get("id") for _, reaction, _ in variable_candidates],
                    }
                )
                continue

            option_groups = defaultdict(list)
            for option in options:
                option_groups[_public_option_key(option["variable_substrate"])].append(option)

            deduped_options = []
            duplicate_conflicts = []
            duplicate_option_count = 0
            for duplicate_key, duplicate_options in sorted(option_groups.items()):
                duplicate_option_count += max(0, len(duplicate_options) - 1)
                scores = {
                    (
                        item["metadata_hidden"].get("ee"),
                        item["metadata_hidden"].get("yield"),
                        item["metadata_hidden"].get("er"),
                        item["metadata_hidden"].get("score"),
                    )
                    for item in duplicate_options
                }
                if len(scores) > 1:
                    duplicate_conflicts.append(
                        {
                            "variable_substrate": duplicate_options[0]["variable_substrate"],
                            "reaction_ids": [
                                item["metadata_hidden"].get("reaction_id")
                                for item in duplicate_options
                            ],
                            "results": [
                                {
                                    "ee": item["metadata_hidden"].get("ee"),
                                    "yield": item["metadata_hidden"].get("yield"),
                                    "er": item["metadata_hidden"].get("er"),
                                    "score": item["metadata_hidden"].get("score"),
                                }
                                for item in duplicate_options
                            ],
                        }
                    )
                    continue
                deduped_options.append(sorted(duplicate_options, key=_option_sort_key)[0])

            if duplicate_conflicts:
                review_set.append(
                    {
                        "reason": "duplicate_variable_substrate_conflicting_results",
                        "action": "omitted_conflicting_variable_substrate",
                        "source_paper": paper,
                        "reaction_type": reaction_type,
                        "conditions": conditions,
                        "condition_signature": condition_signature,
                        "condition_grouping_signature": condition_grouping_signature,
                        "scaffold_combo": list(combo),
                        "variable_scaffold": variable_scaffold,
                        "details": duplicate_conflicts,
                    }
                )

            if len(deduped_options) < 2:
                review_set.append(
                    {
                        "reason": "insufficient_distinct_variable_substrates",
                        "source_paper": paper,
                        "reaction_type": reaction_type,
                        "conditions": conditions,
                        "condition_signature": condition_signature,
                        "condition_grouping_signature": condition_grouping_signature,
                        "scaffold_combo": list(combo),
                        "variable_scaffold": variable_scaffold,
                        "distinct_option_count": len(deduped_options),
                        "duplicate_option_count": duplicate_option_count,
                        "omitted_conflicting_duplicate_options": duplicate_conflicts,
                        "omitted_no_targets": omitted_no_targets,
                    }
                )
                continue

            ranked_options = sorted(deduped_options, key=_option_sort_key)
            max_score = ranked_options[0]["metadata_hidden"]["score"]

            seed = f"{paper}|{combo}|{cond_key}|{variable_scaffold}"
            shuffled_options = _shuffle_options(deduped_options, seed)
            by_original_index = {
                option["metadata_hidden"]["original_answer_index"]: option
                for option in shuffled_options
            }
            gold_option_ids = [
                option["option_id"]
                for option in shuffled_options
                if option["metadata_hidden"]["score"] == max_score
            ]
            gold_ranked_option_ids = [
                by_original_index[option["metadata_hidden"]["original_answer_index"]][
                    "option_id"
                ]
                for option in ranked_options
            ]

            public_options = [
                {
                    "option_id": option["option_id"],
                    "variable_substrate": option["variable_substrate"],
                }
                for option in shuffled_options
            ]

            q2_benchmark.append(
                {
                    "id": f"Q2_{len(q2_benchmark) + 1}",
                    "reaction_type": reaction_type,
                    "reaction_conditions": conditions,
                    "condition_signature": condition_signature,
                    "condition_grouping_signature": condition_grouping_signature,
                    "fixed_substrates": fixed_substrates,
                    "variable_substrate_scaffold": variable_scaffold,
                    "product_scaffold_class": product_class,
                    "question_en": question_en,
                    "source_paper": paper,
                    "option_count": len(public_options),
                    "options": public_options,
                    "gold_option_ids": sorted(gold_option_ids),
                    "gold_ranked_option_ids": gold_ranked_option_ids,
                    "metric_eligibility": {
                        "top1": True,
                        "top3": len(public_options) >= 3,
                    },
                    "metadata_hidden": {
                        "option_results": [
                            {
                                "option_id": option["option_id"],
                                **option["metadata_hidden"],
                            }
                            for option in shuffled_options
                        ],
                        "has_top_score_tie": len(gold_option_ids) > 1,
                        "omitted_no_targets": omitted_no_targets,
                        "omitted_conflicting_duplicate_options": duplicate_conflicts,
                        "deduped_duplicate_option_count": duplicate_option_count,
                    },
                }
            )

    option_count_distribution = {
        str(count): sum(1 for item in q2_benchmark if item["option_count"] == count)
        for count in sorted({item["option_count"] for item in q2_benchmark})
    }

    report = {
        "input_groups": len(paper_combo_data),
        "main_questions": len(q2_benchmark),
        "review_questions": len(review_set),
        "min_options": min((item["option_count"] for item in q2_benchmark), default=0),
        "max_options": max((item["option_count"] for item in q2_benchmark), default=0),
        "option_count_distribution": option_count_distribution,
        "top3_eligible_questions": sum(
            1 for item in q2_benchmark if item["metric_eligibility"]["top3"]
        ),
        "top3_ineligible_questions": sum(
            1 for item in q2_benchmark if not item["metric_eligibility"]["top3"]
        ),
        "top_score_tie_questions": sum(
            1 for item in q2_benchmark if item["metadata_hidden"]["has_top_score_tie"]
        ),
        "review_reasons": dict(
            sorted(
                {
                    reason: sum(1 for item in review_set if item["reason"] == reason)
                    for reason in {item["reason"] for item in review_set}
                }.items()
            )
        ),
    }

    return {
        "benchmark": q2_benchmark,
        "review_set": review_set,
        "report": report,
    }


def generate_q2_benchmark(q2_data: List[dict]) -> List[dict]:
    return generate_q2_benchmark_package(q2_data)["benchmark"]


SUPPORTED_SOURCE_MODALITIES = ("text", "image")


def normalize_source_modality(value) -> str:
    modality = str(value or "").strip().casefold()
    aliases = {
        "text": "text",
        "textual": "text",
        "image": "image",
        "chemeagle": "image",
        "vision": "image",
    }
    return aliases.get(modality, modality or "unknown")


def _attach_modality_to_package(task: str, modality: str, package: dict) -> dict:
    task_key = str(task).upper()
    modality_key = normalize_source_modality(modality)
    for index, question in enumerate(package["benchmark"], start=1):
        question["id"] = f"{task_key}_{modality_key.upper()}_{index:04d}"
        question["source_modality"] = modality_key
        for option_result in question.get("metadata_hidden", {}).get(
            "option_results", []
        ):
            option_result["source_modality"] = modality_key
    for review in package["review_set"]:
        review["source_modality"] = modality_key
    package["report"] = {
        **package["report"],
        "source_modality": modality_key,
    }
    return package


def _aggregate_modality_reports(task: str, packages: Dict[str, dict]) -> dict:
    questions = [
        question
        for modality in SUPPORTED_SOURCE_MODALITIES
        for question in packages[modality]["benchmark"]
    ]
    reviews = [
        review
        for modality in SUPPORTED_SOURCE_MODALITIES
        for review in packages[modality]["review_set"]
    ]
    option_counts = sorted({question["option_count"] for question in questions})
    return {
        "task": str(task).upper(),
        "input_groups": sum(
            packages[modality]["report"].get("input_groups", 0)
            for modality in SUPPORTED_SOURCE_MODALITIES
        ),
        "main_questions": len(questions),
        "review_questions": len(reviews),
        "min_options": min((q["option_count"] for q in questions), default=0),
        "max_options": max((q["option_count"] for q in questions), default=0),
        "option_count_distribution": {
            str(count): sum(1 for question in questions if question["option_count"] == count)
            for count in option_counts
        },
        "top3_eligible_questions": sum(
            1 for question in questions if question["metric_eligibility"]["top3"]
        ),
        "top3_ineligible_questions": sum(
            1 for question in questions if not question["metric_eligibility"]["top3"]
        ),
        "top_score_tie_questions": sum(
            1
            for question in questions
            if question["metadata_hidden"]["has_top_score_tie"]
        ),
        "review_reasons": dict(
            sorted(
                {
                    reason: sum(1 for review in reviews if review.get("reason") == reason)
                    for reason in {review.get("reason") for review in reviews}
                    if reason
                }.items()
            )
        ),
        "by_modality": {
            modality: packages[modality]["report"]
            for modality in SUPPORTED_SOURCE_MODALITIES
        },
    }


def _question_fingerprint(task: str, question: dict) -> str:
    if str(task).upper() == "Q1":
        payload = {
            "source_paper": question.get("source_paper"),
            "reaction_type": question.get("reaction_type"),
            "substrate_combo": question.get("substrate_combo"),
            "product_combo": question.get("product_combo"),
        }
    else:
        payload = {
            "source_paper": question.get("source_paper"),
            "reaction_type": question.get("reaction_type"),
            "condition_signature": question.get("condition_signature"),
            "fixed_substrates": question.get("fixed_substrates"),
            "variable_substrate_scaffold": question.get(
                "variable_substrate_scaffold"
            ),
            "product_scaffold_class": question.get("product_scaffold_class"),
        }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _cross_modal_overlap(task: str, packages: Dict[str, dict]) -> dict:
    fingerprints = {
        modality: {
            _question_fingerprint(task, question)
            for question in packages[modality]["benchmark"]
        }
        for modality in SUPPORTED_SOURCE_MODALITIES
    }
    shared = fingerprints["text"] & fingerprints["image"]
    return {
        "text_only": len(fingerprints["text"] - shared),
        "image_only": len(fingerprints["image"] - shared),
        "both": len(shared),
    }


def generate_benchmark_packages_by_modality(
    q1_data: List[dict],
    q2_data: List[dict],
) -> dict:
    q1_by_modality = {modality: [] for modality in SUPPORTED_SOURCE_MODALITIES}
    q2_by_modality = {modality: [] for modality in SUPPORTED_SOURCE_MODALITIES}
    unexpected_modalities = set()

    for reaction in q1_data:
        modality = normalize_source_modality(reaction.get("source_modality"))
        if modality not in q1_by_modality:
            unexpected_modalities.add(modality)
            continue
        q1_by_modality[modality].append(reaction)
    for reaction in q2_data:
        modality = normalize_source_modality(reaction.get("source_modality"))
        if modality not in q2_by_modality:
            unexpected_modalities.add(modality)
            continue
        q2_by_modality[modality].append(reaction)

    if unexpected_modalities:
        raise ValueError(
            "Unsupported or missing source_modality values: "
            + ", ".join(sorted(unexpected_modalities))
        )

    q1_packages = {
        modality: _attach_modality_to_package(
            "Q1",
            modality,
            generate_q1_benchmark_package(q1_by_modality[modality]),
        )
        for modality in SUPPORTED_SOURCE_MODALITIES
    }
    q2_packages = {
        modality: _attach_modality_to_package(
            "Q2",
            modality,
            generate_q2_benchmark_package(q2_by_modality[modality]),
        )
        for modality in SUPPORTED_SOURCE_MODALITIES
    }

    combined = {}
    for task, packages in (("Q1", q1_packages), ("Q2", q2_packages)):
        combined[task.lower()] = {
            "benchmark": [
                question
                for modality in SUPPORTED_SOURCE_MODALITIES
                for question in packages[modality]["benchmark"]
            ],
            "review_set": [
                review
                for modality in SUPPORTED_SOURCE_MODALITIES
                for review in packages[modality]["review_set"]
            ],
            "report": _aggregate_modality_reports(task, packages),
        }

    question_option_counts = []
    for task in ("q1", "q2"):
        for question in combined[task]["benchmark"]:
            question_option_counts.append(
                {
                    "question_id": question["id"],
                    "task": task.upper(),
                    "source_modality": question["source_modality"],
                    "source_paper": question.get("source_paper"),
                    "reaction_type": question.get("reaction_type"),
                    "option_count": question["option_count"],
                    "option_ids": [
                        option["option_id"] for option in question.get("options", [])
                    ],
                    "top3_eligible": question["metric_eligibility"]["top3"],
                    "has_top_score_tie": question["metadata_hidden"][
                        "has_top_score_tie"
                    ],
                }
            )

    return {
        "by_modality": {
            modality: {
                "q1_reactions": q1_by_modality[modality],
                "q2_reactions": q2_by_modality[modality],
                "q1": q1_packages[modality],
                "q2": q2_packages[modality],
            }
            for modality in SUPPORTED_SOURCE_MODALITIES
        },
        "q1": combined["q1"],
        "q2": combined["q2"],
        "question_option_counts": question_option_counts,
        "cross_modal_overlap": {
            "q1": _cross_modal_overlap("Q1", q1_packages),
            "q2": _cross_modal_overlap("Q2", q2_packages),
        },
    }
