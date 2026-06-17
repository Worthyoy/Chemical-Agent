"""
GP (General Procedure) 提取工具 - 独立调试模块
从 SI PDF 中提取、总结和过滤 General Procedure 相关内容

功能:
1. GP 文本提取 - 从全文提取 General Procedure 段落
2. GP 总结 - 用 LLM 总结 GP 文本为结构化信息
3. GP 过滤 - 检测 chunk 中引用了哪些 GP
4. GP 条件注入 - 为 Stage2 提取提供 GP 上下文
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# 确保能导入同目录下的 pdf_to_gpt_extractor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_to_gpt_extractor import PDFReactionExtractor


class GPExtractor(PDFReactionExtractor):
    """GP (General Procedure) 提取器，用于从化学文献中提取通用程序信息"""

    # =====================================================================
    # GP 相关常量
    # =====================================================================

    # GP 标题正则 — 用于定位 General Procedure 段落
    EXPLICIT_GP_TITLE_PATTERN = (
        r'(?im)^\s*(?:\d+[\).]\s*)?'
        r'(?:general\s+procedure|representative\s+procedure|typical\s+procedure|'
        r'standard\s+procedure|standard\s+conditions|experimental\s+procedure|procedure)'
        r'\s+[A-Z0-9]+\b\s*(?::|\uff1a)'
    )

    GP_TITLE_PATTERNS = [
        EXPLICIT_GP_TITLE_PATTERN,
        r'(?i)(general\s+procedure)(?:\s+for\s+synthesis\s+of\s+([\w\d\s,\-–and]+))?\s*([A-Z])?\b',
        r'(?im)^\s*(?:\d+[\).]\s*)?(representative\s+procedure)(?:\s+for\b[^\n]{0,120}|\s+[A-Z]\b\s*(?::|\uff1a|\.|-|\u2013|\u2014|$)|\b)',
        r'(?im)^\s*(?:\d+[\).]\s*)?(typical\s+procedure)(?:\s+for\b[^\n]{0,120}|\s+[A-Z]\b\s*(?::|\uff1a|\.|-|\u2013|\u2014|$)|\b)',
        r'(?im)^\s*(?:\d+[\).]\s*)?(standard\s+procedure)(?:\s+for\b[^\n]{0,120}|\s+[A-Z]\b\s*(?::|\uff1a|\.|-|\u2013|\u2014|$)|\b)',
        r'(?im)^\s*(?:\d+[\).]\s*)?(standard\s+conditions)(?:\s+[A-Z]\b\s*(?::|\uff1a|\.|-|\u2013|\u2014|$)|\b)',
        r'(?im)^\s*(?:\d+[\).]\s*)?(experimental\s+procedure)\s+for\b[^\n]{0,120}',
        r'(?im)^\s*(?:\d+[\).]\s*)?procedure\s+[A-Z0-9]+\b\s*(?::|\uff1a|\.|-|\u2013|\u2014|$)',
    ]

    # GP 定义 vs 引用的区分关键词
    GP_DEFINITION_KEYWORDS = [
        'charged with', 'was added', 'flask', 'dissolved', 'stirred',
        'irradiated', 'heated', 'cooled', 'degassed', 'sealed',
        'mixture', 'solution', 'suspension', 'reaction mixture',
        'inert atmosphere', 'under n2', 'round-bottom',
    ]

    GP_REFERENCE_KEYWORDS = [
        'according to', 'prepared according', 'using ',
        'following the', 'following procedure', 'following procedures',
        'as described', 'following general',
    ]

    # General Procedure 注入到 Stage2 的 prompt 模板
    GP_INJECTION_TEMPLATE = """
GENERAL PROCEDURE CONTEXT — apply these conditions when the entry specifies none:

{gp_block}

RULES:
1. Entry-specified reagents/solvents/conditions ALWAYS override GP conditions.
2. Do NOT mix reaction types (e.g., NaBH4 = reduction, not photocatalysis).
3. Check KNOWN SYMBOL-TO-NAME MAPPINGS to infer substrate chemical class.
4. Do NOT set substrates = products.
"""

    # GP 总结 prompt
    GP_SUMMARY_PROMPT = """You are a chemistry data extraction specialist. 
Extract structured information from this General Procedure text.
GP Label: {gp_label}
Return ONLY a JSON object with these fields:
- substrates: 底物(可能是多个，符号或名称)
- catalysts: 催化剂/光催化剂/配体(可能是多个)  
- solvents: 溶剂(可能是多个)
- additives: 添加剂(可能是多个)
- reagents: 其他试剂
- conditions: 温度/时间/光源/波长/气氛/其他
- reaction_type_derived: 根据文本内容推断的反应类型
- summary: 有可能其会给出适用的产物名称是什么，要根据此进行匹配
JSON format:
{{
  "gp_label": "{gp_label}",
  "substrates": ["list of substrates"],
  "catalysts": ["list of catalysts"],
  "solvents": ["list of solvents"],
  "conditions": {{
    "temperature": "value or null",
    "time": "value or null", 
    "light_source": "value or null",
    "wavelength": "value or null",
    "atmosphere": "value or null",
    "other_conditions": "value or null"
  }},
  "summary": "brief description"
}}
Process this GP text:
{gp_text}
Return ONLY valid JSON."""

    def __init__(self, api_key: Optional[str] = None,
                 screen_model: str = "gpt-5-mini",
                 regex_only: bool = True):
        """
        初始化 GP 提取器

        Args:
            api_key: OpenAI API 密钥（regex_only=False 时必需）
            screen_model: 用于 GP 总结的模型（默认 gpt-5-mini）
            regex_only: 是否只使用正则提取（默认 True，不需要 API 密钥）
        """
        self.screen_model = screen_model
        self.regex_only = regex_only
        self.client = None

        if not regex_only:
            # 需要 GPT 时才初始化父类
            super().__init__(api_key)
            self.client = self.client  # 从父类获取 client

    # =====================================================================
    # GP 文本提取方法
    # =====================================================================

    def _make_gp_key(self, title: str, gp_counter: Dict[str, int]) -> str:
        """
        根据 GP 标题生成 key
        例: 'General procedure for synthesis of 1-38:' → 'GeneralProcedure_1-38'
            'General Procedure A' → 'GeneralProcedureA'
            无标识时 → 'GeneralProcedureA', 'GeneralProcedureB', ...
        """
        # 尝试提取 scope (数字范围)
        if self._is_explicit_gp_title(title):
            return self._normalize_gp_title_key(title)

        scope_match = re.search(
            r'(\d+\w*)\s*[-–]\s*(\d+\w*)(?:\s+and\s+(\d+\w*))?',
            title
        )
        if scope_match:
            scope = f"{scope_match.group(1)}-{scope_match.group(2)}"
            if scope_match.group(3):
                scope += f"-{scope_match.group(3)}"
            scope = scope.replace(' ', '')
            return f"GeneralProcedure_{scope}"

        # 尝试提取字母标识 (如 "Procedure A", "Procedure B")
        letter_match = re.search(
            r'procedure\s+([A-Z0-9]+)\b\s*(?::|\uff1a|\.|-|\u2013|\u2014|$)',
            title,
            re.IGNORECASE,
        )
        if letter_match:
            return f"GeneralProcedure{letter_match.group(1).upper()}"

        # 无标识，按顺序编号
        if 'GeneralProcedure' not in gp_counter:
            gp_counter['GeneralProcedure'] = 0
        gp_counter['GeneralProcedure'] += 1
        idx = gp_counter['GeneralProcedure']
        suffix = chr(ord('A') + idx - 1) if idx <= 26 else str(idx)
        return f"GeneralProcedure{suffix}"

    def _is_explicit_gp_title(self, title: str) -> bool:
        """Return True for line-start GP headings with an explicit label and colon."""
        return bool(re.match(self.EXPLICIT_GP_TITLE_PATTERN, title.strip()))

    def _normalize_gp_title_key(self, title: str) -> str:
        """Normalize explicit GP headings to human-readable keys."""
        match = re.match(
            r'(?i)^\s*(?:\d+[\).]\s*)?'
            r'(general\s+procedure|representative\s+procedure|typical\s+procedure|'
            r'standard\s+procedure|standard\s+conditions|experimental\s+procedure|procedure)'
            r'\s+([A-Z0-9]+)\b\s*(?::|\uff1a)',
            title.strip(),
        )
        if not match:
            return title.strip().rstrip(':：').strip()

        canonical_prefixes = {
            'general procedure': 'General Procedure',
            'representative procedure': 'Representative Procedure',
            'typical procedure': 'Typical Procedure',
            'standard procedure': 'Standard Procedure',
            'standard conditions': 'Standard Conditions',
            'experimental procedure': 'Experimental Procedure',
            'procedure': 'Procedure',
        }
        prefix = canonical_prefixes[re.sub(r'\s+', ' ', match.group(1).lower())]
        label = match.group(2).upper()
        return f"{prefix} {label}"

    def _gp_key_aliases(self, gp_key: str) -> List[str]:
        """Build old and new labels for matching GP references in text."""
        aliases = {gp_key}
        label_match = re.search(
            r'(?i)\b(?:general\s+procedure|representative\s+procedure|typical\s+procedure|'
            r'standard\s+procedure|standard\s+conditions|experimental\s+procedure|procedure)\s+([A-Z0-9]+)\b',
            gp_key,
        )
        if not label_match:
            label_match = re.search(r'GeneralProcedure([A-Z0-9]+)$', gp_key, re.IGNORECASE)

        if label_match:
            label = label_match.group(1).upper()
            aliases.update({
                f"GP {label}",
                f"Procedure {label}",
                f"Procedures {label}",
                f"General Procedure {label}",
                f"General Procedures {label}",
                f"Typical Procedures {label}",
                f"Representative Procedures {label}",
                f"Standard Procedures {label}",
                f"Experimental Procedures {label}",
                f"GeneralProcedure{label}",
            })

        return sorted(aliases, key=len, reverse=True)

    def _text_contains_gp_alias(self, text: str, gp_key: str) -> bool:
        for alias in self._gp_key_aliases(gp_key):
            pattern = r'(?i)\b' + r'\s+'.join(re.escape(part) for part in alias.split()) + r'\b'
            if re.search(pattern, text):
                return True
        return False

    def _has_reference_cue_before_title(self, full_text: str, pos: int) -> bool:
        """Return True when a GP title is immediately preceded by a reference cue."""
        pre_context = full_text[max(0, pos - 160):pos]
        pre_context = re.sub(r'---\s*Page\s+\d+\s*---', ' ', pre_context, flags=re.IGNORECASE)
        pre_context = re.sub(r'\s+', ' ', pre_context).strip()
        return bool(re.search(
            r'(?i)(?:according(?:\s+to|\s+the)?|following(?:\s+the|\s+procedures?)?)\s*$',
            pre_context,
        ))

    def _is_gp_definition(self, text_after: str) -> bool:
        """
        判断一个 GP 标题匹配是 "GP 定义" 还是 "GP 引用"
        GP 定义：后面跟操作步骤（"A flask was charged with..."）
        GP 引用：后面跟具体用量（"using 95.7 mg..."、"according to..."）
        """
        text_lower = text_after[:300].lower()

        # 引用型：开头就是 according to / using / prepared
        for kw in self.GP_REFERENCE_KEYWORDS:
            if kw in text_lower[:100]:
                return False

        # 定义型：包含操作关键词
        for kw in self.GP_DEFINITION_KEYWORDS:
            if kw in text_lower:
                return True

        return False

    def _build_gp_full_text(self, pages: List[Dict]) -> str:
        full_text = "\n".join(
            f"--- Page {p['page_num']} ---\n{p['text']}"
            for p in pages
        )
        full_text = re.sub(r'(?<=[A-Za-z])-\n\s*(?=[a-z])', '', full_text)
        full_text = re.sub(r'\n\s+(?=[a-z(])', ' ', full_text)
        return full_text

    def _find_filtered_gp_titles(self, full_text: str) -> List[Tuple[int, str]]:
        all_matches = []
        for pattern in self.GP_TITLE_PATTERNS:
            for m in re.finditer(pattern, full_text):
                all_matches.append((m.start(), m.group(0).strip()))

        all_matches.sort(key=lambda x: x[0])
        deduped_matches = []
        for pos, title in all_matches:
            if deduped_matches and deduped_matches[-1][0] == pos:
                if len(title) > len(deduped_matches[-1][1]):
                    deduped_matches[-1] = (pos, title)
            else:
                deduped_matches.append((pos, title))
        all_matches = deduped_matches

        filtered = []
        for match_index, (pos, title) in enumerate(all_matches):
            if self._has_reference_cue_before_title(full_text, pos):
                continue
            if (
                not self._is_explicit_gp_title(title)
                and match_index + 1 < len(all_matches)
                and all_matches[match_index + 1][0] - pos <= 200
                and self._is_explicit_gp_title(all_matches[match_index + 1][1])
            ):
                continue
            context_after = full_text[pos:pos + 300]
            if self._is_explicit_gp_title(title) or self._is_gp_definition(context_after):
                filtered.append((pos, title))
        return filtered

    def extract_general_procedure_records(self, pages: List[Dict]) -> List[Dict[str, Any]]:
        """Regex-only GP extraction with boundary metadata for debugging."""
        full_text = self._build_gp_full_text(pages)
        filtered = self._find_filtered_gp_titles(full_text)
        if not filtered:
            self.last_gp_records = []
            return []

        gp_counter = {}
        max_gp_chars = 2000
        records: List[Dict[str, Any]] = []
        for i, (pos, title) in enumerate(filtered):
            has_next_gp = i + 1 < len(filtered)
            end_pos = filtered[i + 1][0] if has_next_gp else len(full_text)
            next_title = filtered[i + 1][1] if has_next_gp else None
            raw_text = full_text[pos:end_pos].strip()
            raw_chars = len(raw_text)
            key = self._make_gp_key(title, gp_counter)

            if has_next_gp and raw_chars <= max_gp_chars:
                end_reason = "next_gp_title_complete"
                needs_llm = False
            elif has_next_gp:
                end_reason = "max_chars_before_next"
                needs_llm = True
            elif raw_chars <= max_gp_chars:
                end_reason = "last_gp_eof_short"
                needs_llm = True
            else:
                end_reason = "last_gp_max_chars"
                needs_llm = True

            final_text = raw_text[:max_gp_chars]
            records.append({
                "key": key,
                "title": title,
                "raw_text": raw_text,
                "final_text": final_text,
                "start_pos": pos,
                "raw_end_pos": end_pos,
                "has_next_gp": has_next_gp,
                "next_gp_title": next_title,
                "raw_chars_to_next_or_eof": raw_chars,
                "stored_chars": len(final_text),
                "max_gp_chars": max_gp_chars,
                "end_reason": end_reason,
                "pre_llm_end_reason": end_reason,
                "needs_llm_truncation": needs_llm,
            })

        self.last_gp_records = records
        return records

    def extract_general_procedure_texts(self, pages: List[Dict]) -> Dict[str, str]:
        records = self.extract_general_procedure_records(pages)
        if not records:
            return {}

        gp_texts: Dict[str, str] = {}
        for record in records:
            key = record["key"]
            text = record.get("final_text") or record.get("raw_text", "")
            if key in gp_texts:
                gp_texts[key] += "\n\n" + text
            else:
                gp_texts[key] = text
        return gp_texts

    def extract_gp_from_text(self, text: str) -> Dict[str, str]:
        """
        直接从文本提取 GP（不需要按页格式）
        便于调试时直接传入文本

        Args:
            text: 完整的文本内容

        Returns:
            {"GeneralProcedureA": "GP原文...", ...}
        """
        # 模拟页面格式
        pages = [{"page_num": 1, "text": text}]
        return self.extract_general_procedure_texts(pages)

    # =====================================================================
    # GP 总结方法 (LLM)
    # =====================================================================

    def summarize_gp_texts(self, gp_texts: Dict[str, str]) -> Dict[str, Dict]:
        """
        用 LLM 总结 GP 文本，提取结构化信息

        Args:
            gp_texts: {"GeneralProcedureA": "GP原文...", ...}

        Returns:
            {"GeneralProcedureA": {"substrates": [...], "catalysts": [...], ...}, ...}
        """
        if not gp_texts:
            return {}

        if self.client is None:
            raise ValueError("需要 OpenAI API 密钥才能使用 GPT 总结功能。请使用 --api_key 参数或设置 OPENAI_API_KEY 环境变量。")

        summarized = {}

        for gp_label, gp_text in gp_texts.items():
            print(f"    总结GP: {gp_label}")

            # 截断过长的 GP 文本
            text_to_process = gp_text[:2000] if len(gp_text) > 2000 else gp_text

            try:
                response = self.client.chat.completions.create(
                    model=self.screen_model,
                    messages=[
                        {"role": "system", "content": "You are a chemistry data extractor. Always respond with valid JSON only."},
                        {"role": "user", "content": self.GP_SUMMARY_PROMPT.format(
                            gp_label=gp_label,
                            gp_text=text_to_process
                        )}
                    ],
                    temperature=0.0,
                )
                raw = response.choices[0].message.content.strip()
                parsed = self._parse_gp_summary(raw)
                if parsed:
                    summarized[gp_label] = parsed
                    print(f"      成功提取: substrates={len(parsed.get('substrates', []))}, catalysts={len(parsed.get('catalysts', []))}")
                else:
                    print(f"      解析失败")
                    summarized[gp_label] = {"error": "parse_failed"}
            except Exception as e:
                print(f"      GPT总结出错: {e}")
                summarized[gp_label] = {"error": str(e)}

        return summarized

    def _parse_gp_summary(self, raw: str) -> Optional[Dict]:
        """解析 GPT 返回的 GP 总结 JSON"""
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
            if isinstance(data, dict):
                # 确保必要字段存在
                return {
                    "gp_label": data.get("gp_label", ""),
                    "substrates": data.get("substrates", []),
                    "catalysts": data.get("catalysts", []),
                    "solvents": data.get("solvents", []),
                    "additives": data.get("additives", []),
                    "reagents": data.get("reagents", []),
                    "conditions": data.get("conditions", {}),
                    "reaction_type_derived": data.get("reaction_type_derived", ""),
                    "summary": data.get("summary", "")
                }
        except json.JSONDecodeError:
            pass
        return None

    # =====================================================================
    # GP 过滤方法
    # =====================================================================

    def _filter_gp_for_chunk(self, chunk_text: str,
                             gp_summaries: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        检测 chunk_text 中引用了哪些 GP，只返回被引用 GP 的摘要。
        降级: 单 GP 直接返回; 多 GP 无引用则返回空 dict。

        Args:
            chunk_text: 分块文本
            gp_summaries: GP 总结结果

        Returns:
            过滤后的 GP 摘要
        """
        if not gp_summaries:
            return gp_summaries

        if len(gp_summaries) == 1:
            return gp_summaries

        text_lower = chunk_text.lower()
        matched = {}

        for gp_label, summary in gp_summaries.items():
            if not isinstance(summary, dict) or "error" in summary:
                continue

            # 模式1: 字母标识 (GeneralProcedureA → "GP A", "General Procedure A")
            if self._text_contains_gp_alias(chunk_text, gp_label):
                matched[gp_label] = summary
                continue

            letter_match = re.search(r'GeneralProcedure([A-Z])$', gp_label)
            if letter_match:
                letter = letter_match.group(1)
                if re.search(rf'(?i)\bgp\s*{re.escape(letter)}\b', chunk_text):
                    matched[gp_label] = summary
                    continue
                if re.search(rf'(?i)general\s+procedure\s+{re.escape(letter)}\b', chunk_text):
                    matched[gp_label] = summary
                    continue

            # 模式2: 数字范围 (GeneralProcedure_1-38 → "for 1-38", "GP for 1")
            scope_match = re.search(r'GeneralProcedure_([\d\w]+)-([\d\w]+)', gp_label)
            if scope_match:
                lo = scope_match.group(1)
                if re.search(rf'(?i)general\s+procedure\s+for\s+{re.escape(lo)}', chunk_text):
                    matched[gp_label] = summary
                    continue
                if re.search(rf'(?i)for\s+{re.escape(lo)}\s*[-\u2013]', chunk_text):
                    matched[gp_label] = summary
                    continue

            # 模式3: 催化剂名称匹配（兜底，只检查 >=5 字符的名称）
            catalysts = summary.get("catalysts", [])
            for cat in catalysts:
                if isinstance(cat, str) and len(cat) >= 5 and cat.lower() in text_lower:
                    matched[gp_label] = summary
                    break

        return matched  # 可能为空 {}

    # =====================================================================
    # GP 条件注入方法
    # =====================================================================

    def build_gp_injection_block(self, gp_summaries: Dict[str, Dict]) -> str:
        """
        构建 GP 注入块，用于注入到 Stage2 提取 prompt 中

        Args:
            gp_summaries: GP 总结结果

        Returns:
            格式化的 GP 注入文本
        """
        if not gp_summaries:
            return ""

        gp_entries = []
        for label, summary in gp_summaries.items():
            if isinstance(summary, dict) and "error" not in summary:
                # 构建结构化的 GP 摘要注入
                substrates = summary.get("substrates", [])
                catalysts = summary.get("catalysts", [])
                solvents = summary.get("solvents", [])
                conditions = summary.get("conditions", {})
                summary_text = summary.get("summary", "")

                cond_parts = []
                if conditions.get("temperature"):
                    cond_parts.append(f"T: {conditions.get('temperature')}")
                if conditions.get("time"):
                    cond_parts.append(f"Time: {conditions.get('time')}")
                if conditions.get("light_source"):
                    cond_parts.append(f"Light: {conditions.get('light_source')}")
                if conditions.get("wavelength"):
                    cond_parts.append(f"λ: {conditions.get('wavelength')}")
                if conditions.get("atmosphere"):
                    cond_parts.append(f"Atmosphere: {conditions.get('atmosphere')}")

                cond_str = ", ".join(cond_parts) if cond_parts else "N/A"

                entry = f"""=== {label} ===
Substrates: {', '.join(substrates) if substrates else 'N/A'}
Catalysts: {', '.join(catalysts) if catalysts else 'N/A'}
Solvents: {', '.join(solvents) if solvents else 'N/A'}
Conditions: {cond_str}
Summary: {summary_text}"""
                gp_entries.append(entry)

        if gp_entries:
            return self.GP_INJECTION_TEMPLATE.format(
                gp_block="\n\n".join(gp_entries)
            )
        return ""

    # =====================================================================
    # GP 条件来源标记（兜底）
    # =====================================================================

    def apply_gp_conditions_fallback(self, reactions: List[Dict],
                                      gp_texts: Dict[str, str]) -> List[Dict]:
        """
        兜底标记：对 conditions 全 null 的反应，根据 id 前缀匹配 GP 来源

        匹配策略（按优先级）：
        1. 只有一个 GP → 直接关联
        2. 按 id 前缀匹配（如 GeneralProcedure_1-38-Entry5 → GeneralProcedure_1-38）
        3. 按产物 symbol 数字范围匹配 GP scope
        """
        if not gp_texts:
            return reactions

        # 预处理：从 GP key 中提取 scope 范围
        gp_scopes = {}
        for gp_key in gp_texts:
            scope_match = re.search(
                r'(\d+\w*)\s*[-–]\s*(\d+\w*)',
                gp_key
            )
            if scope_match:
                gp_scopes[gp_key] = (scope_match.group(1), scope_match.group(2))

        filled_count = 0
        for reaction in reactions:
            if not isinstance(reaction, dict):
                continue

            # 已有 _gp_source，跳过
            if reaction.get('_gp_source'):
                continue

            # 检查 conditions 是否全 null
            conditions = reaction.get('conditions', {})
            if not isinstance(conditions, dict):
                continue
            if any(v is not None for v in conditions.values()):
                continue

            # 策略1: 只有一个 GP → 直接关联
            if len(gp_texts) == 1:
                reaction['_gp_source'] = list(gp_texts.keys())[0]
                filled_count += 1
                continue

            # 策略2: 按 id 前缀匹配
            rid = str(reaction.get('id', ''))
            matched = False
            for gp_key in gp_texts:
                if gp_key in rid or self._text_contains_gp_alias(rid, gp_key):
                    reaction['_gp_source'] = gp_key
                    filled_count += 1
                    matched = True
                    break
            if matched:
                continue

            # 策略3: 按产物 symbol 数字范围匹配
            products = reaction.get('products', [])
            if isinstance(products, list) and gp_scopes:
                for prod in products:
                    if not isinstance(prod, dict):
                        continue
                    symbol = str(prod.get('symbol', ''))
                    num_match = re.match(r'[A-Za-z]?(\d+)', symbol)
                    if num_match:
                        prod_num = int(num_match.group(1))
                        for gp_key, (lo, hi) in gp_scopes.items():
                            try:
                                lo_num = int(re.match(r'\d+', lo).group())
                                hi_num = int(re.match(r'\d+', hi).group())
                                if lo_num <= prod_num <= hi_num:
                                    reaction['_gp_source'] = gp_key
                                    filled_count += 1
                                    matched = True
                                    break
                            except (ValueError, AttributeError):
                                continue
                    if matched:
                        break

        if filled_count > 0:
            print(f"  [GP Fallback] 标记了 {filled_count} 条反应的 GP 来源")

        return reactions

    # =====================================================================
    # 辅助方法：按页提取文本
    # =====================================================================

    def extract_text_by_pages(self, pdf_path: str) -> List[Dict]:
        """
        按页提取 PDF 文本，返回页列表

        Returns:
            [{"page_num": 1, "text": "..."}, ...]
        """
        from pdf_to_gpt_extractor import PDF_LIBRARY

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        pages = []

        if PDF_LIBRARY == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append({"page_num": page_num, "text": page_text})

        elif PDF_LIBRARY == "PyPDF2":
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append({"page_num": page_num, "text": page_text})
        else:
            raise ImportError("未安装PDF处理库。请运行: pip install pdfplumber")

        return pages

    # =====================================================================
    # 调试用：完整 GP 提取流程
    # =====================================================================

    def debug_extract_gp(self, pdf_path: str, output_dir: Optional[str] = None,
                         do_summarize: bool = False) -> Dict:
        """
        调试模式：从单个 PDF 提取所有 GP 信息

        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录（可选，用于保存调试结果）
            do_summarize: 是否使用 GPT 总结 GP（需要 API 密钥）

        Returns:
            {
                "gp_texts": {...},
                "gp_summaries": {...},
                "stats": {...}
            }
        """
        pdf_name = Path(pdf_path).name
        print(f"\n{'='*60}")
        print(f"GP 调试模式: {pdf_name}")
        print(f"{'='*60}")

        # Step 1: 按页提取文本
        print("[Step 1] 按页提取文本...")
        pages = self.extract_text_by_pages(pdf_path)
        print(f"  提取到 {len(pages)} 页文本")

        if not pages:
            print("  [SKIP] PDF 无文本内容")
            return {"gp_texts": {}, "gp_summaries": {}, "stats": {"total_pages": 0}}

        # Step 2: 提取 GP 文本（正则）
        print("[Step 2] 提取 General Procedure 文本...")
        gp_texts = self.extract_general_procedure_texts(pages)
        print(f"  找到 {len(gp_texts)} 个 GP 段落")

        for key, text in gp_texts.items():
            print(f"\n  --- {key} ---")
            print(f"  {text[:200]}...")

        # Step 3: 总结 GP（GPT，可选）
        gp_summaries = {}
        if do_summarize and gp_texts:
            print("\n[Step 3] 总结 GP 文本 (GPT)...")
            gp_summaries = self.summarize_gp_texts(gp_texts)

            for key, summary in gp_summaries.items():
                print(f"\n  --- {key} Summary ---")
                if "error" in summary:
                    print(f"  Error: {summary['error']}")
                else:
                    print(f"  substrates: {summary.get('substrates', [])}")
                    print(f"  catalysts: {summary.get('catalysts', [])}")
                    print(f"  solvents: {summary.get('solvents', [])}")
                    print(f"  summary: {summary.get('summary', '')[:100]}")
        elif gp_texts:
            print("\n[Step 3] 跳过 GPT 总结（使用 --summarize 启用）")

        # 保存调试结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            debug_file = output_path / f"{Path(pdf_path).stem}_gp_debug.json"
            debug_data = {
                "source": str(pdf_path),
                "total_pages": len(pages),
                "gp_texts": gp_texts,
                "gp_summaries": gp_summaries,
            }
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            print(f"\n调试结果已保存到: {debug_file}")

        return {
            "gp_texts": gp_texts,
            "gp_summaries": gp_summaries,
            "stats": {
                "total_pages": len(pages),
                "gp_count": len(gp_texts),
            }
        }

    # =====================================================================
    # 批量处理方法
    # =====================================================================

    def batch_process(self, input_dir: str, output_dir: str,
                      do_summarize: bool = False) -> Dict:
        """
        批量处理文件夹中的所有 PDF 文件

        Args:
            input_dir: 输入文件夹路径
            output_dir: 输出目录路径
            do_summarize: 是否使用 GPT 总结

        Returns:
            处理统计结果
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 收集所有 PDF 文件
        pdf_files = sorted(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"未找到 PDF 文件: {input_dir}")
            return {"total": 0, "processed": 0}

        print(f"找到 {len(pdf_files)} 个 PDF 文件")
        print(f"输出目录: {output_path}")

        results = {
            "total": len(pdf_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_gp": 0,
            "per_file": {},
        }
        all_gp_texts = {}

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n{'#'*60}")
            print(f"# 进度: {i}/{len(pdf_files)}")
            print(f"{'#'*60}")

            try:
                result = self.debug_extract_gp(
                    str(pdf_file),
                    str(output_path),
                    do_summarize=do_summarize
                )

                gp_count = result['stats']['gp_count']
                if gp_count > 0:
                    results["processed"] += 1
                    results["total_gp"] += gp_count
                    results["per_file"][pdf_file.name] = {
                        "status": "success",
                        "gp_count": gp_count,
                        "gp_keys": list(result['gp_texts'].keys()),
                    }
                    # 合并到总字典
                    for key, text in result['gp_texts'].items():
                        full_key = f"{pdf_file.stem}__{key}"
                        all_gp_texts[full_key] = text
                else:
                    results["skipped"] += 1
                    results["per_file"][pdf_file.name] = {
                        "status": "skipped",
                        "gp_count": 0,
                        "reason": "no_gp_found",
                    }
            except Exception as e:
                print(f"  [ERROR] 处理失败: {e}")
                results["failed"] += 1
                results["per_file"][pdf_file.name] = {
                    "status": "failed",
                    "error": str(e),
                }

        # 保存汇总
        summary_file = output_path / "all_gp_summary.json"
        summary = {
            "input_dir": str(input_dir),
            "total_pdfs": len(pdf_files),
            "processed": results["processed"],
            "skipped": results["skipped"],
            "failed": results["failed"],
            "total_gp": results["total_gp"],
            "per_file": results["per_file"],
            "all_gp_texts": all_gp_texts,
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 打印最终统计
        print(f"\n{'='*60}")
        print(f"批量处理完成!")
        print(f"{'='*60}")
        print(f"  总 PDF 数:    {len(pdf_files)}")
        print(f"  成功处理:     {results['processed']}")
        print(f"  跳过(无GP):   {results['skipped']}")
        print(f"  失败:         {results['failed']}")
        print(f"  总 GP 数:     {results['total_gp']}")
        print(f"  汇总文件:     {summary_file}")

        return results


# =====================================================================
# 命令行入口（调试用）
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GP 提取工具 - 从 SI PDF 中提取 General Procedure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只使用正则提取单个 PDF（不需要 API 密钥）
  python gp_extractor.py --pdf /path/to/si.pdf

  # 批量处理文件夹中所有 PDF（只用正则）
  python gp_extractor.py --input ../supporting_information

  # 批量处理 + GPT 总结（需要 API 密钥）
  python gp_extractor.py --input ../supporting_information --summarize

  # 指定输出目录
  python gp_extractor.py --input ../supporting_information --output ./gp_output

工作流程:
  1. 先用正则测试: python gp_extractor.py --input ../supporting_information
  2. 确认正则结果正确后，再用 GPT 总结: python gp_extractor.py --input ../supporting_information --summarize
        """
    )
    parser.add_argument("--pdf", default=None,
                        help="单个 PDF 文件路径")
    parser.add_argument("--input", default=None,
                        help="输入文件夹路径（处理文件夹中所有 PDF）")
    parser.add_argument("--output", default=None,
                        help="输出目录 (默认: ./gp_debug_output)")
    parser.add_argument("--summarize", action="store_true",
                        help="使用 GPT 总结 GP（需要 API 密钥）")
    parser.add_argument("--api_key", default=None,
                        help="OpenAI API 密钥 (默认从 OPENAI_API_KEY 环境变量读取)")
    parser.add_argument("--screen_model", default="gpt-5-mini",
                        help="GP 总结模型 (默认: gpt-5-mini)")

    args = parser.parse_args()

    # 检查输入参数
    if not args.pdf and not args.input:
        print("错误: 请指定 --pdf 或 --input 参数")
        print("  使用 --pdf 处理单个文件")
        print("  使用 --input 处理文件夹中所有 PDF")
        print("  使用 --help 查看帮助")
        return

    if args.pdf and args.input:
        print("错误: --pdf 和 --input 不能同时使用")
        return

    # 确定是否只使用正则
    regex_only = not args.summarize

    # 如果需要 GPT，检查 API 密钥
    api_key = None
    if not regex_only:
        api_key = args.api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("错误: --summarize 需要 OpenAI API 密钥")
            print("  方法1: 设置环境变量 $env:OPENAI_API_KEY='your-api-key'")
            print("  方法2: 使用参数 --api_key YOUR_KEY")
            print("  或者不使用 --summarize，只用正则提取")
            return

    # 确定输出目录
    output_dir = args.output or str(Path(__file__).resolve().parent / "gp_debug_output")

    # 创建 GP 提取器
    extractor = GPExtractor(
        api_key=api_key,
        screen_model=args.screen_model,
        regex_only=regex_only,
    )

    # 执行提取
    if args.pdf:
        # 单个文件模式
        print(f"单文件模式: {args.pdf}")
        result = extractor.debug_extract_gp(args.pdf, output_dir, do_summarize=args.summarize)

        # 打印最终结果
        print(f"\n{'='*60}")
        print(f"GP 提取完成!")
        print(f"{'='*60}")
        print(f"  总页数: {result['stats']['total_pages']}")
        print(f"  GP 数量: {result['stats']['gp_count']}")
        print(f"  输出目录: {output_dir}")
        if args.summarize:
            print(f"  GPT 总结: 已完成")
        else:
            print(f"  GPT 总结: 未执行（使用 --summarize 启用）")
    else:
        # 批量处理模式
        print(f"批量模式: {args.input}")
        extractor.batch_process(args.input, output_dir, do_summarize=args.summarize)


if __name__ == "__main__":
    main()
