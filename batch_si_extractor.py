"""
SI PDF批量提取工具 - Token节省策略全部组合
从supporting_information文件夹批量读取PDF，提取反应数据

策略:
1. Name Registry (Stage 0) - 预提取符号→完整化学名称映射表
2. 关键词过滤 - 筛选化学相关页面，去掉参考文献/致谢等无关页
3. 页面分块 - 按N页分块，每块独立调GPT
4. 两阶段提取 - gpt-5-mini先筛，gpt-5-mini再精提（注入registry上下文）
5. 后处理对齐 (Stage 5) - 用registry补全提取结果中的符号
6. 结合去重 - 合并所有块的结果并去重
"""

import json
import math
import os
import re
import sys
import time
import contextvars
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

# 确保能导入同目录下的pdf_to_gpt_extractor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_to_gpt_extractor import PDFReactionExtractor, PDF_LIBRARY

GP_CONTEXT_CHAR_LIMIT = 1500


class SIExtractor(PDFReactionExtractor):
    """SI PDF批量提取器，集成token节省策略"""

    # 化学相关关键词（大小写不敏感匹配）
    CHEMISTRY_KEYWORDS = [
        'reaction', 'substrate', 'product', 'catalyst', 'solvent',
        'table', 'entry', 'scheme', 'optimization', 'general procedure',
        'coupling', 'cross-coupling', 'suzuki', 'miyaura', 'heck',
        'sonogashira', 'buchwald', 'ullmann', 'negishi', 'stille',
        'boronic', 'boronate', 'aryl halide', 'palladium', 'pd',
        'biaryl', 'base', 'reduction', 'oxidation', 'hydrogenation',
        'amination', 'amidation', 'esterification', 'cyclization',
        'annulation', 'cycloaddition', 'enantioselective',
        'stereoselectiv', 'diastereoselectiv', 'regioselectiv',
        'asymmetric', 'chiral', 'enantiomer', 'photocatal',
        'irradiation', 'wavelength', 'sensitiz', 'photocycloaddition',
        'triplet', 'radical', 'dearomat', 'pericyclic',
    ]

    # 定量数据关键词（高权重，通常只出现在有反应数据的页）
    QUANT_KEYWORDS = [
        'yield', ' enantiomeric', ' ee ', ' er ', ' mmol', ' equiv',
        ' conv', 'diastereomeric',
    ]

    # 需要跳过的页面内容特征
    SKIP_PATTERNS = [
        r'\breferences?\b',
        r'\backnowledg(e)?ment(s)?\b',
        # r'\bsupporting information\b',  # 移除：SI每页都有此页眉，会误删反应页面
        r'\bgeneral methods\b',
        r'\binstrumentation\b',
        r'\bn\.?m\.?r\.?\s+data\b',
        r'\bcharacterization data\b',
        r'\bcopies of\b.*\bspectra\b',
    ]

    # GP 标题正则 — 用于定位 General Procedure 段落
    EXPLICIT_GP_TITLE_PATTERN = (
        r'(?im)^\s*(?:\d+[\).]\s*)?'
        r'(?:general\s+procedure|representative\s+procedure|typical\s+procedure|'
        r'standard\s+procedure|standard\s+conditions|experimental\s+procedure|procedure)'
        r'\s+[A-Z0-9]+\b\s*(?:\([^)\n]{0,60}\))?\s*(?::|\uff1a)'
    )

    SHORT_GP_TITLE_PATTERN = (
        r'(?im)^\s*(?:\d+[\).]\s*)?'
        r'GP[\s\-\u2010-\u2015]*[A-Z0-9]+\s*(?::|\uff1a|\.)'
    )

    GP_TITLE_PATTERNS = [
        EXPLICIT_GP_TITLE_PATTERN,
        SHORT_GP_TITLE_PATTERN,
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

    # =====================================================================
    # 未使用代码 (已弃用: 正则匹配Registry)
    # =====================================================================
    # NAME_PATTERNS 常量已弃用，现在使用GPT分块提取Registry
    # def _find_registry_pages() 和 _extract_registry_regex() 也已弃用
    
    # Stage1 screening prompt
    LEGACY_SCREEN_PROMPT = """You are a chemistry data screener. 
Look at the following readable text from a chemistry paper. 
Answer only with a JSON object:
{"has_reactions": true/false, "relevant_pages": [list of page numbers]}

"has_reactions" is true only if this text contains chemical reaction data described in paragraphs —
reactions with substrates, products, catalysts, conditions, yield, ee, or er. 
Look for sections like "Substrate Scope", "Optimization", "General Procedure", 
or any readable table/paragraph describing a chemical reaction. 
If no reaction data is found, set "has_reactions" to false and "relevant_pages" to an empty list. 
Do not count references, general descriptions, instrumentation, or NMR data as reaction data. 
Page numbers are marked as "--- Page N ---" in the text."""

    SCREEN_PROMPT = """You are a chemistry reaction-entry screener.

Task:
Decide whether the text contains extractable reaction entries written in paragraph/prose form.

Return ONLY valid JSON:
{"has_reactions": true/false, "relevant_pages": [list of page numbers], "reason": "short reason"}

Set "has_reactions" to true ONLY if the text contains at least one paragraph/prose reaction entry with:
- a specific product or substrate name/symbol, and
- a reported isolated yield, ee, or er.

Yield alone is sufficient. ee and er are optional.

Valid paragraph/prose reaction entries include:
- substrate scope or product scope entries written as sentences/paragraphs
- synthesis or preparation paragraphs for specific compounds
- product characterization entries that report a compound was synthesized, prepared, obtained, afforded, or furnished by following a named/general procedure and include an isolated amount and yield
  Example: "compound name (3a) was synthesized by following Procedure A ... to provide 3a ... (1.54 g, 75% yield)."

Set "has_reactions" to false for:
- tables, optimization tables, screening tables, entry tables, or figure captions
- General Procedure text by itself
- sections that only describe general methods, instrumentation, references, acknowledgements, or background
- NMR, HRMS, HPLC, spectra, exact mass, melting point, optical rotation, or analytical-only text
- title pages, table-of-contents pages, and pages with only headings or section titles

General Procedure text is context only. It is not an extractable reaction entry unless the same paragraph also reports a specific product/substrate and isolated yield, ee, or er.

Ignore table content completely, even if it contains substrates, products, conditions, yield, ee, er, or entry numbers.

For "relevant_pages", include only the page numbers that contain the extractable paragraph/prose reaction entries. Do not include pages that only provide context.
Page numbers are marked as "--- Page N ---" in the text."""

    TOC_DETECTION_PROMPT = """Find the table of contents in these first pages of a chemistry supporting information PDF.

Return ONLY compact valid JSON. Do not explain, reason aloud, or wrap the JSON in Markdown.

Return this exact shape:
{
  "has_toc": true,
  "toc_page_nums": [1],
  "sections": [
    {
      "title": "Preparation of Substrates",
      "printed_page": 3,
      "level": 1,
      "raw_line": "2. Preparation of Substrates. .... 3"
    }
  ]
}

Rules:
- Extract section titles and printed start pages only.
- printed_page is the page number shown in the table of contents, not the PDF page index.
- Do not extract chemical entities or symbol-name mappings.
- Do not infer missing sections.
- If no table of contents is present, return {"has_toc": false, "toc_page_nums": [], "sections": []}.
"""

    # Stage0 registry GPT prompt
    REGISTRY_PROMPT = """Extract symbol-to-chemical-name mappings from chemistry text.

CRITICAL: Only extract from compound DEFINITION statements where symbol and name appear in the SAME sentence.
CRITICAL: The symbol MUST be the EXACT label as written in the source text. Never renumber, reassign, or infer symbols. 

VALID patterns (symbol and name must be directly paired):
1. "full_chemical_name 1a" or "full_chemical_name (1a)" — name immediately before symbol
2. "1a = full_chemical_name" or "1a: full_chemical_name" — symbol before name
3. "Compound 1a: full_chemical_name" — explicit definition
4. "The substrate 1a was prepared from..." — name appears in same sentence

INVALID - DO NOT EXTRACT:
- Names from REACTION RESULTS (e.g., "reaction of 1b afforded 4 as...") — 4 is a PRODUCT, not a definition
- Names in different sentences from the symbol
- Names inferred from tables where symbol and name are far apart
- Names from optimization tables or reaction scope tables
- ANY mapping where the symbol appears in a RESULT context, not a DEFINITION context

Example of CORRECT extraction:
  Input: "3-(4-Methoxyphenyl)-1-(1-(o-tolyl)-1H-imidazol-2-yl)propan-1-one (1b) was used..."
  Output: {"1b": "3-(4-Methoxyphenyl)-1-(1-(o-tolyl)-1H-imidazol-2-yl)propan-1-one"}

Example of INCORRECT extraction (AVOID):
  Input: "((1R,2R,3S)-...methanone (4). Following GP D, reaction of 1b (32.0 mg)..."
  WRONG: {"1b": "((1R,2R,3S)-...methanone"} — 4 is NOT the definition of 1b!
  CORRECT: Skip 1b if no definition sentence exists nearby

Return ONLY valid JSON mapping symbol to FULL chemical name. Skip any symbol without a clear definition."""

    REGISTRY_PROMPT = """Extract symbol-to-chemical-name mappings from chemistry text.

CRITICAL: Extract only mappings that are explicitly present in the source text.
CRITICAL: The symbol MUST be the EXACT label as written in the source text. Never renumber, reassign, normalize, or infer symbols.
CRITICAL: Return the FULL chemical name text that appears directly paired with the symbol.

VALID patterns:
1. "full_chemical_name (1a)" or "full_chemical_name 1a" - compound heading, name immediately before symbol
2. "1a = full_chemical_name" or "1a: full_chemical_name" - symbol before name
3. "Compound 1a: full_chemical_name" - explicit definition
4. Characterization headings such as "full_product_name (3a)" followed by yield, NMR, HRMS, HPLC, or analytical data
5. Substrate/preparation headings such as "full_substrate_name (2t)" followed by "Prepared according to..." or similar text

Important:
- A standalone heading line like "methyl (E)-3-(4-ethynylphenyl)acrylate (2t)" is a valid definition.
- The next sentence does NOT need to repeat the full name if it refers to "compound 2t".
- "Prepared according to a published procedure to afford compound 2t" may confirm the heading "full name (2t)".
- Product characterization entries may define product labels such as 3a, 4, 5, or 6 when the full name appears immediately before the label.

INVALID - DO NOT EXTRACT:
- Do NOT map a substrate label to a product name. Example: "reaction of 1b afforded product_name (4)" means 4 may be product_name, but 1b is NOT product_name.
- Do NOT infer names from distant table columns, optimization tables, or condition tables.
- Do NOT create labels that are not present in the text.
- Do NOT map a symbol if only the symbol appears but the full chemical name is absent nearby.
- Do NOT map general descriptors such as "substrate 2t", "product 3a", "ligand L1", or "compound 4" as names.

Example of CORRECT extraction:
  Input: "methyl (E)-3-(4-ethynylphenyl)acrylate (2t)\nPrepared according to a published procedure to afford compound 2t."
  Output: {"2t": "methyl (E)-3-(4-ethynylphenyl)acrylate"}

Example of CORRECT extraction:
  Input: "12,15,32,35-hexamethoxy-tetrakis(phenylethynyl)-pentabenzenacyclodecaphane (3a)\nThe reaction was performed..."
  Output: {"3a": "12,15,32,35-hexamethoxy-tetrakis(phenylethynyl)-pentabenzenacyclodecaphane"}

Example of INCORRECT extraction (AVOID):
  Input: "product_name (4). Following GP D, reaction of 1b (32.0 mg) afforded 4..."
  WRONG: {"1b": "product_name"}
  CORRECT: {"4": "product_name"} if product_name is a full chemical name

Return ONLY valid JSON mapping symbol to FULL chemical name. Skip any symbol without a clear definition."""

    REGISTRY_SECTION_SELECTOR_PROMPT = """Select supporting-information sections that may contain symbol-to-chemical-name definition statements.

Return ONLY compact valid JSON with this exact shape:
{
  "sections": [
    {"section_index": 1, "decision": "include", "reason": "contains substrate synthesis definitions"}
  ]
}

Definitions are statements where labels such as 1a, 2b, L1, cat-1, S1, or product numbers are directly paired with full chemical names.

Decision rules:
- Use "include" for sections likely to contain compound definitions, synthetic procedures, analytical data, characterization, substrate/product preparation, ligands, catalysts, or photoreactions.
- Use "maybe" when the title is ambiguous but could contain definitions.
- Use "exclude" for references, spectra-only sections, crystallography-only sections, computational-only sections, general instrumentation, or unrelated measurements.
- Bias toward recall: if uncertain, use "maybe".
- Return exactly one item for every input section_index.
- Do not create, omit, renumber, or reorder section_index values.
"""

    REGISTRY_SECTION_VERIFIER_PROMPT = """Verify whether candidate sections may contain symbol-to-chemical-name definitions.

Return ONLY compact valid JSON with this exact shape:
{
  "sections": [
    {"section_index": 1, "registry_relevant": true, "reason": "opening text lists labeled compounds"}
  ]
}

Use the section title, page range, and preview text. A section is registry_relevant if it may contain definitions pairing labels/symbols with full chemical names for substrates, products, catalysts, ligands, reagents, intermediates, or prepared compounds.

Return exactly one item for every input section_index.
Do not create, omit, renumber, or reorder section_index values.
"""
    # General Procedure 注入到 Stage2 的 prompt 模板
    GP_INJECTION_TEMPLATE = """
GENERAL PROCEDURE CONTEXT — apply these conditions when the entry specifies none:

{gp_block}

RULES:
1. Entry-specified reagents/solvents/conditions ALWAYS override GP conditions.
2. Do NOT mix reaction types (e.g., NaBH4 = reduction, not photocatalysis).
3. If only a short label/code is present, keep it in symbol and do not invent a full name.
4. Do NOT set substrates = products.
5. Determine whether the supplied GP describes multiple chemical transformations or only one reaction followed by workup/purification.
6. If the GP is multi-step, preserve step boundaries using the multi-step schema. Do not flatten multi-step GP materials into an unlabelled list.
7. Do not invent an unnamed intermediate. Only an explicit intermediate name or symbol/code belongs in intermediates.
8. Normalize GP conditions by field meaning: solvent records solvent identity only; volume records quantities and short role notes for separate portions, addition solutions, suspensions, dilutions, or reaction mixtures. Do not duplicate quantities in both solvent and volume. Exclude workup, extraction, washing, and chromatography solvents unless they are the reaction medium.
"""

    STAGE2_AUDIT_PROMPT = """You are auditing a chemistry SI reaction extraction.
Compare the source text against the current extracted reaction JSON.

Return ONLY a JSON object:
{"missing_reactions": [<reaction objects>]}

Rules:
- Do not rewrite reactions already present in current_extraction.
- Add only reaction entries that are clearly present in the source text but missing from current_extraction.
- Coverage audit: review the source text in order and check whether every extractable paragraph/prose reaction entry is present in current_extraction.
- A missing reaction should be added when the source contains a specific compound/product/substrate name, label, code, or symbol; preparation/synthesis/obtained/afforded/furnished wording or an explicit procedure reference; and isolated yield, ee, or er.
- Pay special attention to consecutive repeated product characterization entries.
- Do not assume a generic General Procedure record covers later specific product entries.
- Each specific product entry with its own symbol/name and yield must have its own reaction object.
- Do not recover reactions from tables.
- Product characterization entries after a GP are valid reaction entries when they report an isolated product and yield.
- "Prepared according to General Procedure A/B using ..." entries are valid reaction entries.
- For GP-referenced entries, use the supplied GP context for substrates/reagents/conditions, but do not copy product names into substrates.
- When a missing entry uses a multi-step GP or reports an overall multi-step yield, first confirm that the GP/entry contains multiple chemical transformations, not only workup or purification, then reproduce the multi-step schema, step assignments, explicitly named intermediates, and per-step conditions from the extraction prompt.
- Normalize condition fields by meaning: solvent records solvent identity only; volume records quantities and short role notes for separate portions, addition solutions, suspensions, dilutions, or reaction mixtures. Do not duplicate quantities in both solvent and volume, and exclude workup/extraction/washing/chromatography solvents unless they are the reaction medium.
- If any substrate, product, catalyst, additive, reagent, intermediate, or condition uses a step field, the reaction must include integer step_count. Single-step reactions must not include step fields.
- Every missing reaction must include source_pages from the concrete entry's "--- Page N ---" markers. Do not use a supplied GP definition page; use [] if uncertain.
- Keep the current schema exactly: targets may contain yield, ee, and er only.
- Do not output dr, conversion, selectivity, NMR_yield, or GC_yield.
- If no reactions are missing, return {"missing_reactions": []}.
- Each missing reaction must use the same schema as the original extraction prompt."""

    GP_SUMMARY_PROMPT = """You are a chemistry data extraction specialist. 
Extract structured information from this General Procedure text.
GP Label: {gp_label}
Return ONLY a JSON object with these fields:
- substrates: 底物(可能是多个，符号或名称)
- catalysts: 催化剂/配体(可能是多个)  
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

    GP_TRUNCATION_PROMPT = """You are cleaning a chemistry Supporting Information general procedure.

Task:
- Identify where the actual general procedure definition ends.
- Do NOT rewrite or summarize chemistry text.
- Return a short exact end_anchor copied verbatim from the provided candidate text.
- The end_anchor should be the last sentence or phrase that still belongs to the general procedure.
- Exclude characterization sections, NMR/HRMS/HPLC data, optimization tables, screening tables, product entries, and examples that merely say "following/according to the general procedure".

Examples:
- If the procedure is followed by "Optimization of reaction conditions" or a table headed "Entry", the end_anchor should be the last purification/afforded sentence before that optimization section.
- If the procedure is followed by "Characterization and NMR spectra of products" and product entries such as "was synthesized by following Procedure A", the end_anchor should be the last sentence of the procedure before the characterization heading.

Return ONLY compact valid JSON:
{{
  "end_anchor": "exact copied text near the true end",
  "include_anchor": true,
  "trim_reason": "short reason",
  "confidence": "high|medium|low"
}}

If the whole candidate is the general procedure, return an end_anchor near the end and confidence "medium".

GP key: {gp_key}
GP title: {gp_title}

Candidate text:
{candidate_text}
"""

    GP_GENERIC_RESOLUTION_PROMPT = """Resolve which general procedure is referenced by a reaction chunk.

The chunk uses a generic phrase such as "according to the general procedure" or "following the general procedure" and does not name a procedure label.
Select only the GP keys that best match the reaction type, substrates, catalysts, conditions, or product series.

Return ONLY compact valid JSON:
{{
  "selected_gp_keys": ["GP key"],
  "confidence": "high|medium|low",
  "reason": "short reason"
}}

Reaction chunk:
{chunk_text}

Available GP candidates:
{gp_candidates}
"""

    GP_NO_REFERENCE_RESOLUTION_PROMPT = """Decide whether any general procedure should be used for a reaction chunk that does not explicitly name a procedure.

The chunk may contain product characterization entries, compound labels, product series, method wording, substrates, yields, ee/er, or brief "synthesized by using" details.
Select only GP keys that are clearly applicable based on product numbering/series, method labels, reaction family, substrates/reagents, catalysts, or conditions.
You may select multiple GP keys if the chunk clearly contains entries that use multiple procedures.
Return an empty selected_gp_keys list if no GP is applicable. Do not select a GP merely because it exists.

Return ONLY compact valid JSON:
{{
  "selected_gp_keys": ["GP key"],
  "confidence": "high|medium|low",
  "reason": "short reason"
}}

Reaction chunk:
{chunk_text}

Available GP candidates:
{gp_candidates}
"""

    def __init__(self, api_key: Optional[str] = None,
                 pages_per_chunk: int = 5,
                 screen_model: str = "gpt-5-mini",
                 extract_model: str = "gpt-5-mini",
                 base_url: str = "https://hk.xty.app/v1",
                 enable_stage2_audit: bool = True,
                 max_parallel_text_chunks: int = 1,
                 pdf_text_layout: str = "single",
                 pdf_text_x_tolerance: float = 3.0,
                 pdf_text_y_tolerance: float = 5.0):
        """
        初始化SI提取器

        Args:
            api_key: OpenAI API密钥
            pages_per_chunk: 每个分块的页数（默认5页）
            screen_model: 第一阶段筛选模型（默认gpt-5-mini）
            extract_model: 第二阶段提取模型（默认gpt-5-mini）
            base_url: OpenAI-compatible API base URL
        """
        super().__init__(api_key, base_url=base_url)
        self.pages_per_chunk = pages_per_chunk
        self.screen_model = screen_model
        self.extract_model = extract_model
        self.enable_stage2_audit = enable_stage2_audit
        self.max_parallel_text_chunks = max(1, int(max_parallel_text_chunks or 1))
        self.pdf_text_layout = (
            pdf_text_layout if pdf_text_layout in {"single", "two_column", "auto"} else "single"
        )
        self.pdf_text_x_tolerance = float(pdf_text_x_tolerance)
        self.pdf_text_y_tolerance = float(pdf_text_y_tolerance)
        if not math.isfinite(self.pdf_text_x_tolerance) or self.pdf_text_x_tolerance < 0:
            raise ValueError("pdf_text_x_tolerance must be a finite, non-negative number")
        if not math.isfinite(self.pdf_text_y_tolerance) or self.pdf_text_y_tolerance < 0:
            raise ValueError("pdf_text_y_tolerance must be a finite, non-negative number")
        self.stage2_audit_recovered = 0
        self.last_registry_validation_stats = {
            "registry_raw_count": 0,
            "registry_final_count": 0,
        }
        self.last_registry_debug = {}
        self._last_registry_chunk_debug = {}
        self._section_cache = {}
        self._section_debug_cache = {}
        self._toc_cache = {}
        self._gp_selection_lock = threading.Lock()
        self._last_toc_page_offset = None
        self.last_section_chunking_stats = {}
        self.last_section_debug = {}
        self.stats = {
            'total_pages': 0,
            'filtered_pages': 0,
            'total_chunks': 0,
            'screened_chunks': 0,
            'extracted_chunks': 0,
            'skipped_chunks': 0,
        }

    def _run_indexed_parallel(self, items, worker, max_workers: Optional[int] = None):
        """Run independent chunk jobs concurrently and return results in input order."""
        indexed_items = list(enumerate(items))
        if not indexed_items:
            return []
        worker_count = min(max(1, int(max_workers or self.max_parallel_text_chunks)), len(indexed_items))
        if worker_count <= 1:
            return [(index, worker(item)) for index, item in indexed_items]

        results = [None] * len(indexed_items)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for index, item in indexed_items:
                ctx = contextvars.copy_context()
                futures[executor.submit(ctx.run, worker, item)] = index
            for future in as_completed(futures):
                index = futures[future]
                results[index] = (index, future.result())
        return results
    
    # =====================================================================
    # 第一层：按页提取文本
    # =====================================================================

    def _is_likely_two_column_page(self, page) -> bool:
        """Detect common two-column article pages from word coordinates."""
        try:
            words = page.extract_words(use_text_flow=False, keep_blank_chars=False) or []
        except Exception:
            return False

        top_margin = page.height * 0.08
        bottom_margin = page.height * 0.94
        body_words = [
            w for w in words
            if top_margin <= float(w.get("top", 0)) <= bottom_margin
        ]
        if len(body_words) < 120:
            return False

        width = float(page.width)
        mid_left = width * 0.43
        mid_right = width * 0.57
        left_count = sum(1 for w in body_words if float(w.get("x0", 0)) < mid_left)
        right_count = sum(1 for w in body_words if float(w.get("x1", 0)) > mid_right)
        min_side_count = max(30, int(len(body_words) * 0.18))
        return (
            left_count >= min_side_count
            and right_count >= min_side_count
            and min(left_count, right_count) / max(left_count, right_count) >= 0.45
        )

    def _extract_two_column_text(self, page) -> str:
        """Extract page text left column first, then right column."""
        width = float(page.width)
        height = float(page.height)
        split_x = width / 2.0
        top = height * 0.06
        bottom = height * 0.96

        try:
            extract_kwargs = {
                "x_tolerance": self.pdf_text_x_tolerance,
                "y_tolerance": self.pdf_text_y_tolerance,
            }
            left = page.crop((0, top, split_x, bottom)).extract_text(**extract_kwargs) or ""
            right = page.crop((split_x, top, width, bottom)).extract_text(**extract_kwargs) or ""
        except Exception:
            return ""

        text = "\n\n".join(part.strip() for part in (left, right) if part and part.strip())
        return text.strip()

    def _extract_pdfplumber_page_text(self, page) -> str:
        default_text = page.extract_text(
            x_tolerance=self.pdf_text_x_tolerance,
            y_tolerance=self.pdf_text_y_tolerance,
        ) or ""
        if self.pdf_text_layout == "single":
            return default_text

        use_two_column = self.pdf_text_layout == "two_column" or (
            self.pdf_text_layout == "auto" and self._is_likely_two_column_page(page)
        )
        if not use_two_column:
            return default_text

        column_text = self._extract_two_column_text(page)
        if len(column_text.strip()) < max(80, int(len(default_text.strip()) * 0.4)):
            return default_text
        return column_text

    def extract_text_by_pages(self, pdf_path: str) -> List[Dict]:
        """
        按页提取PDF文本，返回页列表

        Returns:
            [{"page_num": 1, "text": "..."}, ...]
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        pages = []

        if PDF_LIBRARY == "pdfplumber":
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = self._extract_pdfplumber_page_text(page)
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
    # 第二层：关键词过滤
    # =====================================================================

    def is_page_relevant(self, page_text: str) -> bool:
        """
        判断一页是否包含化学相关数据

        使用两种关键词加权判断：
        - 定量关键词（yield, ee, mmol等）权重高
        - 一般关键词（catalyst, reaction等）权重低
        """
        text_lower = page_text.lower()

        # 定量关键词计分（权重3）
        quant_score = 0
        for kw in self.QUANT_KEYWORDS:
            count = text_lower.count(kw.lower())
            quant_score += count * 3

        # 一般关键词计分（权重1）
        general_score = 0
        for kw in self.CHEMISTRY_KEYWORDS:
            count = text_lower.count(kw.lower())
            general_score += count

        total_score = quant_score + general_score

        # 阈值：总分 >= 5 视为相关页
        return total_score >= 3

    def filter_relevant_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        过滤页面，只保留化学相关页

        Returns:
            过滤后的页列表
        """
        filtered = [p for p in pages if self.is_page_relevant(p['text'])]
        return filtered

    # =====================================================================
    # Runtime section detection and section-aware chunking
    # =====================================================================

    def _pages_cache_key(self, pages: List[Dict]) -> Tuple:
        if not pages:
            return ("empty", 0)
        page_nums = tuple(p.get("page_num") for p in pages)
        total_chars = sum(len(str(p.get("text") or "")) for p in pages)
        return (page_nums, total_chars)

    def _strip_code_fence(self, raw: str) -> str:
        raw = (raw or "").strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return raw.strip()

    def _normalize_section_title(self, value: str) -> str:
        text = str(value or "").lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _section_title_matches_page(self, title: str, page_text: str) -> bool:
        title_norm = self._normalize_section_title(title)
        if len(title_norm) < 6:
            return False
        head = "\n".join(str(page_text or "").splitlines()[:12])
        head_norm = self._normalize_section_title(head)
        if title_norm in head_norm:
            return True
        words = [w for w in title_norm.split() if len(w) > 2]
        return len(words) >= 2 and all(w in head_norm for w in words[:4])

    def detect_toc_sections(self, pages: List[Dict]) -> Dict:
        probe_pages = pages[:5]
        if not probe_pages:
            return {"has_toc": False, "toc_page_nums": [], "sections": []}

        cache_key = self._pages_cache_key(probe_pages)
        if cache_key in self._toc_cache:
            return self._toc_cache[cache_key]

        probe_text = "\n\n".join(
            f"--- Page {p.get('page_num')} ---\n{p.get('text', '')}"
            for p in probe_pages
        )
        try:
            response = self.client.chat.completions.create(
                model=self.extract_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract table-of-contents metadata. Return compact valid JSON only.",
                    },
                    {"role": "user", "content": f"{self.TOC_DETECTION_PROMPT}\n\n{probe_text}"},
                ],
                temperature=0.0,
            )
            raw = self._strip_code_fence(response.choices[0].message.content)
            if not raw:
                raise ValueError("empty TOC response")
            data = json.loads(raw)
        except Exception as exc:
            print(f"  [SectionChunking] TOC detection failed: {exc}")
            result = {"has_toc": False, "toc_page_nums": [], "sections": [], "_error": str(exc)}
            self._toc_cache[cache_key] = result
            return result

        if not isinstance(data, dict) or not data.get("has_toc"):
            result = {"has_toc": False, "toc_page_nums": [], "sections": []}
            self._toc_cache[cache_key] = result
            return result

        sections = []
        for item in data.get("sections") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            try:
                printed_page = int(item.get("printed_page"))
            except (TypeError, ValueError):
                continue
            if not title or printed_page <= 0:
                continue
            try:
                level = int(item.get("level") or 1)
            except (TypeError, ValueError):
                level = 1
            sections.append({
                "title": title,
                "printed_page": printed_page,
                "level": level,
                "raw_line": str(item.get("raw_line") or "").strip(),
            })

        result = {
            "has_toc": bool(sections),
            "toc_page_nums": data.get("toc_page_nums") or [],
            "sections": sections,
        }
        self._toc_cache[cache_key] = result
        return result

    def _calibrate_toc_page_offset(
        self,
        pages: List[Dict],
        toc_sections: List[Dict],
        toc_page_nums: Optional[List[int]] = None,
    ) -> Optional[int]:
        toc_pages = {p for p in (toc_page_nums or []) if isinstance(p, int)}
        offsets = {}
        for section in toc_sections:
            title = section.get("title") or ""
            printed_page = section.get("printed_page")
            if not isinstance(printed_page, int):
                continue
            for page in pages:
                page_num = page.get("page_num")
                if not isinstance(page_num, int):
                    continue
                if page_num in toc_pages:
                    continue
                if self._section_title_matches_page(title, str(page.get("text") or "")):
                    offset = page_num - printed_page
                    offsets[offset] = offsets.get(offset, 0) + 1
                    break
        if not offsets:
            return None
        return sorted(offsets.items(), key=lambda kv: (-kv[1], abs(kv[0])))[0][0]

    def _sections_from_toc(self, pages: List[Dict], toc: Dict) -> List[Dict]:
        toc_sections = toc.get("sections") or []
        if not toc_sections:
            return []

        offset = self._calibrate_toc_page_offset(pages, toc_sections, toc.get("toc_page_nums") or [])
        self._last_toc_page_offset = offset
        if offset is None:
            return []

        available = [p.get("page_num") for p in pages if isinstance(p.get("page_num"), int)]
        if not available:
            return []
        min_page = min(available)
        max_page = max(available)

        starts = []
        for item in toc_sections:
            printed_page = item.get("printed_page")
            if not isinstance(printed_page, int):
                continue
            actual_start = printed_page + offset
            if min_page <= actual_start <= max_page:
                starts.append({
                    "title": item.get("title") or "",
                    "start_page": actual_start,
                    "source": "toc",
                })
        starts.sort(key=lambda x: x["start_page"])

        sections = []
        for idx, item in enumerate(starts):
            next_start = starts[idx + 1]["start_page"] if idx + 1 < len(starts) else max_page + 1
            end_page = min(max_page, next_start - 1)
            if item["start_page"] <= end_page:
                sections.append({
                    "title": item["title"],
                    "start_page": item["start_page"],
                    "end_page": end_page,
                    "source": "toc",
                })
        return sections

    def _is_likely_section_heading(self, line: str) -> bool:
        stripped = re.sub(r"\s+", " ", str(line or "")).strip()
        if not stripped or len(stripped) > 140:
            return False
        if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9,()/\[\]\- ]{3,}$", stripped):
            return True
        heading_terms = (
            "general information", "preparation", "synthesis", "reaction optimization",
            "substrate scope", "reaction scope", "general procedure", "standard procedure",
            "derivatization", "references", "spectra", "cartesian coordinates",
            "computational", "dft",
        )
        lower = stripped.lower()
        return any(term in lower for term in heading_terms) and len(stripped.split()) <= 12

    def _sections_from_headings(self, pages: List[Dict]) -> List[Dict]:
        starts = []
        for page in pages:
            page_num = page.get("page_num")
            if not isinstance(page_num, int):
                continue
            for line in str(page.get("text") or "").splitlines()[:16]:
                if self._is_likely_section_heading(line):
                    starts.append({"title": line.strip(), "start_page": page_num, "source": "heading"})
                    break
        if not starts:
            return []

        deduped = []
        seen_pages = set()
        for item in sorted(starts, key=lambda x: x["start_page"]):
            if item["start_page"] in seen_pages:
                continue
            seen_pages.add(item["start_page"])
            deduped.append(item)

        max_page = max(p.get("page_num") for p in pages if isinstance(p.get("page_num"), int))
        sections = []
        for idx, item in enumerate(deduped):
            next_start = deduped[idx + 1]["start_page"] if idx + 1 < len(deduped) else max_page + 1
            end_page = min(max_page, next_start - 1)
            if item["start_page"] <= end_page:
                sections.append({
                    "title": item["title"],
                    "start_page": item["start_page"],
                    "end_page": end_page,
                    "source": item["source"],
                })
        return sections

    def split_sections_to_chunks(
        self,
        sections: List[Dict],
        pages: List[Dict],
        max_pages: int = 5,
        max_chars: int = 8000,
    ) -> List[Dict]:
        pages_by_num = {
            p.get("page_num"): p
            for p in pages
            if isinstance(p.get("page_num"), int)
        }
        section_chunks = []
        for section in sections:
            section_pages = [
                pages_by_num[num]
                for num in sorted(pages_by_num)
                if section["start_page"] <= num <= section["end_page"]
            ]
            if not section_pages:
                continue

            parts = self._split_pages_with_limits(
                section_pages,
                page_limit=max_pages,
                char_limit=max_chars,
                section_title=section.get("title", ""),
                chunk_strategy="section",
            )
            for part_index, part in enumerate(parts, 1):
                page_nums = part.get("page_nums") or []
                if not page_nums:
                    continue
                section_chunks.append({
                    "section_title": section.get("title", ""),
                    "section_start_page": section.get("start_page"),
                    "section_end_page": section.get("end_page"),
                    "start_page": page_nums[0],
                    "end_page": page_nums[-1],
                    "part_index": part_index,
                    "source": section.get("source", ""),
                })
        return section_chunks

    def get_runtime_sections(self, pages: List[Dict]) -> List[Dict]:
        key = self._pages_cache_key(pages)
        if key in self._section_cache:
            self.last_section_debug = self._section_debug_cache.get(key, {})
            self.last_section_chunking_stats = self.last_section_debug.get("stats", {})
            return self._section_cache[key]

        toc = self.detect_toc_sections(pages)
        self._last_toc_page_offset = None
        raw_sections = self._sections_from_toc(pages, toc)
        source = "toc" if raw_sections else "fixed_fallback"

        section_chunks = self.split_sections_to_chunks(raw_sections, pages) if raw_sections else []
        self.last_section_chunking_stats = {
            "section_chunking_enabled": bool(section_chunks),
            "section_chunking_source": source,
            "section_scope": "full_document",
            "raw_section_count": len(raw_sections),
            "section_count": len(raw_sections),
            "section_chunk_count": len(section_chunks),
            "section_chunk_max_pages": 5,
            "section_chunk_max_chars": 8000,
            "toc_detected": bool(toc.get("has_toc")),
        }
        self.last_section_debug = {
            "source": source,
            "toc_page_nums": toc.get("toc_page_nums") or [],
            "toc_page_offset": self._last_toc_page_offset,
            "toc_sections": toc.get("sections") or [],
            "toc_error": toc.get("_error"),
            "raw_sections": raw_sections,
            "section_chunks": section_chunks,
            "stats": self.last_section_chunking_stats,
        }
        self._section_cache[key] = section_chunks
        self._section_debug_cache[key] = self.last_section_debug
        return section_chunks

    def _page_range_label(self, page_nums: List[int]) -> str:
        if not page_nums:
            return ""
        return f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])

    def _format_chunk_text(self, pages: List[Dict]) -> str:
        return "\n\n".join(f"--- Page {p['page_num']} ---\n{p['text']}" for p in pages)

    def _pages_from_chunk_text(self, chunk: Dict) -> Dict[int, str]:
        text = str(chunk.get("text") or "")
        matches = list(re.finditer(r"--- Page\s+(\d+)\s+---\s*\n?", text))
        if not matches:
            return {}

        pages = {}
        for idx, match in enumerate(matches):
            try:
                page_num = int(match.group(1))
            except (TypeError, ValueError):
                continue
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            pages[page_num] = text[start:end].strip()
        return pages

    def _stage1_relevant_page_nums(self, chunk: Dict, screen: Dict) -> List[int]:
        raw_pages = screen.get("relevant_pages") if isinstance(screen, dict) else None
        if not isinstance(raw_pages, list):
            return []

        allowed_pages = []
        for value in chunk.get("page_nums") or []:
            try:
                allowed_pages.append(int(value))
            except (TypeError, ValueError):
                continue
        allowed = set(allowed_pages)
        if not allowed:
            return []

        result = []
        seen = set()
        for value in raw_pages:
            try:
                page_num = int(value)
            except (TypeError, ValueError):
                continue
            if page_num in allowed and page_num not in seen:
                result.append(page_num)
                seen.add(page_num)
        return result

    def _trim_chunk_to_stage1_pages(self, chunk: Dict, screen: Dict) -> Dict:
        relevant_page_nums = self._stage1_relevant_page_nums(chunk, screen)
        if not relevant_page_nums:
            return chunk

        original_page_nums = []
        for value in chunk.get("page_nums") or []:
            try:
                original_page_nums.append(int(value))
            except (TypeError, ValueError):
                continue
        if not original_page_nums or set(relevant_page_nums) == set(original_page_nums):
            return chunk

        page_text_by_num = self._pages_from_chunk_text(chunk)
        if any(page_num not in page_text_by_num for page_num in relevant_page_nums):
            return chunk

        trimmed_pages = [
            {"page_num": page_num, "text": page_text_by_num[page_num]}
            for page_num in original_page_nums
            if page_num in set(relevant_page_nums)
        ]
        if not trimmed_pages:
            return chunk

        trimmed = dict(chunk)
        page_nums = [p["page_num"] for p in trimmed_pages]
        chunk_text = self._format_chunk_text(trimmed_pages)
        trimmed["page_nums"] = page_nums
        trimmed["page_range"] = self._page_range_label(page_nums)
        trimmed["text"] = chunk_text
        trimmed["approx_tokens"] = len(chunk_text) // 4
        return trimmed

    def _split_pages_with_limits(
        self,
        pages: List[Dict],
        page_limit: int,
        char_limit: int = 8000,
        section_title: str = "",
        chunk_strategy: str = "section",
    ) -> List[Dict]:
        chunks = []
        current = []
        current_chars = 0
        page_limit = max(1, int(page_limit or 1))

        for page in pages:
            page_chars = len(str(page.get("text") or ""))
            if current and (len(current) >= page_limit or current_chars + page_chars > char_limit):
                chunks.append((current, section_title))
                current = []
                current_chars = 0
            current.append(page)
            current_chars += page_chars

        if current:
            chunks.append((current, section_title))

        result = []
        for chunk_pages, title in chunks:
            page_nums = [p["page_num"] for p in chunk_pages]
            chunk_text = self._format_chunk_text(chunk_pages)
            result.append({
                "chunk_id": 0,
                "page_range": self._page_range_label(page_nums),
                "page_nums": page_nums,
                "text": chunk_text,
                "approx_tokens": len(chunk_text) // 4,
                "section_title": title,
                "chunk_strategy": chunk_strategy,
            })
        return result

    def _assign_chunk_ids(self, chunks: List[Dict]) -> List[Dict]:
        for idx, chunk in enumerate(chunks, 1):
            chunk["chunk_id"] = idx
        return chunks

    def _chunk_pages_fixed(self, pages: List[Dict], pages_per_chunk: int) -> List[Dict]:
        chunks = []
        for i in range(0, len(pages), pages_per_chunk):
            chunk_pages = pages[i:i + pages_per_chunk]
            page_nums = [p['page_num'] for p in chunk_pages]
            chunk_text = self._format_chunk_text(chunk_pages)
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "page_range": self._page_range_label(page_nums),
                "page_nums": page_nums,
                "text": chunk_text,
                "approx_tokens": len(chunk_text) // 4,
                "chunk_strategy": "fixed_pages",
            })
        return chunks

    def chunk_selected_pages_by_sections(
        self,
        selected_pages: List[Dict],
        all_pages: List[Dict],
        pages_per_chunk: int,
        task_name: str,
    ) -> List[Dict]:
        section_chunks = self.get_runtime_sections(all_pages)
        if not section_chunks:
            strategy_key = f"{task_name}_chunk_strategy"
            self.last_section_chunking_stats = {
                **getattr(self, "last_section_chunking_stats", {}),
                "section_scope": "full_document",
                strategy_key: "fixed_pages",
            }
            return self._chunk_pages_fixed(selected_pages, pages_per_chunk)

        selected_by_num = {
            p.get("page_num"): p
            for p in selected_pages
            if isinstance(p.get("page_num"), int)
        }
        chunks = []
        covered = set()
        for section_chunk in section_chunks:
            chunk_pages = [
                selected_by_num[num]
                for num in sorted(selected_by_num)
                if section_chunk["start_page"] <= num <= section_chunk["end_page"]
            ]
            if not chunk_pages:
                continue
            covered.update(p["page_num"] for p in chunk_pages)
            page_nums = [p["page_num"] for p in chunk_pages]
            chunk_text = self._format_chunk_text(chunk_pages)
            chunks.append({
                "chunk_id": 0,
                "page_range": self._page_range_label(page_nums),
                "page_nums": page_nums,
                "text": chunk_text,
                "approx_tokens": len(chunk_text) // 4,
                "section_title": section_chunk.get("section_title", ""),
                "section_start_page": section_chunk.get("section_start_page"),
                "section_end_page": section_chunk.get("section_end_page"),
                "section_part_index": section_chunk.get("part_index"),
                "chunk_strategy": "section",
            })

        uncovered = [p for p in selected_pages if p.get("page_num") not in covered]
        if uncovered:
            chunks.extend(self._chunk_pages_fixed(uncovered, pages_per_chunk))

        if not chunks:
            return self._chunk_pages_fixed(selected_pages, pages_per_chunk)

        strategy_key = f"{task_name}_chunk_strategy"
        self.last_section_chunking_stats = {
            **getattr(self, "last_section_chunking_stats", {}),
            "section_scope": "full_document",
            strategy_key: "section",
        }
        return self._assign_chunk_ids(chunks)

    def chunk_pages_by_sections(self, selected_pages: List[Dict], all_pages: List[Dict]) -> List[Dict]:
        return self.chunk_selected_pages_by_sections(
            selected_pages=selected_pages,
            all_pages=all_pages,
            pages_per_chunk=self.pages_per_chunk,
            task_name="reaction",
        )

    # =====================================================================
    # Stage 0: Name Registry — 符号→完整化学名称映射
    # =====================================================================

    def _chunk_pages_for_registry(
        self,
        selected_pages: List[Dict],
        pages_per_chunk: int = 5,
        all_pages: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """将页面按指定数量分块（用于registry提取）

        Args:
            pages: 页列表
            pages_per_chunk: 每块页数

        Returns:
            [{"chunk_id": 1, "text": "...", "page_nums": [1,2,...]}, ...]
        """
        return self.chunk_selected_pages_by_sections(
            selected_pages=selected_pages,
            all_pages=all_pages or selected_pages,
            pages_per_chunk=pages_per_chunk,
            task_name="registry",
        )

    def _chat_json_for_registry_selection(self, prompt: str, payload: Dict, max_tokens: int = 4000) -> Tuple[Optional[Dict], str, Optional[str]]:
        messages = [
            {
                "role": "system",
                "content": "You select chemistry document sections. Return compact valid JSON only.",
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nINPUT JSON:\n{json.dumps(payload, ensure_ascii=False)}",
            },
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.extract_model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as exc:
            return None, "", f"{type(exc).__name__}: {exc}"
        raw = self._strip_code_fence(response.choices[0].message.content)
        if not raw:
            return None, "", "empty response"
        try:
            data = json.loads(raw)
        except Exception as exc:
            return None, raw, f"{type(exc).__name__}: {exc}"
        if not isinstance(data, dict):
            return None, raw, "response JSON is not an object"
        return data, raw, None

    def _validate_registry_selector_response(self, data: Dict, input_indices: List[int]) -> Tuple[Optional[List[Dict]], Optional[str]]:
        rows = data.get("sections")
        if not isinstance(rows, list):
            return None, "missing sections list"
        expected = set(input_indices)
        seen = set()
        validated = []
        for row in rows:
            if not isinstance(row, dict):
                return None, "selector row is not an object"
            try:
                section_index = int(row.get("section_index"))
            except (TypeError, ValueError):
                return None, "selector row missing valid section_index"
            if section_index not in expected:
                return None, f"selector section_index {section_index} not in input"
            decision = str(row.get("decision") or "").strip().lower()
            if decision not in {"include", "maybe", "exclude"}:
                return None, f"invalid selector decision for section_index {section_index}: {decision}"
            seen.add(section_index)
            validated.append({
                "section_index": section_index,
                "decision": decision,
                "reason": str(row.get("reason") or "").strip(),
            })
        missing = expected - seen
        if missing:
            return None, f"selector missing section_index values: {sorted(missing)[:10]}"
        return sorted(validated, key=lambda r: r["section_index"]), None

    def _validate_registry_verifier_response(self, data: Dict, input_indices: List[int]) -> Tuple[Optional[List[Dict]], Optional[str]]:
        rows = data.get("sections")
        if not isinstance(rows, list):
            return None, "missing sections list"
        expected = set(input_indices)
        seen = set()
        validated = []
        for row in rows:
            if not isinstance(row, dict):
                return None, "verifier row is not an object"
            try:
                section_index = int(row.get("section_index"))
            except (TypeError, ValueError):
                return None, "verifier row missing valid section_index"
            if section_index not in expected:
                return None, f"verifier section_index {section_index} not in candidates"
            relevant = row.get("registry_relevant")
            if not isinstance(relevant, bool):
                return None, f"registry_relevant must be boolean for section_index {section_index}"
            seen.add(section_index)
            validated.append({
                "section_index": section_index,
                "registry_relevant": relevant,
                "reason": str(row.get("reason") or "").strip(),
            })
        missing = expected - seen
        if missing:
            return None, f"verifier missing section_index values: {sorted(missing)[:10]}"
        return sorted(validated, key=lambda r: r["section_index"]), None

    def _section_preview(self, section: Dict, pages: List[Dict], char_limit: int = 1200) -> str:
        start = section.get("start_page")
        end = section.get("end_page")
        texts = []
        for page in pages:
            page_num = page.get("page_num")
            if isinstance(page_num, int) and isinstance(start, int) and isinstance(end, int) and start <= page_num <= end:
                text = str(page.get("text") or "").strip()
                if text:
                    texts.append(f"--- Page {page_num} ---\n{text}")
            if sum(len(t) for t in texts) >= char_limit:
                break
        preview = "\n\n".join(texts)
        return preview[:char_limit]

    def _raw_sections_for_registry_selection(self) -> List[Dict]:
        debug = getattr(self, "last_section_debug", {}) or {}
        raw_sections = debug.get("raw_sections") or []
        result = []
        for idx, section in enumerate(raw_sections, 1):
            result.append({
                "section_index": idx,
                "title": section.get("title") or "",
                "start_page": section.get("start_page"),
                "end_page": section.get("end_page"),
                "source": section.get("source") or "",
            })
        return result

    def _chunks_for_selected_registry_sections(self, selected_sections: List[Dict], section_chunks: List[Dict], pages: List[Dict]) -> List[Dict]:
        selected_keys = {
            (
                section.get("title") or "",
                section.get("start_page"),
                section.get("end_page"),
            )
            for section in selected_sections
        }
        pages_by_num = {
            page.get("page_num"): page
            for page in pages
            if isinstance(page.get("page_num"), int)
        }
        chunks = []
        for section_chunk in section_chunks:
            key = (
                section_chunk.get("section_title") or "",
                section_chunk.get("section_start_page"),
                section_chunk.get("section_end_page"),
            )
            if key not in selected_keys:
                continue
            start_page = section_chunk.get("start_page")
            end_page = section_chunk.get("end_page")
            if not isinstance(start_page, int) or not isinstance(end_page, int):
                continue
            chunk_pages = [
                pages_by_num[num]
                for num in sorted(pages_by_num)
                if start_page <= num <= end_page
            ]
            if not chunk_pages:
                continue
            page_nums = [p["page_num"] for p in chunk_pages]
            chunk_text = self._format_chunk_text(chunk_pages)
            chunks.append({
                "chunk_id": 0,
                "page_range": self._page_range_label(page_nums),
                "page_nums": page_nums,
                "text": chunk_text,
                "approx_tokens": len(chunk_text) // 4,
                "section_title": section_chunk.get("section_title", ""),
                "section_start_page": section_chunk.get("section_start_page"),
                "section_end_page": section_chunk.get("section_end_page"),
                "section_part_index": section_chunk.get("part_index"),
                "chunk_strategy": "registry_section_llm",
            })
        return self._assign_chunk_ids(chunks)

    def _select_registry_chunks_by_sections(self, pages: List[Dict], pages_per_chunk: int) -> Tuple[List[Dict], Dict]:
        section_chunks = self.get_runtime_sections(pages)
        section_stats = getattr(self, "last_section_chunking_stats", {}) or {}
        base_stats = {
            **section_stats,
            "registry_selection_strategy": "skipped",
            "registry_selection_status": "not_started",
            "registry_chunk_strategy": "skipped",
            "registry_sections_selected": 0,
            "registry_sections_rejected": 0,
            "registry_fallback_scan_pages": 20,
            "registry_selector_error": None,
            "registry_verifier_error": None,
        }
        if not section_chunks:
            scan_pages = pages[:20]
            fallback_chunks = self._chunk_pages_fixed(scan_pages, pages_per_chunk)
            stats = {
                **base_stats,
                "registry_selection_strategy": "fixed_20_pages_fallback",
                "registry_selection_status": "section_unavailable_fallback",
                "registry_chunk_strategy": "fixed_pages",
                "registry_chunks_total": len(fallback_chunks),
            }
            self._update_registry_selection_debug(stats=stats)
            return fallback_chunks, stats

        raw_sections = self._raw_sections_for_registry_selection()
        if not raw_sections:
            stats = {
                **base_stats,
                "registry_selection_strategy": "skipped",
                "registry_selection_status": "no_sections_selected",
                "registry_chunk_strategy": "skipped",
                "registry_chunks_total": 0,
            }
            self._update_registry_selection_debug(stats=stats)
            return [], stats

        selector_payload = {
            "allowed_section_indices": [s["section_index"] for s in raw_sections],
            "sections": [
                {
                    "section_index": s["section_index"],
                    "title": s["title"],
                    "start_page": s["start_page"],
                    "end_page": s["end_page"],
                }
                for s in raw_sections
            ]
        }
        selector_data, selector_raw, selector_error = self._chat_json_for_registry_selection(
            self.REGISTRY_SECTION_SELECTOR_PROMPT,
            selector_payload,
        )
        selector_rows = None
        if selector_error is None:
            selector_rows, selector_error = self._validate_registry_selector_response(
                selector_data,
                [s["section_index"] for s in raw_sections],
            )
        if selector_error:
            stats = {
                **base_stats,
                "registry_selection_strategy": "skipped",
                "registry_selection_status": "selector_failed",
                "registry_chunk_strategy": "skipped",
                "registry_chunks_total": 0,
                "registry_selector_error": selector_error,
            }
            self._update_registry_selection_debug(
                stats=stats,
                selector_rows=selector_rows or [],
                selector_raw=selector_raw,
            )
            return [], stats

        candidate_indices = {
            row["section_index"]
            for row in selector_rows
            if row["decision"] in {"include", "maybe"}
        }
        candidate_sections = [s for s in raw_sections if s["section_index"] in candidate_indices]
        if not candidate_sections:
            stats = {
                **base_stats,
                "registry_selection_strategy": "section_llm",
                "registry_selection_status": "no_sections_selected",
                "registry_chunk_strategy": "section_llm",
                "registry_sections_rejected": len(raw_sections),
                "registry_chunks_total": 0,
            }
            self._update_registry_selection_debug(
                stats=stats,
                selector_rows=selector_rows,
                selected_sections=[],
                selected_chunks=[],
                selector_raw=selector_raw,
            )
            return [], stats

        verifier_payload = {
            "allowed_section_indices": [s["section_index"] for s in candidate_sections],
            "sections": [
                {
                    "section_index": s["section_index"],
                    "title": s["title"],
                    "start_page": s["start_page"],
                    "end_page": s["end_page"],
                    "preview": self._section_preview(s, pages, char_limit=1200),
                }
                for s in candidate_sections
            ]
        }
        verifier_data, verifier_raw, verifier_error = self._chat_json_for_registry_selection(
            self.REGISTRY_SECTION_VERIFIER_PROMPT,
            verifier_payload,
        )
        verifier_rows = None
        if verifier_error is None:
            verifier_rows, verifier_error = self._validate_registry_verifier_response(
                verifier_data,
                [s["section_index"] for s in candidate_sections],
            )
        if verifier_error:
            stats = {
                **base_stats,
                "registry_selection_strategy": "skipped",
                "registry_selection_status": "verifier_failed",
                "registry_chunk_strategy": "skipped",
                "registry_chunks_total": 0,
                "registry_selector_error": None,
                "registry_verifier_error": verifier_error,
            }
            self._update_registry_selection_debug(
                stats=stats,
                selector_rows=selector_rows,
                verifier_rows=verifier_rows or [],
                selector_raw=selector_raw,
                verifier_raw=verifier_raw,
            )
            return [], stats

        relevant_indices = {
            row["section_index"]
            for row in verifier_rows
            if row["registry_relevant"]
        }
        selected_sections = [s for s in raw_sections if s["section_index"] in relevant_indices]
        selected_chunks = self._chunks_for_selected_registry_sections(selected_sections, section_chunks, pages)
        status = "selected" if selected_sections and selected_chunks else "no_sections_selected"
        stats = {
            **base_stats,
            "registry_selection_strategy": "section_llm",
            "registry_selection_status": status,
            "registry_chunk_strategy": "section_llm",
            "registry_sections_selected": len(selected_sections),
            "registry_sections_rejected": len(raw_sections) - len(selected_sections),
            "registry_chunks_total": len(selected_chunks),
        }
        self._update_registry_selection_debug(
            stats=stats,
            selector_rows=selector_rows,
            verifier_rows=verifier_rows,
            selected_sections=selected_sections,
            selected_chunks=selected_chunks,
            selector_raw=selector_raw,
            verifier_raw=verifier_raw,
        )
        return selected_chunks, stats

    def _update_registry_selection_debug(
        self,
        *,
        stats: Dict,
        selector_rows: Optional[List[Dict]] = None,
        verifier_rows: Optional[List[Dict]] = None,
        selected_sections: Optional[List[Dict]] = None,
        selected_chunks: Optional[List[Dict]] = None,
        selector_raw: str = "",
        verifier_raw: str = "",
    ) -> None:
        debug = dict(getattr(self, "last_section_debug", {}) or {})
        registry_selection = {
            "selector_rows": selector_rows or [],
            "verifier_rows": verifier_rows or [],
            "stats": stats,
        }
        debug_stats = {
            **(debug.get("stats") or {}),
            **stats,
        }
        debug.update({
            "registry_section_selection": registry_selection,
            "selected_registry_sections": selected_sections or [],
            "selected_registry_section_chunks": [
                {k: v for k, v in chunk.items() if k != "text"}
                for chunk in (selected_chunks or [])
            ],
            "registry_selector_raw_preview": (selector_raw or "")[:1000],
            "registry_verifier_raw_preview": (verifier_raw or "")[:1000],
            "registry_selector_error": stats.get("registry_selector_error"),
            "registry_verifier_error": stats.get("registry_verifier_error"),
            "stats": debug_stats,
        })
        self.last_section_debug = debug

    def _merge_registry_results(self, registries: List[Dict[str, str]]) -> Dict[str, str]:
        """合并多个registry结果，保留最先出现的名称
        
        Args:
            registries: registry列表
            
        Returns:
            合并后的registry
        """
        merged = {}
        
        for registry in registries:
            for symbol, name in registry.items():
                # 保留首次出现的名称，后续同symbol不覆盖
                if symbol not in merged:
                    merged[symbol] = name
        
        return merged

    def extract_name_registry(self, pages: List[Dict],
                              max_scan_pages: int = 20,
                              pages_per_chunk: int = 5) -> Dict[str, str]:
        """
        提取 symbol→full_chemical_name 映射表

        策略：section-aware LLM选择 → GPT分块提取 → 合并去重 → 清理

        Args:
            pages: 按页提取的文本列表
            max_scan_pages: 兼容旧调用；仅在section不可用的fallback中保留前20页扫描
            pages_per_chunk: 每块页数

        Returns:
            {"1a": "(E)-3-(4-methoxyphenyl)-...", "2a": "...", "C1": "..."}
        """
        if not pages:
            self.last_registry_validation_stats = {
                "registry_raw_count": 0,
                "registry_final_count": 0,
                "registry_selection_strategy": "skipped",
                "registry_selection_status": "no_pages",
            }
            self.last_registry_debug = {
                "strategy": "skipped",
                "status": "no_pages",
                "section_selection": {},
                "chunks": [],
                "section_debug": {},
                "selection_stats": {},
                "summary": {
                    "raw_registry_merged": {},
                    "final_registry": {},
                    "registry_raw_count": 0,
                    "registry_final_count": 0,
                },
            }
            return {}
        
        # --- Step 1: section-aware registry chunk selection ---
        print("  [Registry Step 1] section-aware registry chunk selection...")
        chunks, selection_stats = self._select_registry_chunks_by_sections(
            pages,
            pages_per_chunk=pages_per_chunk,
        )
        if selection_stats.get("registry_selection_status") in {
            "selector_failed",
            "verifier_failed",
            "no_sections_selected",
        }:
            print(f"  [Registry] skipped: {selection_stats.get('registry_selection_status')}")
            self.last_registry_validation_stats = {
                "registry_raw_count": 0,
                "registry_final_count": 0,
                **selection_stats,
            }
            self.last_registry_debug = {
                "strategy": selection_stats.get("registry_selection_strategy"),
                "status": selection_stats.get("registry_selection_status"),
                "section_selection": {
                    "selected_sections": (getattr(self, "last_section_debug", {}) or {}).get("selected_registry_sections") or [],
                    "selected_chunks": (getattr(self, "last_section_debug", {}) or {}).get("selected_registry_section_chunks") or [],
                    "selector_raw_preview": (getattr(self, "last_section_debug", {}) or {}).get("registry_selector_raw_preview"),
                    "verifier_raw_preview": (getattr(self, "last_section_debug", {}) or {}).get("registry_verifier_raw_preview"),
                    "selector_error": (getattr(self, "last_section_debug", {}) or {}).get("registry_selector_error"),
                    "verifier_error": (getattr(self, "last_section_debug", {}) or {}).get("registry_verifier_error"),
                },
                "chunks": [],
                "section_debug": getattr(self, "last_section_debug", {}) or {},
                "selection_stats": selection_stats,
                "summary": {
                    "raw_registry_merged": {},
                    "final_registry": {},
                    "registry_raw_count": 0,
                    "registry_final_count": 0,
                },
            }
            return {}
        print(f"  [Registry Step 2] 分为 {len(chunks)} 个分块，分别调用GPT提取...")
        
        # --- Step 2: GPT分块提取 ---
        all_registries = []
        chunk_debugs = []
        validation_stats = {
            "registry_raw_count": 0,
        }
        validation_stats.update(getattr(self, "last_section_chunking_stats", {}) or {})
        validation_stats.update(selection_stats)
        validation_stats["registry_chunks_total"] = len(chunks)
        registry_parallel_results = None
        if self.max_parallel_text_chunks > 1:
            def _extract_registry_chunk(chunk: Dict) -> Tuple[Dict[str, str], Dict]:
                return self.extract_registry_with_gpt_debug(chunk['text'])

            registry_parallel_results = [
                result
                for _, result in self._run_indexed_parallel(
                    chunks,
                    _extract_registry_chunk,
                    max_workers=self.max_parallel_text_chunks,
                )
            ]
        for chunk_index, chunk in enumerate(chunks):
            print(f"    处理分块{chunk['chunk_id']}: 页 {chunk['page_nums']}")
            if registry_parallel_results is None:
                registry, chunk_debug = self.extract_registry_with_gpt_debug(chunk['text'])
            else:
                registry, chunk_debug = registry_parallel_results[chunk_index]
            chunk_debug.update({
                "chunk_id": chunk.get("chunk_id"),
                "page_nums": chunk.get("page_nums") or [],
                "section_title": chunk.get("section_title"),
                "section_start_page": chunk.get("section_start_page"),
                "section_end_page": chunk.get("section_end_page"),
                "chunk_strategy": chunk.get("chunk_strategy"),
                "text_preview": (chunk.get("text") or "")[:1200],
            })
            chunk_debugs.append(chunk_debug)
            validation_stats["registry_raw_count"] += len(chunk_debug.get("raw_registry") or {})
            if registry:
                print(f"      提取到 {len(registry)} 条映射")
                all_registries.append(registry)
            else:
                print(f"      未提取到映射")
        
        # --- Step 3: 合并去重 ---
        print(f"  [Registry Step 3] 合并 {len(all_registries)} 个分块的结果...")
        registry = self._merge_registry_results(all_registries)
        validation_stats["registry_final_count"] = len(registry)
        self.last_registry_validation_stats = validation_stats
        section_debug = getattr(self, "last_section_debug", {}) or {}
        self.last_registry_debug = {
            "strategy": selection_stats.get("registry_selection_strategy"),
            "status": selection_stats.get("registry_selection_status"),
            "section_selection": {
                "selected_sections": section_debug.get("selected_registry_sections") or [],
                "selected_chunks": section_debug.get("selected_registry_section_chunks") or [],
                "selector_raw_preview": section_debug.get("registry_selector_raw_preview"),
                "verifier_raw_preview": section_debug.get("registry_verifier_raw_preview"),
                "selector_error": section_debug.get("registry_selector_error"),
                "verifier_error": section_debug.get("registry_verifier_error"),
            },
            "chunks": chunk_debugs,
            "section_debug": section_debug,
            "selection_stats": selection_stats,
            "summary": {
                "raw_registry_merged": registry,
                "final_registry": registry,
                "registry_raw_count": validation_stats.get("registry_raw_count", 0),
                "registry_final_count": len(registry),
            },
        }

        return registry

    # =====================================================================
    # Per-Chunk 精准过滤
    # =====================================================================

    def _filter_registry_for_chunk(self, chunk_text: str, registry: Dict[str, str]) -> Dict[str, str]:
        """
        只返回 chunk_text 中实际出现的 registry 符号对应的映射。
        未匹配任何符号时返回完整 registry（保守兜底）。
        """
        if not registry:
            return registry

        matched = {}
        for sym in registry:
            if len(sym) < 2:
                continue
            if re.search(re.escape(sym), chunk_text):
                matched[sym] = registry[sym]

        if not matched:
            return registry  # 降级：发送全部

        return matched

    def _filter_gp_for_chunk(self, chunk_text: str,
                             gp_summaries: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        检测 chunk_text 中引用了哪些 GP，只返回被引用 GP 的摘要。
        降级: 单GP直接返回; 多GP无引用则返回空 dict。
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

    def _has_gp_reference_cue(self, chunk_text: str) -> bool:
        return bool(
            re.search(
                r'(?i)\b(?:according\s+to|following)\s+(?:the\s+)?(?:general\s+)?procedures?\b',
                chunk_text,
            )
            or re.search(r'\bGP[\s\-\u2010-\u2015]*[A-Z0-9]+\b', chunk_text)
            or re.search(r'\bGeneralProcedure[A-Z0-9]+\b', chunk_text)
        )

    def _has_generic_gp_reference(self, chunk_text: str) -> bool:
        return bool(re.search(
            r'(?i)\b(?:according\s+to|following)\s+(?:the\s+)?general procedure\b(?!\s+[A-Z0-9]\b)'
            r'|\bfollowing\s+(?:the\s+)?procedures?\b(?!\s+[A-Z0-9]\b)',
            chunk_text,
        ))

    def _build_gp_resolution_candidates(self, gp_texts: Dict[str, str]) -> List[Dict[str, str]]:
        return [
            {
                "key": key,
                "title": str(text).splitlines()[0][:160] if isinstance(text, str) and text.strip() else key,
                "preview": str(text)[:800],
            }
            for key, text in gp_texts.items()
            if isinstance(text, str) and text.strip()
        ]

    def _resolve_generic_gp_for_chunk(self, chunk_text: str, gp_texts: Dict[str, str]) -> Dict[str, str]:
        if not getattr(self, "client", None):
            self.last_gp_selection_debug = {
                "mode": "llm_unavailable_generic",
                "selected_gp_keys": [],
            }
            return {}

        candidates = self._build_gp_resolution_candidates(gp_texts)
        if not candidates:
            return {}

        try:
            response = self.client.chat.completions.create(
                model=self.extract_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You resolve references to chemistry general procedures. Return valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": self.GP_GENERIC_RESOLUTION_PROMPT.format(
                            chunk_text=chunk_text[:5000],
                            gp_candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
                        ),
                    },
                ],
                temperature=0.0,
            )
            parsed = self._parse_compact_json_response(response.choices[0].message.content or "")
            if not parsed:
                return {}
            selected = parsed.get("selected_gp_keys", [])
            if not isinstance(selected, list):
                return {}
            selected_gp = {
                key: gp_texts[key]
                for key in selected
                if key in gp_texts and isinstance(gp_texts.get(key), str)
            }
            self.last_gp_selection_debug = {
                "mode": "llm_resolved_generic",
                "selected_gp_keys": list(selected_gp.keys()),
                "confidence": parsed.get("confidence"),
                "reason": parsed.get("reason"),
            }
            return selected_gp
        except Exception as exc:
            print(f"  [WARN] GP generic resolution failed: {exc}")
            self.last_gp_selection_debug = {
                "mode": "llm_failed_generic",
                "error": str(exc),
            }
            return {}

    def _resolve_no_reference_gp_for_chunk(self, chunk_text: str, gp_texts: Dict[str, str]) -> Dict[str, str]:
        if not getattr(self, "client", None):
            self.last_gp_selection_debug = {
                "mode": "no_reference",
                "selected_gp_keys": [],
            }
            return {}

        candidates = self._build_gp_resolution_candidates(gp_texts)
        if not candidates:
            return {}

        try:
            response = self.client.chat.completions.create(
                model=self.extract_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You resolve whether chemistry general procedures apply to reaction chunks. Return valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": self.GP_NO_REFERENCE_RESOLUTION_PROMPT.format(
                            chunk_text=chunk_text[:5000],
                            gp_candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
                        ),
                    },
                ],
                temperature=0.0,
            )
            parsed = self._parse_compact_json_response(response.choices[0].message.content or "")
            if not parsed:
                self.last_gp_selection_debug = {
                    "mode": "llm_unresolved_no_reference",
                    "selected_gp_keys": [],
                    "reason": "empty_or_invalid_json",
                }
                return {}
            selected = parsed.get("selected_gp_keys", [])
            if not isinstance(selected, list):
                self.last_gp_selection_debug = {
                    "mode": "llm_unresolved_no_reference",
                    "selected_gp_keys": [],
                    "reason": "selected_gp_keys_not_list",
                }
                return {}
            selected_gp = {
                key: gp_texts[key]
                for key in selected
                if key in gp_texts and isinstance(gp_texts.get(key), str)
            }
            if selected_gp:
                self.last_gp_selection_debug = {
                    "mode": "llm_resolved_no_reference",
                    "selected_gp_keys": list(selected_gp.keys()),
                    "confidence": parsed.get("confidence"),
                    "reason": parsed.get("reason"),
                }
                return selected_gp

            self.last_gp_selection_debug = {
                "mode": "llm_unresolved_no_reference",
                "selected_gp_keys": [],
                "confidence": parsed.get("confidence"),
                "reason": parsed.get("reason"),
            }
            return {}
        except Exception as exc:
            print(f"  [WARN] GP no-reference resolution failed: {exc}")
            self.last_gp_selection_debug = {
                "mode": "llm_failed_no_reference",
                "selected_gp_keys": [],
                "error": str(exc),
            }
            return {}

    def select_gp_for_chunk(self, chunk_text: str, gp_texts: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Select only GP texts relevant to a Stage2 chunk."""
        if not gp_texts:
            self.last_gp_selection_debug = {"mode": "none", "selected_gp_keys": []}
            return {}

        valid_gp_texts = {
            key: text for key, text in gp_texts.items()
            if isinstance(text, str) and text.strip()
        }
        if not valid_gp_texts:
            self.last_gp_selection_debug = {"mode": "none", "selected_gp_keys": []}
            return {}

        if len(valid_gp_texts) == 1:
            if self._has_gp_reference_cue(chunk_text):
                self.last_gp_selection_debug = {
                    "mode": "single_gp",
                    "selected_gp_keys": list(valid_gp_texts.keys()),
                }
                return valid_gp_texts
            return self._resolve_no_reference_gp_for_chunk(chunk_text, valid_gp_texts)

        matched = {
            key: text
            for key, text in valid_gp_texts.items()
            if self._text_contains_gp_alias(chunk_text, key)
        }
        if matched:
            self.last_gp_selection_debug = {
                "mode": "explicit_match",
                "selected_gp_keys": list(matched.keys()),
            }
            return matched

        if self._has_generic_gp_reference(chunk_text):
            selected = self._resolve_generic_gp_for_chunk(chunk_text, valid_gp_texts)
            if selected:
                return selected
            if not getattr(self, "last_gp_selection_debug", {}).get("mode", "").startswith("llm_"):
                self.last_gp_selection_debug = {
                    "mode": "generic_unresolved",
                    "selected_gp_keys": [],
            }
            return {}

        return self._resolve_no_reference_gp_for_chunk(chunk_text, valid_gp_texts)

    # 以下方法已弃用，现在使用GPT分块提取Registry
    # def _find_registry_pages(self, pages: List[Dict]) -> List[Dict]:
    #     """找出包含 symbol→name 模式的页面"""
    #     ...
    # def _extract_registry_regex(self, text: str) -> Dict[str, str]:
    #     """用正则从文本中提取 symbol→name 映射"""
    #     ...

    def extract_registry_with_gpt_debug(self, text: str) -> Tuple[Dict[str, str], Dict]:
        """
        用 gpt-5-mini 从文本中提取 symbol→name 映射（正则的fallback）

        Args:
            text: 要分析的文本（通常是PDF前N页）

        Returns:
            {"1a": "full_name", "2a": "..."}
        """
        try:
            # 截断过长文本，只取前15000字符（registry通常在前十几页）
            if len(text) > 15000:
                text = text[:15000]

            messages = [
                {"role": "system",
                 "content": "You are a chemistry data extractor. Always respond with compact valid JSON only."},
                {"role": "user",
                 "content": f"{self.REGISTRY_PROMPT}\n\n{text}"}
            ]
            response = self.client.chat.completions.create(
                model=self.screen_model,
                messages=messages,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            if not raw:
                print("  [WARN] GPT registry extraction returned empty content")
            raw_registry = self._parse_registry_response(raw)
            debug = {
                "raw_response": raw,
                "raw_registry": raw_registry,
                "counts": {
                    "raw": len(raw_registry),
                },
                "error": None,
            }
            return raw_registry, debug
        except Exception as e:
            print(f"  [WARN] GPT registry提取出错: {e}")
            debug = {
                "raw_response": "",
                "raw_registry": {},
                "counts": {
                    "raw": 0,
                },
                "error": str(e),
            }
            return {}, debug

    def extract_registry_with_gpt(self, text: str) -> Dict[str, str]:
        registry, debug = self.extract_registry_with_gpt_debug(text)
        self._last_registry_chunk_debug = debug
        return registry

    def _parse_registry_response(self, raw: str) -> Dict[str, str]:
        """解析GPT返回的registry JSON，不做抽取后清洗。"""
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
                result = {}
                for k, v in data.items():
                    sym = str(k).strip()
                    name = str(v).strip()
                    if not sym or not name:
                        continue
                    result[sym] = name
                return result
        except json.JSONDecodeError:
            pass
        return {}

    # =====================================================================
    # Stage -1: General Procedure 文本提取
    # =====================================================================

    def _make_unique_gp_key(self, base_key: str, used_keys: set) -> str:
        """Return a stable unique GP key without merging distinct records."""
        key = (base_key or "GeneralProcedureUnlabeled").strip()
        if key not in used_keys:
            used_keys.add(key)
            return key

        suffix = 2
        while f"{key}_{suffix}" in used_keys:
            suffix += 1
        unique_key = f"{key}_{suffix}"
        used_keys.add(unique_key)
        return unique_key

    def _semantic_gp_key_from_text(self, title: str, text_sample: str = "") -> Optional[str]:
        """Build a descriptive key for unlabeled GP headings."""
        source = (title or "").strip()
        generic_title = bool(re.match(r'(?i)^\s*general\s+procedures?\s*$', source))
        if generic_title and text_sample:
            source = str(text_sample)[:320]

        source = re.sub(r'---\s*Page\s+\d+\s*---', ' ', source, flags=re.IGNORECASE)
        source = re.sub(r'\bS\d+\b', ' ', source)
        source = source.replace('\u2010', '-').replace('\u2011', '-').replace('\u2012', '-')
        source = source.replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')
        source = re.split(r'[:\uff1a]', source, maxsplit=1)[0]
        source = re.split(
            r'(?i)\b(?:under|to\s+a|a\s+flask|in\s+a|the\s+reaction|a\s+mixture)\b',
            source,
            maxsplit=1,
        )[0]
        source = re.sub(
            r'(?i)^\s*(?:general\s+procedures?|representative\s+procedures?|typical\s+procedures?|'
            r'standard\s+procedures?|experimental\s+procedures?|procedures?)\b'
            r'\s*(?:for|of|to|on)?\s*',
            '',
            source,
        )
        source = re.sub(r'(?i)^\s*(?:the|a|an)\s+', '', source)
        source = re.sub(r'[^A-Za-z0-9]+', ' ', source).strip().lower()
        words = [word for word in source.split() if word]
        if len(words) < 2:
            return None

        slug = "_".join(words[:12])
        slug = slug[:96].strip("_")
        if not slug or len(slug) < 8:
            return None
        return f"GeneralProcedure_{slug}"

    def _make_gp_key(self, title: str, gp_counter: Dict[str, int], text_sample: str = "") -> str:
        """
        根据 GP 标题生成 key
        例: 'General procedure for synthesis of 1-38:' → 'GeneralProcedure_1-38'
            'General Procedure A' → 'GeneralProcedureA'
            无标识时 → semantic key 或 'GeneralProcedureUnlabeled1', ...
        """
        # 尝试提取 scope (数字范围)
        if self._is_short_gp_title(title):
            return self._normalize_short_gp_title_key(title)

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

        semantic_key = self._semantic_gp_key_from_text(title, text_sample)
        if semantic_key:
            return semantic_key

        # Unlabeled GPs must not consume A/B/C keys used by explicit titles.
        if 'GeneralProcedureUnlabeled' not in gp_counter:
            gp_counter['GeneralProcedureUnlabeled'] = 0
        gp_counter['GeneralProcedureUnlabeled'] += 1
        return f"GeneralProcedureUnlabeled{gp_counter['GeneralProcedureUnlabeled']}"

    def _is_explicit_gp_title(self, title: str) -> bool:
        """Return True for line-start GP headings with an explicit label and colon."""
        return bool(re.match(self.EXPLICIT_GP_TITLE_PATTERN, title.strip()))

    def _is_short_gp_title(self, title: str) -> bool:
        """Return True for short GP headings such as GP-1: or Gp-A:."""
        return bool(re.match(self.SHORT_GP_TITLE_PATTERN, title.strip()))

    def _normalize_short_gp_title_key(self, title: str) -> str:
        """Preserve short GP labels as public keys while normalizing separators."""
        match = re.match(
            r'(?i)^\s*(?:\d+[\).]\s*)?(GP)(?:[\s\-\u2010-\u2015]*)([A-Z0-9]+)\s*(?::|\uff1a|\.)',
            title.strip(),
        )
        if not match:
            return title.strip().rstrip(':：.').strip()

        prefix = match.group(1)
        label = match.group(2)
        return f"{prefix}-{label}"

    def _normalize_gp_title_key(self, title: str) -> str:
        """Normalize explicit GP headings to human-readable keys."""
        match = re.match(
            r'(?i)^\s*(?:\d+[\).]\s*)?'
            r'(general\s+procedure|representative\s+procedure|typical\s+procedure|'
            r'standard\s+procedure|standard\s+conditions|experimental\s+procedure|procedure)'
            r'\s+([A-Z0-9]+)\b\s*(?:\([^)\n]{0,60}\))?\s*(?::|\uff1a)',
            title.strip(),
        )
        if not match:
            return title.strip().rstrip(':：').strip()

        label = match.group(2).upper()
        return f"GeneralProcedure{label}"

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

        short_match = re.match(r'(?i)^(GP)[\s\-\u2010-\u2015]*([A-Z0-9]+)$', gp_key)
        if short_match:
            prefix = short_match.group(1)
            label = short_match.group(2)
            aliases.update({
                f"{prefix} {label}",
                f"{prefix}-{label}",
                f"GP {label}",
                f"GP-{label}",
            })

        return sorted(aliases, key=len, reverse=True)

    def _text_contains_gp_alias(self, text: str, gp_key: str) -> bool:
        for alias in self._gp_key_aliases(gp_key):
            parts = alias.split()
            separator = r'[\s\-\u2010-\u2015]+' if len(parts) > 1 else r'\s+'
            pattern = r'(?i)\b' + separator.join(re.escape(part) for part in parts) + r'\b'
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

    def _parse_compact_json_response(self, raw: str) -> Optional[Dict[str, Any]]:
        raw = (raw or "").strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                return None

    def _find_anchor_end(self, text: str, anchor: str) -> Optional[int]:
        anchor = re.sub(r'\s+', ' ', (anchor or "").strip())
        if len(anchor) < 20:
            return None
        exact = text.find(anchor)
        if exact >= 0:
            return exact + len(anchor)
        parts = [part for part in re.split(r'\s+', anchor) if part]
        if not parts:
            return None
        pattern = r'\s+'.join(re.escape(part) for part in parts)
        match = re.search(pattern, text)
        return match.end() if match else None

    def _llm_trim_gp_record(self, record: Dict[str, Any], candidate_char_limit: int = GP_CONTEXT_CHAR_LIMIT) -> Dict[str, Any]:
        if not getattr(self, "client", None):
            record["final_text"] = record["raw_text"][:record["max_gp_chars"]]
            record["stored_chars"] = len(record["final_text"])
            record["end_reason"] = "llm_failed_fallback"
            record["llm_trim"] = {"error": "missing_client"}
            return record

        candidate_text = record["raw_text"][:candidate_char_limit]
        try:
            response = self.client.chat.completions.create(
                model=self.extract_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a chemistry SI text boundary detector. Return valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": self.GP_TRUNCATION_PROMPT.format(
                            gp_key=record["key"],
                            gp_title=record["title"],
                            candidate_text=candidate_text,
                        ),
                    },
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content or ""
            parsed = self._parse_compact_json_response(raw)
            if not parsed:
                raise ValueError("empty_or_invalid_json")

            anchor = str(parsed.get("end_anchor") or "").strip()
            anchor_end = self._find_anchor_end(candidate_text, anchor)
            if anchor_end is None:
                raise ValueError("anchor_not_found")

            include_anchor = parsed.get("include_anchor", True)
            final_end = anchor_end if include_anchor else max(0, anchor_end - len(anchor))
            final_text = candidate_text[:final_end].strip()[:GP_CONTEXT_CHAR_LIMIT]
            if len(final_text) < 80:
                raise ValueError("trim_too_short")

            record["final_text"] = final_text
            record["stored_chars"] = len(final_text)
            record["end_reason"] = "llm_trimmed"
            record["llm_trim"] = {
                "end_anchor": anchor,
                "include_anchor": bool(include_anchor),
                "trim_reason": parsed.get("trim_reason"),
                "confidence": parsed.get("confidence"),
            }
            return record
        except Exception as exc:
            record["final_text"] = record["raw_text"][:record["max_gp_chars"]]
            record["stored_chars"] = len(record["final_text"])
            record["end_reason"] = "llm_failed_fallback"
            record["llm_trim"] = {"error": str(exc)}
            return record

    def extract_general_procedure_records(self, pages: List[Dict]) -> List[Dict[str, Any]]:
        """Extract GP records with boundary metadata. Public outputs still use gp_texts."""
        full_text = self._build_gp_full_text(pages)
        filtered = self._find_filtered_gp_titles(full_text)
        if not filtered:
            self.last_gp_records = []
            return []

        gp_counter = {}
        used_gp_keys = set()
        max_gp_chars = GP_CONTEXT_CHAR_LIMIT
        rule_complete_trust_chars = GP_CONTEXT_CHAR_LIMIT
        records: List[Dict[str, Any]] = []

        for i, (pos, title) in enumerate(filtered):
            has_next_gp = i + 1 < len(filtered)
            end_pos = filtered[i + 1][0] if has_next_gp else len(full_text)
            next_title = filtered[i + 1][1] if has_next_gp else None
            raw_text = full_text[pos:end_pos].strip()
            raw_chars = len(raw_text)
            key = self._make_unique_gp_key(
                self._make_gp_key(title, gp_counter, raw_text),
                used_gp_keys,
            )

            if has_next_gp and raw_chars <= rule_complete_trust_chars:
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

            record: Dict[str, Any] = {
                "key": key,
                "title": title,
                "raw_text": raw_text,
                "final_text": raw_text[:max_gp_chars],
                "start_pos": pos,
                "raw_end_pos": end_pos,
                "has_next_gp": has_next_gp,
                "next_gp_title": next_title,
                "raw_chars_to_next_or_eof": raw_chars,
                "stored_chars": min(raw_chars, max_gp_chars),
                "max_gp_chars": max_gp_chars,
                "rule_complete_trust_chars": rule_complete_trust_chars,
                "end_reason": end_reason,
                "pre_llm_end_reason": end_reason,
                "needs_llm_truncation": needs_llm,
            }
            if needs_llm:
                record = self._llm_trim_gp_record(record)
            records.append(record)

        self.last_gp_records = records
        return records

    def extract_general_procedure_texts(self, pages: List[Dict]) -> Dict[str, str]:
        """Extract General Procedure text and return the existing public dict schema."""
        records = self.extract_general_procedure_records(pages)
        if not records:
            return {}

        gp_texts: Dict[str, str] = {}
        used_gp_keys = set()
        for record in records:
            key = self._make_unique_gp_key(str(record["key"]), used_gp_keys)
            text = record.get("final_text") or record.get("raw_text", "")
            gp_texts[key] = str(text)[:GP_CONTEXT_CHAR_LIMIT]

        split_texts = self._split_embedded_procedure_scopes(gp_texts)
        capped_texts: Dict[str, str] = {}
        used_final_keys = set()
        for key, text in split_texts.items():
            unique_key = self._make_unique_gp_key(str(key), used_final_keys)
            capped_texts[unique_key] = str(text)[:GP_CONTEXT_CHAR_LIMIT]
        return capped_texts

    def _split_embedded_procedure_scopes(self, gp_texts: Dict[str, str]) -> Dict[str, str]:
        """Split one General Procedure section into scoped Procedure A/B entries."""
        split_texts = {}
        used_split_keys = set()
        heading_re = re.compile(
            r'(?:^|\n)\s*(?:\d+\)\s*)?Procedure\s+([A-Z])\s*'
            r'\((?:for\s+)?([^)]+?\b(?:products?|compounds?)\s+'
            r'([A-Za-z]?\d+[A-Za-z]*)\s*[-\u2010-\u2015]\s*([A-Za-z]?\d+[A-Za-z]*))\)',
            re.IGNORECASE,
        )

        for base_key, text in gp_texts.items():
            matches = list(heading_re.finditer(text))
            if len(matches) < 2:
                split_texts[self._make_unique_gp_key(base_key, used_split_keys)] = text
                continue

            for idx, match in enumerate(matches):
                letter = match.group(1).upper()
                lo = match.group(3)
                hi = match.group(4)
                start = match.start()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                scoped_text = text[start:end].strip()
                split_key = f"GeneralProcedure{letter}_{lo}-{hi}"
                split_texts[self._make_unique_gp_key(split_key, used_split_keys)] = scoped_text

        return split_texts

    # =====================================================================
    # Stage -1b: GP总结 (LLM提取结构化信息)
    # =====================================================================

    def summarize_gp_texts(self, gp_texts: Dict[str, str]) -> Dict[str, Dict]:
        """用LLM总结GP文本，提取结构化信息
        
        Args:
            gp_texts: {"GeneralProcedureA": "GP原文...", ...}
            
        Returns:
            {"GeneralProcedureA": {"substrates": [...], "catalysts": [...], ...}, ...}
        """
        if not gp_texts:
            return {}
        
        summarized = {}
        
        for gp_label, gp_text in gp_texts.items():
            print(f"    总结GP: {gp_label}")
            
            # 截断过长的GP文本
            text_to_process = gp_text[:2000] if len(gp_text) > 2000 else gp_text
            
            try:
                response = self.client.chat.completions.create(
                    model=self.extract_model,
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
        """解析GPT返回的GP总结JSON"""
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
    # 第三层：页面分块
    # =====================================================================

    def chunk_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        将页面按pages_per_chunk分块

        Returns:
            [{"chunk_id": 1, "page_range": "1-8", "text": "...", "pages": [1,2,...]}, ...]
        """
        chunks = []
        for i in range(0, len(pages), self.pages_per_chunk):
            chunk_pages = pages[i:i + self.pages_per_chunk]
            page_nums = [p['page_num'] for p in chunk_pages]
            chunk_text = "\n\n".join(
                f"--- Page {p['page_num']} ---\n{p['text']}"
                for p in chunk_pages
            )
            page_range = f"{page_nums[0]}-{page_nums[-1]}" if len(page_nums) > 1 else str(page_nums[0])
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "page_range": page_range,
                "page_nums": page_nums,
                "text": chunk_text,
                "approx_tokens": len(chunk_text) // 4,  # 粗估token数
            })
        return chunks

    # =====================================================================
    # 第四层：Stage1 - gpt-5-mini 快速筛选
    # =====================================================================

    def stage1_screen(self, chunk_text: str, chunk_label: str) -> Dict:
        """
        用便宜模型快速判断分块是否包含反应数据

        Returns:
            {"has_reactions": bool, "relevant_pages": [int]}
        """
        try:
            response = self.client.chat.completions.create(
                model=self.screen_model,
                messages=[
                    {"role": "system", "content": "You are a chemistry data screener. Always respond with valid JSON only."},
                    {"role": "user", "content": f"{self.SCREEN_PROMPT}\n\n{chunk_text}"}
                ],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_screen_response(raw)
        except Exception as e:
            print(f"  [WARN] Stage1筛选出错 ({chunk_label}): {e}")
            # 筛选出错时保守处理：假设相关，让Stage2处理
            return {"has_reactions": True, "relevant_pages": [], "_error": True}

    def _parse_screen_response(self, raw: str) -> Dict:
        """解析Stage1筛选响应"""
        raw = raw.strip()
        # 去掉markdown代码块
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            data = json.loads(raw)
            return {
                "has_reactions": data.get("has_reactions", False),
                "relevant_pages": data.get("relevant_pages", []),
            }
        except json.JSONDecodeError:
            # 尝试用正则提取
            if re.search(r'"has_reactions"\s*:\s*true', raw, re.IGNORECASE):
                return {"has_reactions": True, "relevant_pages": []}
            return {"has_reactions": False, "relevant_pages": []}

    # =====================================================================
    # 第五层：Stage2 - gpt-5-mini 精确提取
    # =====================================================================

    def _empty_metric_value(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.lower() in {
                "null",
                "none",
                "not reported",
                "not specified",
                "n/a",
                "na",
            }:
                return None
            return stripped
        return value

    def _source_page_numbers_from_chunk_text(self, chunk_text: str) -> List[int]:
        """Return 1-based PDF pages explicitly marked in a Stage2 chunk."""
        pages = set()
        for match in re.findall(r"--- Page\s+(\d+)\s+---", str(chunk_text or "")):
            page = int(match)
            if page > 0:
                pages.add(page)
        return sorted(pages)

    def _normalize_source_pages(
        self,
        value,
        allowed_page_nums: Optional[List[int]] = None,
    ) -> List[int]:
        """Normalize model page provenance and optionally keep only current chunk pages."""
        if isinstance(value, (list, tuple, set)):
            values = list(value)
        elif value in (None, ""):
            values = []
        else:
            values = [value]

        pages = set()
        for item in values:
            if isinstance(item, bool) or item is None:
                continue
            if isinstance(item, int):
                candidates = [item]
            elif isinstance(item, float) and item.is_integer():
                candidates = [int(item)]
            else:
                candidates = [
                    int(match)
                    for match in re.findall(r"(?i)(?:\bpage\s*)?(\d+)\b", str(item))
                ]
            pages.update(page for page in candidates if page > 0)

        if allowed_page_nums is not None:
            allowed = {
                int(page)
                for page in allowed_page_nums
                if isinstance(page, int) and not isinstance(page, bool) and page > 0
            }
            pages.intersection_update(allowed)

        return sorted(pages)

    def _multistep_review_needed(self, text: str) -> bool:
        """Weak signal that a GP may need semantic multi-step review.

        This does not decide the schema. It only flags text that should prompt
        the model to check whether multiple chemical transformations are
        present, while avoiding pure workup/purification descriptions.
        """
        normalized = re.sub(r"\s+", " ", str(text or "").casefold())
        if not normalized:
            return False

        if re.search(r"\b(?:over|in)\s+(?:two|three|\d+)\s+steps?\b", normalized):
            return True
        if re.search(r"\b(?:two|three|\d+)[-\s]?step\s+(?:sequence|procedure|synthesis|preparation)\b", normalized):
            return True
        if re.search(r"\bused\s+directly\s+in\s+the\s+next\s+step\b", normalized):
            return True
        if re.search(r"\b(?:without\s+(?:further\s+)?(?:isolation|purification))\b", normalized):
            return True
        if re.search(
            r"\b(?:crude\s+(?:product|material)|residue|material)\b.{0,140}"
            r"\b(?:subjected|treated|dissolved|used|carried|converted|added)\b",
            normalized,
        ):
            return True

        chemical_action = re.search(
            r"\b(?:deprotect(?:ed|ion)?|desilylat(?:ed|ion)?|hydrolys(?:ed|is)|"
            r"reduc(?:ed|tion)|oxid(?:ized|ised|ation)|cycli[sz](?:ed|ation)|"
            r"coupl(?:ed|ing)|converted|functionalized|functionalised)\b",
            normalized,
        )
        sequence_marker = re.search(r"\b(?:then|subsequently|after completion|followed by)\b", normalized)
        workup_only = re.fullmatch(
            r".*\b(?:quenched|extracted|washed|dried|concentrated|filtered|purified|chromatography)\b.*",
            normalized,
        ) and not chemical_action
        return bool(chemical_action and sequence_marker and not workup_only)

    def _normalize_step_number(self, value) -> Optional[int]:
        """Normalize model step labels such as 1, "1", or "Step 1"."""
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float) and value.is_integer():
            number = int(value)
            return number if number > 0 else None
        match = re.fullmatch(r"(?i)\s*(?:step\s*)?(\d+)\s*", str(value))
        if not match:
            return None
        number = int(match.group(1))
        return number if number > 0 else None

    def _single_step_schema(self, reaction: Dict) -> Dict:
        """Remove accidental step annotations so ordinary reactions stay compatible."""
        reaction.pop("step_count", None)
        for field in ("substrates", "products", "catalysts", "additives", "reagents"):
            values = reaction.get(field)
            if not isinstance(values, list):
                continue
            for item in values:
                if isinstance(item, dict):
                    item.pop("step", None)

        conditions = reaction.get("conditions")
        if isinstance(conditions, dict):
            for key, value in list(conditions.items()):
                if not isinstance(value, list):
                    continue
                step_values = [
                    item.get("value")
                    for item in value
                    if isinstance(item, dict)
                    and self._normalize_step_number(item.get("step")) == 1
                    and item.get("value") not in (None, "")
                ]
                if step_values:
                    conditions[key] = "; ".join(str(item) for item in step_values)
                elif not value:
                    conditions[key] = None

        if reaction.get("intermediates") == []:
            reaction.pop("intermediates", None)
        return reaction

    def _collect_step_annotations(self, reaction: Dict) -> List[int]:
        """Return normalized step annotations already present in a reaction object."""
        steps: List[int] = []
        for field in ("substrates", "products", "catalysts", "additives", "reagents"):
            values = reaction.get(field)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                step = self._normalize_step_number(item.get("step"))
                if step is not None:
                    steps.append(step)

        conditions = reaction.get("conditions")
        if isinstance(conditions, dict):
            for value in conditions.values():
                if not isinstance(value, list):
                    continue
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    step = self._normalize_step_number(item.get("step"))
                    if step is not None:
                        steps.append(step)

        intermediates = reaction.get("intermediates")
        if isinstance(intermediates, list):
            for item in intermediates:
                if not isinstance(item, dict):
                    continue
                for key in ("produced_in_step", "consumed_in_step"):
                    step = self._normalize_step_number(item.get(key))
                    if step is not None:
                        steps.append(step)
        return steps

    def _infer_or_clean_step_schema(self, reaction: Dict) -> Dict:
        """Make model-produced step annotations consistent with the schema.

        The model sometimes emits item/condition-level step annotations but omits
        the top-level step_count. Treat any step >= 2 as a true multi-step schema
        and infer step_count deterministically. If all annotations are only step 1,
        treat them as accidental single-step pollution and remove them.
        """
        if reaction.get("step_count") is not None:
            return reaction

        steps = self._collect_step_annotations(reaction)
        if not steps:
            return reaction
        max_step = max(steps)
        if max_step >= 2:
            reaction["step_count"] = max_step
            return reaction
        return self._single_step_schema(reaction)

    def _is_explicit_intermediate(self, item: Dict) -> bool:
        """Require an explicit name or symbol; reject generic residue descriptions."""
        symbol = str(item.get("symbol") or item.get("label") or "").strip()
        generic = (
            r"(?i)(?:the\s+)?(?:corresponding\s+)?(?:crude\s+)?"
            r"(?:product|residue|intermediate)(?:\s+"
            r"(?:thus\s+obtained|obtained\s+above|from\s+step\s+\d+))?"
        )
        if symbol and not re.fullmatch(generic, symbol):
            return True
        name = str(item.get("name") or "").strip()
        return bool(name) and not bool(re.fullmatch(generic, name))

    def _multistep_identity_key(self, item: Dict) -> str:
        symbol = str(item.get("symbol") or item.get("label") or "").strip().casefold()
        if symbol:
            return f"symbol:{symbol}"
        name = str(item.get("name") or "").strip().casefold()
        return f"name:{name}" if name else ""

    def _normalize_multistep_schema(self, reaction: Dict) -> Dict:
        """Canonicalize and validate the optional multi-step reaction extension."""
        reaction = self._infer_or_clean_step_schema(reaction)
        raw_step_count = reaction.get("step_count")
        if raw_step_count is None:
            return reaction

        step_count = self._normalize_step_number(raw_step_count)
        if step_count is None:
            raise ValueError(f"invalid step_count: {raw_step_count!r}")
        if step_count < 2:
            return self._single_step_schema(reaction)
        reaction["step_count"] = step_count

        compound_fields = ("substrates", "products", "catalysts", "additives", "reagents")
        for field in compound_fields:
            values = reaction.get(field)
            if values is None:
                reaction[field] = []
                continue
            if not isinstance(values, list):
                raise ValueError(f"multi-step {field} must be a list")
            for item in values:
                if not isinstance(item, dict):
                    raise ValueError(f"multi-step {field} items must be objects")
                step = self._normalize_step_number(item.get("step"))
                if step is None or step > step_count:
                    raise ValueError(f"multi-step {field} item has invalid step: {item!r}")
                item["step"] = step

        conditions = reaction.get("conditions")
        if conditions is None:
            conditions = {}
            reaction["conditions"] = conditions
        if not isinstance(conditions, dict):
            raise ValueError("multi-step conditions must be an object")
        for key, value in list(conditions.items()):
            if value in (None, ""):
                conditions[key] = []
                continue
            if not isinstance(value, list):
                conditions[key] = [{"step": 1, "value": value}]
                continue
            normalized_values = []
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError(f"multi-step condition {key!r} item must be an object")
                step = self._normalize_step_number(item.get("step"))
                if step is None or step > step_count:
                    raise ValueError(f"multi-step condition {key!r} has invalid step: {item!r}")
                condition_value = item.get("value")
                if condition_value in (None, ""):
                    continue
                normalized_values.append({"step": step, "value": condition_value})
            conditions[key] = normalized_values

        raw_intermediates = reaction.get("intermediates")
        if raw_intermediates is None:
            raw_intermediates = []
        if not isinstance(raw_intermediates, list):
            raise ValueError("multi-step intermediates must be a list")
        intermediates = []
        for item in raw_intermediates:
            if not isinstance(item, dict) or not self._is_explicit_intermediate(item):
                continue
            produced = self._normalize_step_number(item.get("produced_in_step"))
            consumed = self._normalize_step_number(item.get("consumed_in_step"))
            if produced is None or consumed is None or produced >= consumed or consumed > step_count:
                raise ValueError(f"intermediate has invalid step relation: {item!r}")
            normalized = dict(item)
            normalized["produced_in_step"] = produced
            normalized["consumed_in_step"] = consumed
            intermediates.append(normalized)
        reaction["intermediates"] = intermediates

        intermediate_keys = {
            self._multistep_identity_key(item)
            for item in intermediates
            if self._multistep_identity_key(item)
        }
        for field in ("substrates", "products"):
            duplicate_keys = {
                self._multistep_identity_key(item)
                for item in reaction.get(field, [])
                if self._multistep_identity_key(item) in intermediate_keys
            }
            if duplicate_keys:
                raise ValueError(
                    f"intermediates must not be duplicated in {field}: {sorted(duplicate_keys)}"
                )
        return reaction

    def sanitize_reaction_schema(
        self,
        reaction: Dict,
        allowed_page_nums: Optional[List[int]] = None,
    ) -> Dict:
        """Keep only the current LangGraph reaction schema surface."""
        if not isinstance(reaction, dict):
            return reaction

        reaction = self._normalize_multistep_schema(reaction)
        reaction["source_pages"] = self._normalize_source_pages(
            reaction.get("source_pages"),
            allowed_page_nums=allowed_page_nums,
        )

        for key in ("dr", "conversion", "selectivity", "NMR_yield", "GC_yield"):
            reaction.pop(key, None)

        targets = reaction.get("targets")
        if not isinstance(targets, dict):
            targets = reaction.get("target") if isinstance(reaction.get("target"), dict) else {}
        reaction.pop("target", None)

        clean_targets = {}
        for key in ("yield", "ee", "er"):
            clean_targets[key] = self._empty_metric_value(
                targets.get(key) if targets.get(key) is not None else reaction.get(key)
            )
            reaction.pop(key, None)
        reaction["targets"] = clean_targets
        return reaction

    def sanitize_reactions_schema(
        self,
        reactions: List[Dict],
        allowed_page_nums: Optional[List[int]] = None,
    ) -> List[Dict]:
        return [
            self.sanitize_reaction_schema(reaction, allowed_page_nums=allowed_page_nums)
            for reaction in reactions
            if isinstance(reaction, dict)
        ]

    def stage2_audit_missing_reactions(
        self,
        chunk_text: str,
        chunk_label: str,
        current_reactions: List[Dict],
        gp_block: str = "",
    ) -> List[Dict]:
        """Ask the LLM to extract only reactions omitted from the first Stage2 pass."""
        if not self.enable_stage2_audit:
            return []

        try:
            allowed_page_nums = self._source_page_numbers_from_chunk_text(chunk_text)
            current_json = json.dumps(current_reactions, ensure_ascii=False, indent=2)
            response = self.client.chat.completions.create(
                model=self.extract_model,
                messages=[
                    {"role": "system", "content": self.EXTRACTION_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Processing chunk: {chunk_label}\n"
                            f"{self.STAGE2_AUDIT_PROMPT}\n"
                            f"{gp_block}"
                            f"\ncurrent_extraction:\n{current_json}\n"
                            f"\nsource_text:\n{chunk_text}"
                        ),
                    },
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            if not raw:
                return []
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            data = self._parse_json_with_trailing_text(raw.strip())
            if isinstance(data, dict):
                missing = data.get("missing_reactions", data.get("reactions", []))
            elif isinstance(data, list):
                missing = data
            else:
                missing = []
            if not isinstance(missing, list):
                return []
            return self.sanitize_reactions_schema(
                missing,
                allowed_page_nums=allowed_page_nums,
            )
        except Exception as e:
            print(f"  [WARN] Stage2 audit failed ({chunk_label}): {e}")
            return []

    def stage2_extract(self, chunk_text: str, chunk_label: str,
                       registry: Optional[Dict[str, str]] = None,
                       gp_texts: Optional[Dict[str, str]] = None) -> List[Dict]:
        result = self.stage2_extract_with_meta(
            chunk_text,
            chunk_label,
            registry=registry,
            gp_texts=gp_texts,
        )
        self.stage2_audit_recovered = int(
            getattr(self, "stage2_audit_recovered", 0) or 0
        ) + int(result.get("audit_recovered") or 0)
        return result.get("reactions") or []

    def stage2_extract_with_meta(self, chunk_text: str, chunk_label: str,
                                 registry: Optional[Dict[str, str]] = None,
                                 gp_texts: Optional[Dict[str, str]] = None) -> Dict:
        """
        用精确模型从分块中提取反应数据

        Args:
            chunk_text: 分块文本
            chunk_label: 分块标签（用于日志）
            registry: symbol→name 映射表（可选）
            gp_texts: 原始GP文本 dict（可选），格式: {"GP标签": "GP原文..."}
        """
        # 构建 registry 注入块
        # 构建 GP 注入块：只注入当前 chunk 明确引用或 LLM 解析到的 GP
        allowed_page_nums = self._source_page_numbers_from_chunk_text(chunk_text)
        gp_block = ""
        if not hasattr(self, "_gp_selection_lock"):
            self._gp_selection_lock = threading.Lock()
        with self._gp_selection_lock:
            selected_gp_texts = self.select_gp_for_chunk(chunk_text, gp_texts)
            gp_selection_debug = dict(getattr(self, "last_gp_selection_debug", {}) or {})
        if selected_gp_texts:
            gp_entries = []
            for label, text in selected_gp_texts.items():
                if isinstance(text, str):
                    # 截断过长的GP文本
                    text_truncated = (
                        text[: GP_CONTEXT_CHAR_LIMIT - 3] + "..."
                        if len(text) > GP_CONTEXT_CHAR_LIMIT
                        else text
                    )
                    entry = f"""=== {label} ===
{text_truncated}"""
                    gp_entries.append(entry)
            
            if gp_entries:
                gp_block = self.GP_INJECTION_TEMPLATE.format(
                    gp_block="\n\n".join(gp_entries)
                )
                if self._multistep_review_needed("\n\n".join(gp_entries)):
                    gp_block += (
                        "\nMULTI-STEP REVIEW NOTE:\n"
                        "The supplied GP may describe sequential chemical transformations. "
                        "Review semantically whether there are multiple chemical transformations "
                        "or only one reaction followed by workup/purification. Use the multi-step "
                        "schema only for true multi-step chemistry; if the GP is multi-step, every "
                        "GP-referenced concrete entry must inherit that schema.\n"
                    )

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.extract_model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.EXTRACTION_PROMPT
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Processing chunk: {chunk_label}\n"
                                f"Extract ALL qualifying reactions from the text below. Do not omit any.\n"
                                "If a full chemical name is not explicitly present in this text, "
                                "keep the short label/code in symbol and do not invent a full name. "
                                "Registry-based name completion is handled after extraction.\n"
                                f"{gp_block}"
                                f"\nText content:\n{chunk_text}"
                            )
                        }
                    ],
                    temperature=0.1,
                )
                gpt_response = response.choices[0].message.content or ""
                if not gpt_response.strip():
                    print(f"  [WARN] Stage2 attempt {attempt+1}/3: empty response ({chunk_label})")
                    continue
                reactions = self.sanitize_reactions_schema(
                    self.parse_and_validate_json(gpt_response),
                    allowed_page_nums=allowed_page_nums,
                )
                missing = self.stage2_audit_missing_reactions(
                    chunk_text,
                    chunk_label,
                    reactions,
                    gp_block=gp_block,
                )
                if missing:
                    print(f"  [Stage2 audit] recovered {len(missing)} omitted reactions ({chunk_label})")
                    return {
                        "reactions": self.merge_results([reactions, missing]),
                        "audit_recovered": len(missing),
                        "gp_selection_debug": gp_selection_debug,
                        "error": None,
                    }
                return {
                    "reactions": reactions,
                    "audit_recovered": 0,
                    "gp_selection_debug": gp_selection_debug,
                    "error": None,
                }
            except Exception as e:
                print(f"  [WARN] Stage2 attempt {attempt+1}/3 failed ({chunk_label}): {e}")

        print(f"  [ERROR] Stage2 最终失败 ({chunk_label})")
        return {
            "reactions": [],
            "audit_recovered": 0,
            "gp_selection_debug": gp_selection_debug,
            "error": "stage2_failed_after_retries",
        }

    # =====================================================================
    # 结果合并与去重
    # =====================================================================

    def merge_results(self, all_reactions: List[List[Dict]]) -> List[Dict]:
        """
        合并多个分块的提取结果，去重

        去重策略：
        1. 按 (substrates, products, targets) 内容签名去重
        2. merge 后按最终顺序为同一 id prefix 重新编号
        """
        merged = []
        signature_indexes = {}

        for chunk_reactions in all_reactions:
            if not chunk_reactions:
                continue
            for reaction in chunk_reactions:
                if not isinstance(reaction, dict):
                    continue

                normalized_pages = self._normalize_source_pages(reaction.get("source_pages"))
                sig = self._reaction_signature(reaction)
                if sig in signature_indexes:
                    existing = merged[signature_indexes[sig]]
                    existing["source_pages"] = sorted(set(
                        self._normalize_source_pages(existing.get("source_pages"))
                        + normalized_pages
                    ))
                    continue
                new_reaction = dict(reaction)
                new_reaction["source_pages"] = normalized_pages
                signature_indexes[sig] = len(merged)
                merged.append(new_reaction)

        return self._renumber_reaction_ids(merged)

    def _reaction_id_prefix(self, reaction_id: str) -> str:
        """Return the stable id prefix used for final sequential numbering."""
        rid = str(reaction_id or "").strip()
        if not rid:
            return "Reaction"
        match = re.match(r"^(.*?)-Entry\d+$", rid)
        if match:
            prefix = match.group(1).strip()
            return prefix or "Reaction"
        if "-Entry" in rid:
            prefix = rid.split("-Entry", 1)[0].strip()
            return prefix or "Reaction"
        return rid or "Reaction"

    def _renumber_reaction_ids(self, reactions: List[Dict]) -> List[Dict]:
        """Make final reaction ids unique and sequential within each original prefix."""
        counters = {}
        renumbered = []
        for reaction in reactions:
            if not isinstance(reaction, dict):
                continue
            new_reaction = dict(reaction)
            original_id = str(new_reaction.get("id") or "").strip()
            if original_id and not new_reaction.get("_original_id"):
                new_reaction["_original_id"] = original_id
            prefix = self._reaction_id_prefix(original_id)
            counters[prefix] = counters.get(prefix, 0) + 1
            new_reaction["id"] = f"{prefix}-Entry{counters[prefix]}"
            renumbered.append(new_reaction)
        return renumbered

    def _reaction_signature(self, reaction: Dict) -> str:
        """生成反应的唯一签名用于去重"""
        subs = json.dumps(reaction.get('substrates', ''), sort_keys=True, ensure_ascii=False)
        prods = json.dumps(reaction.get('products', ''), sort_keys=True, ensure_ascii=False)
        target = json.dumps(
            reaction.get('targets', reaction.get('target', '')),
            sort_keys=True,
            ensure_ascii=False,
        )
        intermediates = json.dumps(
            reaction.get('intermediates', ''),
            sort_keys=True,
            ensure_ascii=False,
        )
        step_count = reaction.get('step_count', '')
        return f"{subs}|{prods}|{intermediates}|{step_count}|{target}"

    # =====================================================================
    # Stage 5b: GP 条件来源标记（兜底）
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
            if any(v not in (None, "", [], {}) for v in conditions.values()):
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
    # 过滤无效反应
    # =====================================================================

    def _is_placeholder_name(self, name: str) -> bool:
        """检测是否为泛指产物名（如 Compound 3, Product 13）"""
        if not name or not isinstance(name, str):
            return False
        patterns = [
            r'^[Cc]ompound\s+\d+$',
            r'^[Cc]ompound\s+[A-Z]\d*$',
            r'^[Pp]roduct\s+\d+$',
            r'^[Pp]roduct\s+[A-Z]\d*$',
            r'^[Ss]ubstrate\s+\d+$',
            r'^[Ee]ntry\s+\d+$',
            r'^[Ss]ample\s+\d+$',
            r'^[Mm]aterial\s+\d+$',
            r'^not specified$',
            r'^n/?a$',
            r'^unknown$',
            r'^unnamed$',
        ]
        return any(re.match(p, name.strip(), re.IGNORECASE) for p in patterns)

    def _get_ee_value(self, reaction: Dict):
        """从反应数据中提取ee值，兼容多种JSON结构"""
        ee = reaction.get('ee')
        if ee is None:
            target = reaction.get('targets') or reaction.get('target', {})
            if isinstance(target, dict):
                ee = target.get('ee')
        return ee

    def _get_dr_value(self, reaction: Dict):
        """从反应数据中提取dr值，兼容多种JSON结构"""
        dr = reaction.get('dr')
        if dr is None:
            target = reaction.get('targets') or reaction.get('target', {})
            if isinstance(target, dict):
                dr = target.get('dr')
        return dr

    def _get_er_value(self, reaction: Dict):
        """从反应数据中提取er值，兼容多种JSON结构"""
        er = reaction.get('er')
        if er is None:
            target = reaction.get('targets') or reaction.get('target', {})
            if isinstance(target, dict):
                er = target.get('er')
        return er

    def _get_yield_value(self, reaction: Dict):
        """从反应数据中提取yield值，兼容多种JSON结构"""
        y = reaction.get('yield')
        if y is None:
            y = reaction.get('yield_percent')
        if y is None:
            target = reaction.get('targets') or reaction.get('target', {})
            if isinstance(target, dict):
                y = target.get('yield')
                if y is None:
                    y = target.get('yield_percent')
        return y

    def _is_valid_selectivity(self, value) -> bool:
        """判断选择性值是否有效（非null、非空字符串、非not reported）"""
        if value is None:
            return False
        if isinstance(value, str) and value.strip().lower() in ('null', 'none', 'not reported', 'n/a', ''):
            return False
        return True

    def filter_reactions(self, reactions: List[Dict]) -> List[Dict]:
        """
        过滤无效反应：
        1. 产物为泛指名称（如 Compound 3, Product 13）→ 丢弃
        2. yield 为空 → 丢弃
        3. yield、ee、er 都为空 → 丢弃
        4. 至少有一个有效 yield、ee 或 er → 保留
        """
        filtered = []
        filter_reasons = {"placeholder_name": 0, "no_yield_ee_er": 0}

        for reaction in reactions:
            if not isinstance(reaction, dict):
                continue

            # 检查产物名称是否为泛指
            products = reaction.get('products', [])
            has_placeholder = False
            if isinstance(products, list):
                for p in products:
                    if isinstance(p, dict) and self._is_placeholder_name(p.get('name', '')):
                        has_placeholder = True
                        break
            elif isinstance(products, str) and self._is_placeholder_name(products):
                has_placeholder = True

            # 产物为泛指 → 直接丢弃
            if has_placeholder:
                filter_reasons["placeholder_name"] += 1
                continue

            yield_val = self._get_yield_value(reaction)
            ee = self._get_ee_value(reaction)
            er = self._get_er_value(reaction)
            has_target = (
                self._is_valid_selectivity(yield_val)
                or self._is_valid_selectivity(ee)
                or self._is_valid_selectivity(er)
            )

            if not has_target:
                filter_reasons["no_yield_ee_er"] += 1
                continue

            filtered.append(reaction)

        total_removed = sum(filter_reasons.values())
        if total_removed > 0:
            reasons_str = ", ".join(f"{k}: {v}" for k, v in filter_reasons.items() if v > 0)
            print(f"  [Filter] 过滤掉 {total_removed} 条无效反应 ({reasons_str})")

        return filtered

    # =====================================================================
    # Stage 5: 后处理对齐 — 用 registry 补全符号
    # =====================================================================

    def process_pages_with_context(self, pdf_path: str, pages: List[Dict],
                                   entity_context: Dict,
                                   output_dir: Path,
                                   output_path: Optional[Path] = None) -> Optional[List[Dict]]:
        """Run reaction extraction using precomputed entity context."""
        pdf_name = Path(pdf_path).name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) if output_path else output_dir / f"{Path(pdf_path).stem}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        self.stage2_audit_recovered = 0

        context_path_value = entity_context.get("_context_path")
        if context_path_value:
            chunk_log_dir = Path(context_path_value).parent.parent / "chunk_extraction_logs"
        else:
            chunk_log_dir = output_file.parent.parent / "intermediate" / "chunk_extraction_logs"
        chunk_log_path = chunk_log_dir / f"{output_file.stem}.json"
        chunk_extraction_log = []

        file_stats = {
            'total_pages': len(pages),
            'filtered_pages': 0,
            'total_chunks': 0,
            'screened_pass': 0,
            'screened_fail': 0,
            'registry_size': 0,
            'gp_templates': 0,
            'stage2_audit_enabled': self.enable_stage2_audit,
            'stage2_audit_recovered': 0,
        }

        def _as_int_list(values) -> List[int]:
            nums = []
            for value in values or []:
                try:
                    nums.append(int(value))
                except (TypeError, ValueError):
                    continue
            return nums

        def _chunk_log_entry(chunk: Dict) -> Dict:
            return {
                "chunk_id": chunk.get("chunk_id"),
                "page_range": chunk.get("page_range"),
                "page_nums": _as_int_list(chunk.get("page_nums")),
                "section_title": chunk.get("section_title"),
                "section_part_index": chunk.get("section_part_index") or chunk.get("part_index"),
                "approx_tokens": chunk.get("approx_tokens"),
                "stage1": {
                    "has_reactions": False,
                    "relevant_pages": [],
                    "reason": "",
                },
                "trimmed": {
                    "was_trimmed": False,
                    "page_range": chunk.get("page_range"),
                    "page_nums": _as_int_list(chunk.get("page_nums")),
                    "pages_removed": [],
                },
                "stage2": {
                    "attempted": False,
                    "reaction_count": 0,
                    "reaction_ids": [],
                },
            }

        def _atomic_write_json(path: Path, payload: Dict) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tmp_path.replace(path)

        def _write_chunk_log() -> None:
            if not chunk_extraction_log and not chunks:
                return
            payload = {
                "source": str(pdf_path),
                "created_at": datetime.now().isoformat(),
                "schema": "chunk_extraction_log_v1",
                "stats": {
                    "total_chunks": len(chunks),
                    "stage1_pass": sum(
                        1 for item in chunk_extraction_log
                        if item.get("stage1", {}).get("has_reactions")
                    ),
                    "stage1_fail": sum(
                        1 for item in chunk_extraction_log
                        if not item.get("stage1", {}).get("has_reactions")
                    ),
                    "stage2_attempted_chunks": sum(
                        1 for item in chunk_extraction_log
                        if item.get("stage2", {}).get("attempted")
                    ),
                    "stage2_total_reactions_before_merge": sum(
                        int(item.get("stage2", {}).get("reaction_count") or 0)
                        for item in chunk_extraction_log
                    ),
                },
                "chunks": chunk_extraction_log,
            }
            _atomic_write_json(chunk_log_path, payload)

        def _build_output_payload(reactions: List[Dict], is_partial: bool, completed_stage2_chunks: int) -> Dict:
            stats_snapshot = dict(file_stats)
            stats_snapshot['stage2_audit_recovered'] = self.stage2_audit_recovered
            return {
                "source": str(pdf_path),
                "extracted_at": datetime.now().isoformat(),
                "is_partial": bool(is_partial),
                "completed_stage2_chunks": completed_stage2_chunks,
                "total_reactions": len(reactions),
                "name_registry": registry,
                "general_procedures": {
                    k: v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v
                    for k, v in gp_texts.items()
                } if gp_texts else {},
                "entity_context_path": entity_context.get("_context_path"),
                "chunk_extraction_log_path": str(chunk_log_path),
                "stats": stats_snapshot,
                "reactions": reactions,
                "metadata": entity_context.get("metadata"),
            }

        registry = entity_context.get("name_registry") or entity_context.get("symbol_name_mapping") or {}
        gp_texts = entity_context.get("general_procedures") or {}
        file_stats['registry_size'] = len(registry)
        file_stats['gp_templates'] = len(gp_texts)

        if not pages:
            print("  [SKIP] no page text available")
            return None

        print("[ReactionExtractionAgent] filtering relevant pages...")
        relevant_pages = self.filter_relevant_pages(pages)
        file_stats['filtered_pages'] = len(relevant_pages)
        if not relevant_pages:
            print("  [SKIP] no chemistry-related pages")
            return None

        print("[ReactionExtractionAgent] chunking pages...")
        chunks = self.chunk_pages_by_sections(relevant_pages, pages)
        file_stats['total_chunks'] = len(chunks)
        section_stats = getattr(self, "last_section_chunking_stats", {}) or {}
        file_stats.update({
            "section_chunking_enabled": section_stats.get("section_chunking_enabled", False),
            "section_chunking_source": section_stats.get("section_chunking_source", "fixed_fallback"),
            "section_scope": section_stats.get("section_scope", "full_document"),
            "raw_section_count": section_stats.get("raw_section_count", 0),
            "section_chunk_count": section_stats.get("section_chunk_count", 0),
            "section_chunk_max_pages": section_stats.get("section_chunk_max_pages", 5),
            "section_chunk_max_chars": section_stats.get("section_chunk_max_chars", 8000),
            "section_count": section_stats.get("section_count", 0),
            "reaction_chunk_strategy": section_stats.get("reaction_chunk_strategy", "fixed_pages"),
        })

        print(f"[ReactionExtractionAgent] Stage1 screening with {self.screen_model}...")
        def _screen_chunk(chunk: Dict) -> Dict:
            label = f"{pdf_name} Chunk{chunk['chunk_id']} Pages[{chunk['page_range']}]"
            return self.stage1_screen(chunk['text'], label)

        screen_results = [
            result
            for _, result in self._run_indexed_parallel(
                chunks,
                _screen_chunk,
                max_workers=self.max_parallel_text_chunks,
            )
        ]
        chunk_extraction_log = [_chunk_log_entry(chunk) for chunk in chunks]
        chunk_log_by_id = {
            item.get("chunk_id"): item
            for item in chunk_extraction_log
        }

        relevant_chunks = []
        stage1_trimmed_chunks = 0
        stage1_trimmed_pages_removed = 0
        for chunk, screen in zip(chunks, screen_results):
            log_item = chunk_log_by_id.get(chunk.get("chunk_id"))
            if log_item is not None:
                log_item["stage1"] = {
                    "has_reactions": bool(screen.get("has_reactions")),
                    "relevant_pages": _as_int_list(screen.get("relevant_pages")),
                    "reason": str(screen.get("reason") or ""),
                }
            if not screen.get('has_reactions'):
                continue
            trimmed_chunk = self._trim_chunk_to_stage1_pages(chunk, screen)
            original_pages = chunk.get("page_nums") or []
            trimmed_pages = trimmed_chunk.get("page_nums") or []
            original_page_nums = _as_int_list(original_pages)
            trimmed_page_nums = _as_int_list(trimmed_pages)
            if len(trimmed_pages) < len(original_pages):
                stage1_trimmed_chunks += 1
                stage1_trimmed_pages_removed += len(original_pages) - len(trimmed_pages)
            if log_item is not None:
                log_item["trimmed"] = {
                    "was_trimmed": set(trimmed_page_nums) != set(original_page_nums),
                    "page_range": trimmed_chunk.get("page_range"),
                    "page_nums": trimmed_page_nums,
                    "pages_removed": [
                        page_num for page_num in original_page_nums
                        if page_num not in set(trimmed_page_nums)
                    ],
                }
            relevant_chunks.append((trimmed_chunk, screen))
        file_stats['screened_pass'] = len(relevant_chunks)
        file_stats['screened_fail'] = len(chunks) - len(relevant_chunks)
        file_stats['stage1_trimmed_chunks'] = stage1_trimmed_chunks
        file_stats['stage1_trimmed_pages_removed'] = stage1_trimmed_pages_removed
        if not relevant_chunks:
            print("  [SKIP] Stage1 found no reaction chunks")
            _write_chunk_log()
            return None

        print(f"[ReactionExtractionAgent] Stage2 extraction with {self.extract_model}...")
        all_chunk_results = []
        if self.max_parallel_text_chunks > 1:
            def _extract_reaction_chunk(item: Tuple[Dict, Dict]) -> Dict:
                chunk, _screen = item
                label = f"{pdf_name} Chunk{chunk['chunk_id']} Pages[{chunk['page_range']}]"
                try:
                    result = self.stage2_extract_with_meta(
                        chunk['text'],
                        label,
                        gp_texts=gp_texts,
                    )
                except Exception as exc:
                    result = {
                        "reactions": [],
                        "audit_recovered": 0,
                        "gp_selection_debug": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                return result

            stage2_results = [
                result
                for _, result in self._run_indexed_parallel(
                    relevant_chunks,
                    _extract_reaction_chunk,
                    max_workers=self.max_parallel_text_chunks,
                )
            ]
            for (chunk, _screen), result in zip(relevant_chunks, stage2_results):
                reactions = result.get("reactions") or []
                self.stage2_audit_recovered += int(result.get("audit_recovered") or 0)
                log_item = chunk_log_by_id.get(chunk.get("chunk_id"))
                if log_item is not None:
                    log_item["stage2"] = {
                        "attempted": True,
                        "reaction_count": len(reactions),
                        "reaction_ids": [
                            str(reaction.get("id"))
                            for reaction in reactions
                            if isinstance(reaction, dict) and reaction.get("id")
                        ],
                    }
                    if result.get("error"):
                        log_item["stage2"]["error"] = result.get("error")
                    if result.get("gp_selection_debug"):
                        log_item["stage2"]["gp_selection"] = result.get("gp_selection_debug")
                all_chunk_results.append(reactions)
            _write_chunk_log()
        else:
            for chunk, _screen in relevant_chunks:
                label = f"{pdf_name} Chunk{chunk['chunk_id']} Pages[{chunk['page_range']}]"
                reactions = self.stage2_extract(
                    chunk['text'],
                    label,
                    gp_texts=gp_texts,
                )
                log_item = chunk_log_by_id.get(chunk.get("chunk_id"))
                if log_item is not None:
                    log_item["stage2"] = {
                        "attempted": True,
                        "reaction_count": len(reactions),
                        "reaction_ids": [
                            str(reaction.get("id"))
                            for reaction in reactions
                            if isinstance(reaction, dict) and reaction.get("id")
                        ],
                    }
                all_chunk_results.append(reactions)
                _write_chunk_log()
                partial_merged = self.sanitize_reactions_schema(self.merge_results(all_chunk_results))
                _atomic_write_json(
                    output_file,
                    _build_output_payload(
                        partial_merged,
                        is_partial=True,
                        completed_stage2_chunks=len(all_chunk_results),
                    ),
                )

        merged = self.merge_results(all_chunk_results)
        if registry and merged:
            merged = self.align_names_in_reactions(merged, registry)
            registry_stats = getattr(self, "last_registry_resolution_stats", {}) or {}
            file_stats['registry_resolved_count'] = registry_stats.get("resolved", 0)
            file_stats['registry_verified_count'] = registry_stats.get("verified", 0)
            file_stats['registry_conflict_count'] = registry_stats.get("conflicts", 0)
        else:
            file_stats['registry_resolved_count'] = 0
            file_stats['registry_verified_count'] = 0
            file_stats['registry_conflict_count'] = 0
        scaffold_mapping = entity_context.get("scaffold_substituent_mapping") or {}
        if scaffold_mapping and merged:
            merged = self.enrich_reactions_with_scaffold_mapping(merged, scaffold_mapping)
        if gp_texts and merged:
            merged = self.apply_gp_conditions_fallback(merged, gp_texts)
        merged = self.sanitize_reactions_schema(merged)
        file_stats['stage2_audit_recovered'] = self.stage2_audit_recovered
        _write_chunk_log()

        output_data = _build_output_payload(
            merged,
            is_partial=False,
            completed_stage2_chunks=len(all_chunk_results),
        )
        _atomic_write_json(output_file, output_data)

        print(f"  saved reaction output: {output_file}")
        return merged

    def enrich_reactions_with_scaffold_mapping(self, reactions: List[Dict],
                                               scaffold_mapping: Dict[str, Dict]) -> List[Dict]:
        """Attach precomputed scaffold/substituent data to reaction compounds."""
        if not scaffold_mapping:
            return reactions

        enriched = []
        for reaction in reactions:
            if not isinstance(reaction, dict):
                enriched.append(reaction)
                continue

            new_reaction = dict(reaction)
            for field in ["substrates", "products"]:
                items = new_reaction.get(field)
                if not isinstance(items, list):
                    continue

                new_items = []
                for item in items:
                    if not isinstance(item, dict):
                        new_items.append(item)
                        continue

                    new_item = dict(item)
                    keys = [
                        str(new_item.get("symbol") or ""),
                        str(new_item.get("name") or ""),
                    ]
                    match = next((scaffold_mapping[k] for k in keys if k in scaffold_mapping), None)
                    if match:
                        new_item.setdefault("parseable", match.get("parseable", False))
                        if match.get("scaffold"):
                            new_item.setdefault("scaffold", match.get("scaffold"))
                        if match.get("substituents"):
                            new_item.setdefault("substituents", match.get("substituents"))
                    new_items.append(new_item)
                new_reaction[field] = new_items
            enriched.append(new_reaction)
        return enriched

    def _registry_lookup(self, value: str, registry: Dict[str, str], normalized_registry: Dict[str, str]) -> Optional[Tuple[str, str]]:
        symbol = str(value or "").strip()
        if not symbol:
            return None
        if symbol in registry:
            return symbol, registry[symbol]
        normalized = symbol.casefold()
        if normalized in normalized_registry:
            matched_symbol = normalized_registry[normalized]
            return matched_symbol, registry[matched_symbol]
        return None

    def _generic_label_symbol(self, name: str, registry: Dict[str, str], normalized_registry: Dict[str, str]) -> str:
        text = str(name or "").strip()
        if not text:
            return ""
        generic_prefixes = (
            "substrate",
            "product",
            "compound",
            "ligand",
            "catalyst",
            "reagent",
            "additive",
            "alkene",
            "alkyne",
            "aldehyde",
            "ketone",
            "imine",
            "ester",
            "amide",
            "acid",
        )
        match = re.fullmatch(
            rf"(?i)(?:{'|'.join(generic_prefixes)})\s+([A-Za-z]{{0,4}}[-]?\d+[A-Za-z]{{0,4}}|\d+[A-Za-z]{{1,4}})",
            text,
        )
        if match and self._registry_lookup(match.group(1), registry, normalized_registry):
            return match.group(1)
        return ""

    def _normalize_registry_compare_name(self, value: str) -> str:
        text = str(value or "").strip().casefold()
        text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _should_registry_overwrite_name(
        self,
        name: str,
        symbol: str,
        registry: Dict[str, str],
        normalized_registry: Dict[str, str],
    ) -> bool:
        name = str(name or "").strip()
        symbol = str(symbol or "").strip()
        if not name:
            return True
        if symbol and name.casefold() == symbol.casefold():
            return True
        if self._registry_lookup(name, registry, normalized_registry):
            return True
        if symbol and self._generic_label_symbol(name, registry, normalized_registry):
            return True
        return False

    def _resolve_registry_compound(
        self,
        item,
        registry: Dict[str, str],
        normalized_registry: Dict[str, str],
    ) -> Tuple[object, str]:
        if isinstance(item, str):
            lookup = self._registry_lookup(item, registry, normalized_registry)
            if not lookup:
                return item, "none"
            symbol, name = lookup
            return {
                "name": name,
                "symbol": symbol,
                "resolution_source": "name_registry",
                "resolution_method": "same_paper_symbol",
            }, "resolved"

        if not isinstance(item, dict):
            return item, "none"

        new_item = dict(item)
        raw_name = str(new_item.get("name") or "").strip()
        raw_symbol = str(new_item.get("symbol") or new_item.get("label") or "").strip()
        lookup = self._registry_lookup(raw_symbol, registry, normalized_registry) if raw_symbol else None
        if not lookup:
            generic_symbol = self._generic_label_symbol(raw_name, registry, normalized_registry)
            lookup = self._registry_lookup(generic_symbol, registry, normalized_registry) if generic_symbol else None
        if not lookup:
            lookup = self._registry_lookup(raw_name, registry, normalized_registry)
        if not lookup:
            return new_item, "none"

        symbol, name = lookup
        if self._should_registry_overwrite_name(raw_name, symbol, registry, normalized_registry):
            new_item["name"] = name
            new_item["symbol"] = raw_symbol or symbol
            new_item["resolution_source"] = "name_registry"
            new_item["resolution_method"] = "same_paper_symbol"
            return new_item, "resolved"

        new_item["registry_name"] = name
        if self._normalize_registry_compare_name(raw_name) == self._normalize_registry_compare_name(name):
            new_item["registry_match_status"] = "verified"
            return new_item, "verified"

        new_item["registry_match_status"] = "conflict"
        new_item["registry_conflict_reason"] = "extracted_name_differs_from_registry_symbol_name"
        return new_item, "conflict"

    def align_names_in_reactions(self, reactions: List[Dict],
                                 registry: Dict[str, str]) -> List[Dict]:
        """Complete label-only reaction compounds from the name registry."""
        if not registry:
            self.last_registry_resolution_stats = {"resolved": 0, "verified": 0, "conflicts": 0}
            return reactions

        normalized_registry = {
            str(symbol).strip().casefold(): str(symbol).strip()
            for symbol in registry
            if str(symbol).strip()
        }
        aligned = []
        stats = {"resolved": 0, "verified": 0, "conflicts": 0}
        fields = ['substrates', 'products', 'intermediates', 'catalysts', 'additives', 'reagents']
        for reaction in reactions:
            if not isinstance(reaction, dict):
                aligned.append(reaction)
                continue

            new_reaction = dict(reaction)
            for field in fields:
                items = new_reaction.get(field)
                if not items:
                    continue

                if isinstance(items, list):
                    new_items = []
                    for item in items:
                        resolved_item, status = self._resolve_registry_compound(
                            item,
                            registry,
                            normalized_registry,
                        )
                        if status == "resolved":
                            stats["resolved"] += 1
                        elif status == "verified":
                            stats["verified"] += 1
                        elif status == "conflict":
                            stats["conflicts"] += 1
                        new_items.append(resolved_item)
                    new_reaction[field] = new_items
                elif isinstance(items, str):
                    resolved_item, status = self._resolve_registry_compound(
                        items,
                        registry,
                        normalized_registry,
                    )
                    if status == "resolved":
                        new_reaction[field] = resolved_item
                        stats["resolved"] += 1

            aligned.append(new_reaction)

        self.last_registry_resolution_stats = stats
        if any(stats.values()):
            print(
                "  [Registry] "
                f"resolved={stats['resolved']}, "
                f"verified={stats['verified']}, "
                f"conflicts={stats['conflicts']}"
            )
        return aligned

    # =====================================================================
    # 单文件处理
    # =====================================================================

    def process_single_pdf(self, pdf_path: str, output_dir: Path) -> Optional[List[Dict]]:
        """
        处理单个PDF文件，完整的策略流水线

        流程: 提取文本 → Stage0 Name Registry(全页扫描) → Stage-1 GP文本提取 → 关键词过滤 → 分块
              → Stage1筛选 → Stage2提取(注入registry+GP) → 合并去重
              → Stage5 符号对齐 → Stage5b GP来源标记 → 保存
        """
        pdf_name = Path(pdf_path).name
        print(f"\n{'='*60}")
        print(f"处理: {pdf_name}")
        print(f"{'='*60}")

        # 重置统计
        self.stage2_audit_recovered = 0
        file_stats = {'total_pages': 0, 'filtered_pages': 0,
                      'total_chunks': 0, 'screened_pass': 0, 'screened_fail': 0,
                      'registry_size': 0, 'gp_templates': 0,
                      'stage2_audit_enabled': self.enable_stage2_audit,
                      'stage2_audit_recovered': 0}

        try:
            # --- Step 1: 按页提取文本 ---
            print("[Step 1] 按页提取文本...")
            pages = self.extract_text_by_pages(pdf_path)
            file_stats['total_pages'] = len(pages)
            print(f"  提取到 {len(pages)} 页文本")

            if not pages:
                print("  [SKIP] PDF无文本内容（可能是纯图像PDF）")
                return None

            # --- Stage 0: 提取 Name Registry (GPT分块提取) ---
            print("[Stage 0] 提取 symbol→name 注册表 (GPT分块)...")
            registry = self.extract_name_registry(
                pages, 
                max_scan_pages=60,
                pages_per_chunk=5
            )
            file_stats['registry_size'] = len(registry)
            file_stats.update({
                "registry_raw_count": self.last_registry_validation_stats.get("registry_raw_count", len(registry)),
                "registry_final_count": self.last_registry_validation_stats.get("registry_final_count", len(registry)),
            })
            if registry:
                print(f"  注册表: {len(registry)} 条映射")
                for sym, name in list(registry.items())[:5]:
                    display_name = name[:50] + "..." if len(name) > 50 else name
                    print(f"    {sym} → {display_name}")
                if len(registry) > 5:
                    print(f"    ... 还有 {len(registry) - 5} 条")
            else:
                print("  [WARN] 未提取到名称映射，将使用默认提取（可能丢失完整名称）")

            # --- Stage -1: 提取 General Procedure 文本 ---
            print("[Stage -1] 提取 General Procedure 文本...")
            gp_texts = self.extract_general_procedure_texts(pages)
            if gp_texts:
                print(f"  找到 {len(gp_texts)} 个 GP 段落")
                file_stats['gp_templates'] = len(gp_texts)
                
                # --- Stage -1b: GP总结 (已禁用，直接注入原始GP文本) ---
                # print("[Stage -1b] 总结GP文本...")
                # gp_summaries = self.summarize_gp_texts(gp_texts)
                # if gp_summaries:
                #     print(f"  成功总结 {len(gp_summaries)} 个 GP")
                # else:
                #     print("  [WARN] GP总结失败")
                gp_summaries = {}
            else:
                print("  [INFO] 未找到 General Procedure 段落")
                gp_texts = {}
                gp_summaries = {}

            # --- Step 2: 关键词过滤 ---
            print("[Step 2] 关键词过滤...")
            relevant_pages = self.filter_relevant_pages(pages)
            file_stats['filtered_pages'] = len(relevant_pages)
            print(f"  过滤后: {len(relevant_pages)}/{len(pages)} 页保留 "
                  f"(节省约 {(1 - len(relevant_pages)/len(pages))*100:.0f}% 输入)")
            # 输出保留的页码，方便诊断
            relevant_page_nums = [p['page_num'] for p in relevant_pages]
            print(f"  保留页码: {relevant_page_nums}")

            if not relevant_pages:
                print("  [SKIP] 无化学相关页面")
                return None

            # --- Step 3: 分块 ---
            print("[Step 3] 页面分块...")
            chunks = self.chunk_pages_by_sections(relevant_pages, pages)
            file_stats['total_chunks'] = len(chunks)
            section_stats = getattr(self, "last_section_chunking_stats", {}) or {}
            file_stats.update({
                "section_chunking_enabled": section_stats.get("section_chunking_enabled", False),
                "section_chunking_source": section_stats.get("section_chunking_source", "fixed_fallback"),
                "section_scope": section_stats.get("section_scope", "full_document"),
                "raw_section_count": section_stats.get("raw_section_count", 0),
                "section_chunk_count": section_stats.get("section_chunk_count", 0),
                "section_chunk_max_pages": section_stats.get("section_chunk_max_pages", 5),
                "section_chunk_max_chars": section_stats.get("section_chunk_max_chars", 8000),
                "section_count": section_stats.get("section_count", 0),
                "reaction_chunk_strategy": section_stats.get("reaction_chunk_strategy", "fixed_pages"),
            })
            total_approx_tokens = sum(c['approx_tokens'] for c in chunks)
            print(f"  分为 {len(chunks)} 个块, 每块 {self.pages_per_chunk} 页")
            print(f"  预估总文本: ~{total_approx_tokens} tokens")

            # --- Step 4: Stage1 筛选 ---
            print(f"[Step 4] Stage1筛选 (模型: {self.screen_model})...")
            screen_results = []
            for chunk in chunks:
                label = f"{pdf_name} Chunk{chunk['chunk_id']} Pages[{chunk['page_range']}]"
                result = self.stage1_screen(chunk['text'], label)
                screen_results.append(result)
                status = "[YES] has reactions" if result['has_reactions'] else "[NO] no reactions"
                print(f"  块{chunk['chunk_id']} [{chunk['page_range']}]: {status}")

            # 统计通过筛选的块
            relevant_chunks = []
            stage1_trimmed_chunks = 0
            stage1_trimmed_pages_removed = 0
            for chunk, screen in zip(chunks, screen_results):
                if not screen['has_reactions']:
                    continue
                trimmed_chunk = self._trim_chunk_to_stage1_pages(chunk, screen)
                original_pages = chunk.get("page_nums") or []
                trimmed_pages = trimmed_chunk.get("page_nums") or []
                if len(trimmed_pages) < len(original_pages):
                    stage1_trimmed_chunks += 1
                    stage1_trimmed_pages_removed += len(original_pages) - len(trimmed_pages)
                relevant_chunks.append((trimmed_chunk, screen))
            file_stats['screened_pass'] = len(relevant_chunks)
            file_stats['screened_fail'] = len(chunks) - len(relevant_chunks)
            file_stats['stage1_trimmed_chunks'] = stage1_trimmed_chunks
            file_stats['stage1_trimmed_pages_removed'] = stage1_trimmed_pages_removed
            print(f"  筛选结果: {len(relevant_chunks)}/{len(chunks)} 个块通过")
            print(f"  Stage1 token消耗: ~{sum(c['approx_tokens'] for c in chunks)} "
                  f"(比直接全量节省 {(1 - len(relevant_chunks)/max(len(chunks),1))*100:.0f}%)")

            if not relevant_chunks:
                print("  [SKIP] Stage1未发现任何包含反应数据的分块")
                return None

            # --- Step 5: Stage2 提取（注入 registry + 全部GP原文） ---
            print(f"[Step 5] Stage2提取 (模型: {self.extract_model}, {len(relevant_chunks)}个块)...")
            all_chunk_results = []
            for chunk, screen in relevant_chunks:
                label = f"{pdf_name} Chunk{chunk['chunk_id']} Pages[{chunk['page_range']}]"

                # Per-chunk 精准过滤 (Registry)，GP全量注入

                print(f"  块{chunk['chunk_id']}: "
                      f"registry resolver after extraction, "
                      f"GP 全量注入 {len(gp_texts)} 个")

                # 传入原始 gp_texts（全部GP），不过滤
                reactions = self.stage2_extract(chunk['text'], label, gp_texts=gp_texts)
                all_chunk_results.append(reactions)
                # 统计每个chunk的反应类型
                ids = [r.get('id', '') for r in reactions if isinstance(r, dict)]
                gp_count = sum(1 for i in ids if 'General' in str(i))
                rp_count = sum(1 for i in ids if 'Racemic' in str(i))
                other_count = len(ids) - gp_count - rp_count
                print(f"  块{chunk['chunk_id']}: 提取到 {len(reactions)} 条反应 "
                      f"(GeneralProcedure: {gp_count}, RacemicProcedure: {rp_count}, Other: {other_count})")

            # --- Step 6: 合并去重 ---
            print("[Step 6] 合并去重...")
            merged = self.merge_results(all_chunk_results)
            # 统计合并后的反应类型和数据完整性
            gp_merged = sum(1 for r in merged if isinstance(r, dict) and 'General' in str(r.get('id', '')))
            rp_merged = sum(1 for r in merged if isinstance(r, dict) and 'Racemic' in str(r.get('id', '')))
            with_ee = sum(1 for r in merged if isinstance(r, dict) and self._is_valid_selectivity(self._get_ee_value(r)))
            with_er = sum(1 for r in merged if isinstance(r, dict) and self._is_valid_selectivity(self._get_er_value(r)))
            with_yield = sum(1 for r in merged if isinstance(r, dict) and self._is_valid_selectivity(self._get_yield_value(r)))
            print(f"  合并后: {len(merged)} 条反应数据")
            print(f"    GeneralProcedure: {gp_merged}, RacemicProcedure: {rp_merged}")
            print(f"    有yield: {with_yield}, 有ee: {with_ee}, 有er: {with_er}")

            # --- Stage 5: 后处理符号对齐 ---
            if registry and merged:
                print("[Stage 5] 符号对齐（用registry补全剩余符号）...")
                merged = self.align_names_in_reactions(merged, registry)
                registry_stats = getattr(self, "last_registry_resolution_stats", {}) or {}
                file_stats['registry_resolved_count'] = registry_stats.get("resolved", 0)
                file_stats['registry_verified_count'] = registry_stats.get("verified", 0)
                file_stats['registry_conflict_count'] = registry_stats.get("conflicts", 0)
            else:
                file_stats['registry_resolved_count'] = 0
                file_stats['registry_verified_count'] = 0
                file_stats['registry_conflict_count'] = 0

            # --- Stage 5b: GP 条件来源标记 ---
            if gp_texts and merged:
                print("[Stage 5b] GP 条件来源标记（兜底）...")
                merged = self.apply_gp_conditions_fallback(merged, gp_texts)
            merged = self.sanitize_reactions_schema(merged)
            file_stats['stage2_audit_recovered'] = self.stage2_audit_recovered

            # 保存结果
            output_file = output_dir / f"{Path(pdf_path).stem}.json"
            output_data = {
                "source": str(pdf_path),
                "extracted_at": datetime.now().isoformat(),
                "total_reactions": len(merged),
                "name_registry": registry,
                "general_procedures": {k: v[:300] + "..." if len(v) > 300 else v for k, v in gp_texts.items()} if gp_texts else {},
                "stats": file_stats,
                "reactions": merged,
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

            print(f"  保存到: {output_file}")
            print(f"  共提取 {len(merged)} 条反应数据")

            return merged

        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    # =====================================================================
    # 批量处理
    # =====================================================================

    def batch_process(self, si_folder: str, output_dir: str) -> Dict:
        """
        批量处理SI文件夹中的所有PDF

        Args:
            si_folder: SI PDF文件夹路径
            output_dir: 输出目录路径

        Returns:
            处理统计结果
        """
        si_path = Path(si_folder)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 收集所有PDF文件
        pdf_files = sorted(si_path.glob("*.pdf"))

        if not pdf_files:
            print(f"未找到PDF文件: {si_folder}")
            return {"total": 0, "processed": 0}

        print(f"找到 {len(pdf_files)} 个PDF文件")
        print(f"配置: pages_per_chunk={self.pages_per_chunk}, "
              f"screen_model={self.screen_model}, extract_model={self.extract_model}")
        print(f"输出目录: {out_path}")

        results = {
            "total": len(pdf_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "total_reactions": 0,
            "per_file": {},
        }
        all_reactions = []

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n{'#'*60}")
            print(f"# 进度: {i}/{len(pdf_files)}")
            print(f"{'#'*60}")

            reactions = self.process_single_pdf(str(pdf_file), out_path)

            if reactions is None:
                results["skipped"] += 1
                results["per_file"][pdf_file.name] = {"status": "skipped", "reactions": 0}
            elif reactions is False:
                results["failed"] += 1
                results["per_file"][pdf_file.name] = {"status": "failed", "reactions": 0}
            else:
                results["processed"] += 1
                results["total_reactions"] += len(reactions)
                results["per_file"][pdf_file.name] = {"status": "success", "reactions": len(reactions)}
                all_reactions.extend(reactions)

        # 保存汇总
        summary_file = out_path / "all_reactions.json"
        summary = {
            "extracted_at": datetime.now().isoformat(),
            "total_pdfs": len(pdf_files),
            "processed": results["processed"],
            "skipped": results["skipped"],
            "failed": results["failed"],
            "total_reactions": results["total_reactions"],
            "per_file": results["per_file"],
            "reactions": all_reactions,
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 打印最终统计
        print(f"\n{'='*60}")
        print(f"批量处理完成!")
        print(f"{'='*60}")
        print(f"  总PDF数:      {len(pdf_files)}")
        print(f"  成功处理:     {results['processed']}")
        print(f"  跳过:         {results['skipped']}")
        print(f"  失败:         {results['failed']}")
        print(f"  总反应数:     {results['total_reactions']}")
        print(f"  汇总文件:     {summary_file}")

        return results


# =====================================================================
# 命令行入口
# =====================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SI PDF批量提取工具 - Token节省策略全部组合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置处理supporting_information文件夹
  python batch_si_extractor.py

  # 指定输入输出目录
  python batch_si_extractor.py --si_folder /path/to/si_pdfs --output /path/to/output

  # 调整分块大小和模型
  python batch_si_extractor.py --pages_per_chunk 5 --screen_model gpt-5-mini --extract_model gpt-5-mini

环境变量:
  OPENAI_API_KEY: OpenAI API密钥（必需）

Token节省效果:
  关键词过滤:    节省 30-50%（去掉无关页）
  两阶段筛选:    节省 60-70%（大部分块只用mini模型）
  组合总节省:    约 70-80%
        """
    )
    parser.add_argument("--si_folder", default=None,
                        help="SI PDF文件夹路径 (默认: ../supporting_information)")
    parser.add_argument("--output", default=None,
                        help="输出目录 (默认: ./output)")
    parser.add_argument("--api_key", default=None,
                        help="OpenAI API密钥 (默认从OPENAI_API_KEY环境变量读取)")
    parser.add_argument("--pages_per_chunk", type=int, default=5,
                        help="每个分块的页数 (默认: 5)")
    parser.add_argument("--screen_model", default="gpt-5-mini",
                        help="Stage1筛选模型 (默认: gpt-5-mini)")
    parser.add_argument("--extract_model", default="gpt-5-mini",
                        help="Stage2提取模型 (默认: gpt-5-mini)")

    parser.add_argument("--pdf_text_layout", choices=("single", "two_column", "auto"), default="single",
                        help="PDF text layout: single, two_column, or auto (default: single)")
    parser.add_argument("--pdf_text_x_tolerance", type=float, default=3.0,
                        help="pdfplumber horizontal character tolerance (default: 3.0)")
    parser.add_argument("--pdf_text_y_tolerance", type=float, default=5.0,
                        help="pdfplumber vertical line tolerance (default: 5.0)")

    parser.add_argument("--max_parallel_text_chunks", type=int, default=1,
                        help="Maximum chunk-level concurrency inside each PDF (default: 1)")

    args = parser.parse_args()

    # 获取API密钥
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("错误: 请提供OpenAI API密钥")
        print("  方法1: 设置环境变量 $env:OPENAI_API_KEY='your-api-key'")
        print("  方法2: 使用参数 --api_key YOUR_KEY")
        return

    # 确定路径
    script_dir = Path(__file__).resolve().parent
    si_folder = args.si_folder or str(script_dir.parent / "supporting_information")
    output_dir = args.output or str(script_dir / "output")

    # 创建提取器
    extractor = SIExtractor(
        api_key=api_key,
        pages_per_chunk=args.pages_per_chunk,
        screen_model=args.screen_model,
        extract_model=args.extract_model,
        max_parallel_text_chunks=args.max_parallel_text_chunks,
        pdf_text_layout=args.pdf_text_layout,
        pdf_text_x_tolerance=args.pdf_text_x_tolerance,
        pdf_text_y_tolerance=args.pdf_text_y_tolerance,
    )

    # 执行批量处理
    results = extractor.batch_process(si_folder, output_dir)

    # 显示各文件结果
    print("\n各文件处理结果:")
    for fname, info in results.get("per_file", {}).items():
        status_icon = {"success": "[OK]", "skipped": "[--]", "failed": "[XX]"}.get(info["status"], "[?]")
        print(f"  {status_icon} {fname}: {info['status']} ({info['reactions']} reactions)")


if __name__ == "__main__":
    main()
