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

import hashlib
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
GP_TEMPLATE_SOURCE_CHAR_LIMIT = 12000
GP_TEMPLATE_SCHEMA_VERSION = "gp_template_v1"
GP_TEMPLATE_PROMPT_VERSION = "gp_template_prompt_v5"


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
    STRUCTURED_REACTION_PROMPT = """You are a chemistry literature extraction expert.
Extract every qualifying paragraph/prose reaction entry from SI text in source order.
Ignore tables, figures, captions, and analytical-only text.
Return only valid JSON matching the user request.

1. Scope and eligibility
- Extract every qualifying entry in source order. A qualifying entry has a specific compound name, label, or symbol; preparation/procedure language; and at least one reported isolated yield, ee, or er. Yield alone is sufficient.
- Product-characterization paragraphs are valid reaction entries when they identify an isolated product and report yield, ee, or er. Extract result data reported before NMR/HPLC/HRMS/spectra text.
- A line such as "Prepared according to General Procedure A/B..." followed by color, physical form, yield, ee, er, NMR, HPLC, or HRMS is a valid concrete reaction entry.
- Never extract optimization/screening/entry tables, figures, captions, or tabular lists. A General Procedure definition is context, not a standalone reaction, unless that same paragraph reports a specific substrate/product and yield, ee, or er.

2. GP template usage
- A supplied canonical GP JSON is the authoritative source for shared reaction_type, substrates, catalysts, ligands, other_components, conditions, intermediates, and step schema. Never re-parse or re-decide the GP.
- For every GP-referenced entry, return _gp_source with the supplied GP ID and output a complete reaction record now. Do not rely on later GP-template materialization.
- Copy applicable shared GP fields into every GP reaction, then apply explicit concrete-entry overrides. Precedence is concrete entry text override > canonical GP template default.
- GP templates do NOT provide products, targets, or source_pages. Those fields MUST come from the current concrete entry text.
- If a concrete entry overrides the GP template, write the overridden value directly into the top-level reaction field. Do not output audit-only override fields.
- Color, physical form, yield, ee, er, NMR, HPLC, and HRMS are product/result data, not condition overrides.
- Preserve the template step_count exactly. Assign entry-specific compounds to actual steps. Do not invent unnamed intermediates from "crude product", "residue", or "corresponding intermediate".

3. Output schema
- Return a JSON array of reaction objects.
- Every reaction has id, source_pages, reaction_type, substrates, products, catalysts, ligands, other_components, conditions, and targets.
- GP entries additionally use "_gp_source":"GeneralProcedureA".
- source_pages contains only page numbers from the concrete entry's surrounding "--- Page N ---" markers. Do not include a GP definition page merely because its template was supplied.
- For both single-step and multi-step reactions, substrates, products, catalysts, ligands, and other_components must be arrays of objects, never arrays of strings.
- Each compound object separates name, symbol, and amount when reported. Multiple substrates and multiple products are represented by multiple objects in the same array.
- name stores the chemical name only. symbol stores the reported label, compound number, product number, or short code.
- Do not keep trailing labels such as (34), (35'), (36), SM15, or 3a inside name. If no full name exists, use the symbol for both name and symbol. Never copy a product name into substrates.
- Label separation examples: source "... carboxylic acid (34)" -> {"name":"... carboxylic acid","symbol":"34"}; source "... dibromovinyl ... (35')" -> {"name":"... dibromovinyl ...","symbol":"35'"}; source "... methanol (36)" -> {"name":"... methanol","symbol":"36"}.
- catalysts contains catalysts, precatalysts, and complete preformed catalyst-complex names with reported amounts.
- ligands contains name and optional symbol for single-step reactions, and name, optional symbol, plus step for multi-step reactions. Ligand loading belongs with the reported catalyst or other component, not in ligands.
- other_components contains reacting reagents, bases, additives, and reductants. Exclude workup, washing, extraction, drying, chromatography, and purification materials.

4. Single-step schema
- Single-step compound objects do not use step.
- Single-step shape:
  {"id":"...","source_pages":[1],"reaction_type":"...","substrates":[{"name":"starting material A","symbol":null,"amount":"reported amount"},{"name":"starting material B","symbol":null,"amount":"reported amount"}],"products":[{"name":"reported product","symbol":"34","amount":"reported isolated mass"}],"catalysts":[{"name":"reported catalyst","symbol":null,"amount":"reported amount"}],"ligands":[{"name":"reported ligand","symbol":null}],"other_components":[{"name":"reported reagent/additive/base","symbol":null,"amount":"reported amount"}],"conditions":{"solvent":"reported solvent","solvent_amount":"reported solvent amount","atmosphere":"reported atmosphere","light_source":null,"wavelength":null,"temperature":"reported temperature","time":"reported time"},"targets":{"yield":"percent only","ee":null,"er":null}}

5. Multi-step schema
- Multi-step rules apply to every multi-step reaction, whether GP or non-GP.
- Use step_count >= 2 only for sequential chemical transformations. Staged addition, heating, workup, extraction, filtration, concentration, chromatography, and purification are not chemical steps by themselves.
- Every substrate, product, catalyst, ligand, and other_component object in a multi-step reaction must include integer step.
- intermediates uses produced_in_step and consumed_in_step, not step.
- conditions must be an object. Each condition field is [] or a list of {"step":N,"value":"..."} objects.
- Multi-step shape:
  {"id":"...","source_pages":[1,2],"reaction_type":"...","step_count":2,"substrates":[{"name":"starting material A","symbol":null,"amount":"reported amount","step":1},{"name":"starting material B","symbol":null,"amount":"reported amount","step":2}],"intermediates":[{"name":"intermediate from step 1","symbol":"reported label or null","produced_in_step":1,"consumed_in_step":2}],"products":[{"name":"final product","symbol":"36","amount":"reported isolated mass","step":2}],"catalysts":[{"name":"reported catalyst","symbol":null,"amount":"reported amount","step":1}],"ligands":[{"name":"reported ligand","symbol":null,"step":1}],"other_components":[{"name":"reagent for step 1","symbol":null,"amount":"reported amount","step":1},{"name":"reagent for step 2","symbol":null,"amount":"reported amount","step":2}],"conditions":{"solvent":[{"step":1,"value":"reported solvent for step 1"},{"step":2,"value":"reported solvent for step 2"}],"solvent_amount":[{"step":1,"value":"reported solvent amount for step 1"},{"step":2,"value":"reported solvent amount for step 2"}],"temperature":[{"step":1,"value":"reported temperature for step 1"},{"step":2,"value":"reported temperature for step 2"}],"time":[{"step":1,"value":"reported time for step 1"},{"step":2,"value":"reported time for step 2"}],"atmosphere":[],"light_source":[],"wavelength":[]},"targets":{"yield":"percent only","ee":null,"er":null}}

6. Generic substrate handling
- Keep generic substrate names as extracted unless the concrete substrate name is explicitly stated in the current entry text.
- A generic substrate name is a class-level, placeholder, corresponding/appropriate, or non-specific chemical identity. A fully specified chemical name is not generic.
- Product-derived generic substrate resolution is handled after extraction. Do not invent a product-derived concrete substrate name during reaction extraction.
- Label-only product names are not sufficient evidence for product-derived substrate completion.
- Never copy the complete product name into substrates.

7. Targets normalization
- targets contains yield, ee, and er only, using null when unreported. targets does not come from GP templates.
- products[].amount is mass or isolated amount only, never yield.
- targets.yield stores only the percent string, for example "81%". Use "81%", not "81% yield" or "81% (2.1 g) yield".
- Normalize "81% yield" to targets.yield "81%"; "81% (2.1 g) yield" to targets.yield "81%" and products[].amount "2.1 g"; "75% yield, 96% ee" to targets.yield "75%" and targets.ee "96%".
- targets.ee stores only the percent string, for example "96%"; normalize "96% ee" and "96% e.e." to "96%".
- targets.er keeps the reported ratio string such as "95:5"; do not convert er to ee.

8. Exclusions and invalid formats
- Invalid single-step formats: never output {"substrates":["substrate A"]}; never output {"products":["product A (81% yield)"]}; use object arrays and put yield only in targets.
- Invalid multi-step formats: never output compound arrays containing bare strings; never omit step on multi-step compound objects; never output conditions as a top-level list.
- Do not output explanations, Markdown, coverage lists, dr, conversion, selectivity, NMR_yield, or GC_yield."""

    GENERIC_SUBSTRATE_RESOLUTION_PROMPT = """You resolve generic chemistry substrate names after reaction extraction.

Task:
- Decide whether each generic substrate can be replaced by a specific substrate name inferred from the concrete product name or names.
- Resolve only when the inference is chemically valid, high confidence, and one-to-one.
- The resolved substrate must be the complete starting-material identity, not the minimal scaffold. Preserve all ring substituents, chain substituents, heteroatom substituents, protecting groups, and N/O/S substituents that can be mapped from the product back to that substrate.
- Do not copy the complete product name as a substrate.
- Do not resolve when the product name is only a label such as "Oxindole 2", "Product 3", or "compound 5".
- Evaluate every input substrate independently.
- If there are multiple products, resolution_evidence.product_name must name the exact product used as evidence.
- If there are multiple generic substrates, resolve only the substrates that have an explicit one-to-one mapping; return can_resolve=false for ambiguous substrates.
- Use the other substrates in the same reaction to decide whether a product fragment maps one-to-one to a generic substrate. If the product fragment could come from more than one starting material, return can_resolve=false.
- For generic amine, aniline, alcohol, phenol, thiol, amide, carbamate, sulfonamide, urea, or similar heteroatom-containing substrates, inspect substituents attached to the heteroatom in the product. If a heteroatom substituent is part of the original substrate identity, the resolved_name must retain it. If the substituent could have been introduced in a later step or its source is not unique, return can_resolve=false.
- Do not infer a substrate from the final product scaffold alone.
- Resolve only when the product name preserves a chemically meaningful fragment that maps one-to-one to the complete identity of the generic substrate.
- If the original substrate identity is not uniquely recoverable from the product name, return can_resolve=false.
- If multiple starting materials could have contributed the same product fragment, use the other reported substrates to disambiguate. If the mapping remains ambiguous, return can_resolve=false.

Current-paper regression examples:
- generic substrate: aniline
  product: (Z)-N-(2-bromophenyl)-N,2-dimethylbut-2-enamide
  resolved substrate: 2-bromoaniline
- generic substrate: aniline
  product: (Z)-N-(2-bromophenyl)-N-(methoxymethyl)-2-methylbut-2-enamide
  resolved substrate: 2-bromo-N-(methoxymethyl)aniline

Return ONLY compact valid JSON:
{
  "reaction_id": "...",
  "resolutions": [
    {
      "substrate_index": 1,
      "can_resolve": true,
      "resolved_name": "2-bromoaniline",
      "original_name": "aniline",
      "resolution_source": "product_name",
      "resolution_method": "gp_product_to_substrate_mapping",
      "resolution_confidence": "high",
      "resolution_evidence": {
        "product_name": "(Z)-N-(2-bromophenyl)-N,2-dimethylbut-2-enamide",
        "reason": "The N-(2-bromophenyl) fragment corresponds to 2-bromoaniline."
      }
    }
  ]
}

If a substrate cannot be resolved, return can_resolve=false with a short reason."""

    NON_GP_REACTION_PROMPT = """You extract standalone paragraph/prose reaction records from chemistry Supporting Information.
Return only valid JSON array of reaction objects.

Scope
- Extract only qualifying standalone reactions that do NOT rely on a General Procedure.
- Skip entries that say "prepared according to General Procedure", "following GP", "according to GP", or equivalent GP-reference wording.
- Do not inherit any General Procedure conditions, catalysts, ligands, or reagents.
- Extract every qualifying standalone preparation/synthesis/isolation entry in source order. Yield alone is sufficient; ee and er are optional.
- Ignore optimization/screening tables, figures, captions, and analytical-only NMR/HRMS/HPLC/spectra text after the reported yield.

Schema
- Use source_pages from the surrounding "--- Page N ---" markers. A spanning entry includes all evidence pages.
- catalysts contains catalysts and precatalysts; ligands contains ligand name only for single-step reactions; other_components contains bases, reagents, additives, and reductants.
- For both single-step and multi-step reactions, substrates, products, catalysts, ligands, and other_components must be arrays of objects, never arrays of strings.
- Each compound object separates name, symbol, and amount when reported. Multiple substrates and multiple products are represented by multiple objects in the same array.
- name stores the chemical name only. symbol stores the reported label, compound number, product number, or short code.
- Do not keep trailing labels such as (34), (35'), (36), SM15, or 3a inside name. If no full name exists, use the symbol for both name and symbol.
- Label separation examples: source "... carboxylic acid (34)" -> {"name":"... carboxylic acid","symbol":"34"}; source "... dibromovinyl ... (35')" -> {"name":"... dibromovinyl ...","symbol":"35'"}; source "... methanol (36)" -> {"name":"... methanol","symbol":"36"}.
- Single-step shape: {"id":"...","source_pages":[1],"reaction_type":"...","substrates":[{"name":"reported substrate A","symbol":null,"amount":"reported amount"},{"name":"reported substrate B","symbol":null,"amount":"reported amount"}],"products":[{"name":"reported product A","symbol":"34","amount":"reported isolated mass"},{"name":"reported product B","symbol":null,"amount":"reported isolated mass"}],"catalysts":[{"name":"reported catalyst","symbol":null,"amount":"reported amount"}],"ligands":[{"name":"reported ligand","symbol":null}],"other_components":[{"name":"reported reagent/additive/base","symbol":null,"amount":"reported amount"}],"conditions":{"solvent":"...","solvent_amount":"...","atmosphere":"...","light_source":null,"wavelength":null,"temperature":"...","time":"..."},"targets":{"yield":"percent only","ee":null,"er":null}}.
- Invalid single-step formats:
  - Never output {"substrates":["substrate A"]}; use {"substrates":[{"name":"substrate A","amount":null}]}.
  - Never output {"products":["product A (71% yield)"]}; use {"products":[{"name":"product A","amount":null}],"targets":{"yield":"71%"}}.
  - Never put yield into products[].name or products[].amount; targets.yield stores only the percent string.
- Multi-step shape rules:
  - Use step_count >= 2 only for two or more sequential chemical transformations. Staged addition, heating, workup, extraction, filtration, concentration, and purification do not create new steps by themselves.
  - If step_count >= 2, every item in substrates, products, catalysts, ligands, and other_components MUST be an object. Do not output bare strings in any compound array.
  - Every compound object in a multi-step entry MUST include an integer step field.
  - intermediates MUST be a list of objects using produced_in_step and consumed_in_step. Do not use step on intermediates.
  - conditions MUST be an object. Each condition field MUST be [] or a list of {"step": integer, "value": string} objects.
  - Do not output conditions as a top-level list.
- Generic multi-step example format, using placeholders only:
  [{"id":"example_multistep_1","source_pages":[1,2],"reaction_type":"standalone multi-step synthesis","step_count":2,"substrates":[{"name":"starting material A","symbol":null,"amount":"reported amount","step":1},{"name":"starting material B","symbol":null,"amount":"reported amount","step":1}],"intermediates":[{"name":"intermediate from step 1","symbol":"reported label or null","produced_in_step":1,"consumed_in_step":2}],"products":[{"name":"final product","symbol":"36","amount":"reported isolated amount","step":2}],"catalysts":[{"name":"reported catalyst","symbol":null,"amount":"reported amount","step":1}],"ligands":[{"name":"reported ligand","symbol":null,"step":1}],"other_components":[{"name":"reagent for step 1","symbol":null,"amount":"reported amount","step":1},{"name":"reagent for step 2","symbol":null,"amount":"reported amount","step":2}],"conditions":{"solvent":[{"step":1,"value":"reported solvent for step 1"},{"step":2,"value":"reported solvent for step 2"}],"solvent_amount":[{"step":1,"value":"reported solvent amount for step 1"},{"step":2,"value":"reported solvent amount for step 2"}],"temperature":[{"step":1,"value":"reported temperature for step 1"},{"step":2,"value":"reported temperature for step 2"}],"time":[{"step":1,"value":"reported time for step 1"},{"step":2,"value":"reported time for step 2"}],"atmosphere":[],"light_source":[],"wavelength":[]},"targets":{"yield":"reported final isolated yield","ee":null,"er":null}}]
- Invalid multi-step formats:
  - Never output {"other_components":["reagent A"]}; use {"other_components":[{"name":"reagent A","amount":null,"step":1}]}.
  - Never output {"other_components":[{"name":"reagent A","amount":"reported amount"}]} because step is missing.
  - Never output {"conditions":[{"step":1,"temperature":"reported temperature"}]} because conditions must be an object with condition-field lists.
- Do not output _gp_source, _entry_overrides, explanations, Markdown, coverage lists, dr, conversion, selectivity, NMR_yield, or GC_yield."""

    MIXED_REACTION_PROMPT = """You are a chemistry literature extraction expert. Extract every qualifying paragraph/prose reaction entry from SI text in source order. Ignore tables, figures, captions, and analytical-only text.
Return only a valid JSON array of reaction objects.

1. Scope
- This chunk may contain both General-Procedure-referenced entries and standalone downstream transformations. Extract all qualifying entries in source order.
- A standalone/downstream reaction starts from an already isolated product, cycloadduct, intermediate, compound label, or previously prepared material and performs a new transformation. Extract it as a non-GP reaction, do not set _gp_source, and do not inherit GP substrates, catalysts, ligands, other_components, or conditions.
- A reaction prepared according to a reported literature procedure, published procedure, reported procedure, literature procedure, or previously reported method is standalone non-GP unless it explicitly references one of the supplied General Procedure templates. For these literature-method entries, do not set _gp_source and do not inherit GP fields.
- Do not output the same concrete reaction twice as both GP and non-GP. If a product/yield entry is a downstream transformation of an isolated GP product, output only the standalone non-GP reaction.
- If multiple product names with yield, ee, or er appear on the same page or in the same chunk, extract each distinct product/result as a separate reaction. Do not stop after the first entry.
- Extract all qualifying entries in source order. Do not omit any.

2. GP template usage
- A GP-referenced entry explicitly says or clearly means it was prepared according to, following, or using one of the supplied canonical General Procedure templates. For that entry only, copy applicable shared GP fields into the complete reaction record and set "_gp_source" to the matching GP id.
- GP templates provide shared reaction_type, substrates, catalysts, ligands, other_components, conditions, intermediates, and step schema only for GP-referenced entries. Products, targets, and source_pages always come from the concrete entry text.
- If the supplied GP template has no step_count, the GP-referenced entry is single-step: copy shared fields, add entry products/targets/source_pages, and do not use item-level step.
- If the supplied GP template has step_count >= 2, every GP-referenced entry must keep the same step_count and the same multi-step schema surface.
- For multi-step GP-referenced entries, entry-specific starting materials usually belong to step 1, and the final isolated product usually belongs to step=step_count. If the concrete entry explicitly assigns a compound to another chemical step, use the entry evidence.

3. Output schema
- Every reaction has id, source_pages, reaction_type, substrates, products, catalysts, ligands, other_components, conditions, and targets.
- For both single-step and multi-step reactions, substrates, products, catalysts, ligands, and other_components must be arrays of objects, never arrays of strings.
- Each compound object separates name, symbol, and amount when reported. name stores the chemical name only. symbol stores the reported label, compound number, product number, or short code.
- Do not keep trailing labels such as (34), (35'), (36), SM15, or 3a inside name. If no full name exists, use the symbol for both name and symbol.
- Label separation examples: source "... carboxylic acid (34)" -> {"name":"... carboxylic acid","symbol":"34"}; source "... dibromovinyl ... (35')" -> {"name":"... dibromovinyl ...","symbol":"35'"}; source "... methanol (36)" -> {"name":"... methanol","symbol":"36"}.
- Use solvent_amount, not volume. Use light_source, not light source.
- catalysts contains catalysts and precatalysts; ligands contains ligand name only for single-step reactions; other_components contains bases, reagents, additives, reductants, and fixed transfer/derivatization reagents.
- source_pages must come from the concrete entry page markers, not from the GP definition page.

4. Single-step schema
- Single-step compound objects do not use step.
- Use this shape for standalone literature-procedure entries and for GP-referenced entries whose supplied GP template has no step_count:
  {"id":"...","source_pages":[18],"reaction_type":"...","substrates":[{"name":"reported substrate A","symbol":null,"amount":"reported amount"},{"name":"reported substrate B","symbol":null,"amount":"reported amount"}],"products":[{"name":"reported product name","symbol":"29","amount":"454 mg"}],"catalysts":[{"name":"reported catalyst","symbol":null,"amount":"reported amount"}],"ligands":[{"name":"reported ligand","symbol":null}],"other_components":[{"name":"reported reagent/base/additive","symbol":null,"amount":"reported amount"}],"conditions":{"solvent":"reported solvent","solvent_amount":"reported solvent amount","temperature":"reported temperature","time":"reported time","atmosphere":"reported atmosphere","light_source":null,"wavelength":null},"targets":{"yield":"81%","ee":null,"er":null}}
- For single-step reactions, do not output step_count and do not put step on any compound object.

5. Multi-step schema
- Use step_count >= 2 only for sequential chemical transformations. Staged addition, heating, workup, extraction, filtration, concentration, chromatography, and purification are not chemical steps by themselves.
- If a reaction has step_count >= 2, every item in substrates, products, catalysts, ligands, and other_components must include integer step.
- intermediates do not use step; they must use produced_in_step and consumed_in_step.
- conditions must be an object. Each condition field is [] or a list of {"step":N,"value":"..."} objects.
- Multi-step GP-referenced shape:
  {"id":"...","_gp_source":"GeneralProcedureC","source_pages":[24],"reaction_type":"...","step_count":2,"substrates":[{"name":"template substrate","symbol":"SM6","amount":"0.1 mmol","step":1}],"intermediates":[{"name":"explicit intermediate name","symbol":"6","produced_in_step":1,"consumed_in_step":2}],"products":[{"name":"final reported product","symbol":"21","amount":"51.9 mg","step":2}],"catalysts":[{"name":"template catalyst","symbol":null,"amount":"reported amount","step":1}],"ligands":[{"name":"template ligand","symbol":null,"step":1}],"other_components":[{"name":"template reagent","symbol":null,"amount":"reported amount","step":2}],"conditions":{"solvent":[{"step":1,"value":"..."},{"step":2,"value":"..."}],"solvent_amount":[],"temperature":[],"time":[],"atmosphere":[],"light_source":[],"wavelength":[]},"targets":{"yield":"65%","ee":"91%","er":null}}
- In multi-step GP-referenced entries, final isolated product objects must include step, usually step=step_count.
- Standalone non-GP entries use step_count only if the paragraph itself describes multiple sequential chemical transformations. A single-step literature procedure must not use step_count or item-level step.

6. Literature and published procedure entries
- reported literature procedure, published procedure, reported procedure, literature procedure, and previously reported method are not supplied GP templates.
- Extract these as standalone non-GP reactions with no _gp_source and no inherited GP fields.
- They are usually single-step unless the paragraph itself reports multiple sequential chemical transformations.

7. Targets normalization
- targets.yield and targets.ee store percent strings only, for example "81%" and "96%"; products[].amount stores mass or isolated amount only. targets.er keeps the reported ratio string.
- Use "81%", not "81% yield" or "81% (454 mg) yield". Put the mass in products[].amount.

8. Invalid formats
- Invalid single-step format: never output {"products":["product A (81% yield)"]}; use object arrays and put yield only in targets.
- Invalid multi-step format: never output a product, substrate, catalyst, ligand, or other_component without integer step when step_count is present.
- Invalid multi-step format: never output conditions as a top-level list.
- Do not output explanations, Markdown, coverage lists, dr, conversion, selectivity, NMR_yield, or GC_yield."""

    CHUNK_REACTION_ROUTER_PROMPT = """Route chemistry SI reaction extraction for one chunk.

Decide whether extractable prose reactions in the chunk rely on General Procedures, do not rely on General Procedures, both, or none.
Use the available GP candidates only as procedure references; do not extract reactions here.

Return ONLY compact valid JSON:
{{
  "route": "gp_only|non_gp_only|mixed|none",
  "jobs": [
    {{
      "job_id": "gp_1",
      "mode": "gp",
      "gp_keys": ["GeneralProcedureA"],
      "source_pages": [3, 5]
    }},
    {{
      "job_id": "non_gp_1",
      "mode": "non_gp",
      "gp_keys": [],
      "source_pages": [18, 19, 20]
    }}
  ],
  "confidence": "high|medium|low",
  "reason": "short reason"
}}

Rules:
- route="gp_only" when all extractable reactions rely on selected GP candidates.
- route="non_gp_only" when extractable reactions are standalone and should not inherit any GP.
- route="mixed" when the chunk contains both GP-referenced reactions and standalone non-GP reactions.
- route="none" when no qualifying prose reaction should be extracted.
- A GP job must list only applicable gp_keys from the available candidates.
- A non_gp job must have gp_keys=[] and must exclude GP-referenced entries.
- Do not enumerate only some product entries or examples in a job. A GP job means all qualifying entries in this chunk that reference its gp_keys; a non-GP job means all qualifying standalone entries in this chunk.
- Do not limit extraction to listed examples, product names, or the first entries seen in the chunk.
- Optional scope text is for logging only and must not narrow extraction coverage.

Reaction chunk:
{chunk_text}

Available GP candidates:
{gp_candidates}
"""

    GP_INJECTION_TEMPLATE = """
GENERAL PROCEDURE CONTEXT — apply these conditions when the entry specifies none:

{gp_block}

Use only the matching template for each entry. Return its gp_id in _gp_source.
The template is authoritative for shared fields and step_count; do not re-parse it.
If an entry overrides the template, write the overridden value directly in the top-level reaction field. Never copy products into substrates.
"""

    STAGE2_AUDIT_PROMPT = """Audit source_text against current_extraction for omitted qualifying prose reaction entries.
Return only {"missing_reactions":[<reaction objects>]}. Do not rewrite existing records.
Follow the system extraction schema and supplied canonical GP templates exactly.
Check consecutive product-characterization entries carefully; each distinct product/result is a record.
Every missing reaction must take source_pages from the concrete entry's page markers.
Never recover tables or use a GP definition page as source_pages.
Return {"missing_reactions":[]} when coverage is complete."""

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

    GP_TEMPLATE_PROMPT = """Extract one canonical General Procedure template from the source text.
Return ONLY valid JSON. Use reported information only.

GP ID: {gp_label}

Rules:
- step_count is present only for two or more genuine sequential chemical transformations.
- Heating/cooling, staged addition, stirring, workup, extraction, washing, drying, filtration, concentration, and purification are not new steps.
- Single-step templates must not contain step_count or item-level step.
- Multi-step templates require step_count >= 2 and a valid step on every substrate, catalyst, ligand, other_component, and every condition value.
- temperature records temperature values only; time records duration values only.
- When duration is embedded in a temperature phrase, split it into both fields. For example, "20 °C for 1 h, then 60 °C for 20 h" becomes temperature "20 °C then 60 °C" and time "1 h then 20 h".
- Do not leave time null when a duration such as min, h, hour(s), day(s), or overnight is reported in the procedure.
- In multi-step templates, substrates are main externally supplied starting-material classes, generic substrate classes, substrate ranges, or variable substrate-scope components that define the reaction series.
- A material generated in an earlier step and consumed in a later step is an intermediate, not a new external substrate.
- Pronouns, generic descriptions, or references to material obtained from a previous step must be assigned by role: use intermediates when they carry material forward between chemical transformations.
- A newly added material in a later step is a substrate only when it is a main building block or variable substrate-scope component.
- Fixed derivatization, alkylation, methylation, acylation, activation, reduction, oxidation, base, additive, or transfer reagents belong in other_components, even if atoms from them appear in the final product.
- A fixed methylating reagent used to derivatize an intermediate belongs in other_components, not substrates.
- intermediates records step-to-step material transfer using produced_in_step and consumed_in_step. Use a concise reported or descriptive name; keep evidence in procedure_details or evidence.
- catalysts retain the complete reported catalyst or preformed metal-ligand complex name and amount.
- ligands contain name only (plus step for multi-step). Never include amount or symbol. A ligand identifiable inside a preformed complex is also listed in ligands.
- other_components contains all reaction additives, reagents, bases, reductants, fixed transfer/derivatization reagents, and other reacting materials except substrates, catalysts, ligands, and solvents. Include name and reported amount; do not drop any reported reacting material.
- Exclude workup, extraction, washing, drying, and purification materials.
- Ignore later product examples, characterization entries, spectra, and analytical data even when they are present in the candidate text.
- If a substrate is a generic class, keep that reported class name and set is_generic_class=true.
- Preserve important order, sealing, degassing, pressure, and staged operation details in procedure_details.

Single-step shape:
{{"reaction_type":"...","substrates":[{{"name":"...","amount":"... or null","is_generic_class":true}}],"catalysts":[{{"name":"...","amount":"... or null"}}],"ligands":[{{"name":"..."}}],"other_components":[{{"name":"...","amount":"... or null"}}],"intermediates":[],"conditions":{{"solvent":"... or null","solvent_amount":"... or null","temperature":"... or null","time":"... or null","atmosphere":"... or null","light_source":"... or null","wavelength":"... or null"}},"procedure_details":[{{"sequence":1,"detail":"...","evidence":"exact source text"}}],"evidence":{{"ligands":[{{"index":0,"text":"exact source text"}}]}}}}

Multi-step shape uses the same fields plus step_count. Every substrate/catalyst/ligand/other_component has step. Every conditions field is a list of {{"step":N,"value":"..."}} objects. Ligand items have only name and step. Intermediates use {{"name":"...","produced_in_step":1,"consumed_in_step":2}} and must not also appear as external substrates.

Source GP text:
{gp_text}
"""

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

    @staticmethod
    def _pages_for_text_span(
        page_markers: List[Tuple[int, int]],
        start_pos: int,
        end_pos: int,
    ) -> List[int]:
        preceding_pages = [page for marker_pos, page in page_markers if marker_pos <= start_pos]
        pages = [preceding_pages[-1]] if preceding_pages else []
        pages.extend(
            page for marker_pos, page in page_markers if start_pos < marker_pos < end_pos
        )
        return sorted(set(pages))

    def _llm_trim_gp_record(self, record: Dict[str, Any], candidate_char_limit: int = GP_CONTEXT_CHAR_LIMIT) -> Dict[str, Any]:
        if not getattr(self, "client", None):
            record["final_text"] = record["raw_text"][:candidate_char_limit]
            record["stored_chars"] = len(record["final_text"])
            record["end_reason"] = "llm_failed_fallback"
            record["llm_trim"] = {"error": "missing_client"}
            record["boundary_suspect"] = True
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
            record["boundary_suspect"] = len(final_text) >= record.get("max_gp_chars", GP_TEMPLATE_SOURCE_CHAR_LIMIT)
            record["llm_trim"] = {
                "end_anchor": anchor,
                "include_anchor": bool(include_anchor),
                "trim_reason": parsed.get("trim_reason"),
                "confidence": parsed.get("confidence"),
            }
            return record
        except Exception as exc:
            record["final_text"] = record["raw_text"][:candidate_char_limit]
            record["stored_chars"] = len(record["final_text"])
            record["end_reason"] = "llm_failed_fallback"
            record["llm_trim"] = {"error": str(exc)}
            record["boundary_suspect"] = True
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
        max_gp_chars = GP_TEMPLATE_SOURCE_CHAR_LIMIT
        rule_complete_trust_chars = GP_TEMPLATE_SOURCE_CHAR_LIMIT
        records: List[Dict[str, Any]] = []
        page_markers = [
            (match.start(), int(match.group(1)))
            for match in re.finditer(r"--- Page\s+(\d+)\s+---", full_text)
        ]

        for i, (pos, title) in enumerate(filtered):
            has_next_gp = i + 1 < len(filtered)
            end_pos = filtered[i + 1][0] if has_next_gp else len(full_text)
            next_title = filtered[i + 1][1] if has_next_gp else None
            raw_text = full_text[pos:end_pos].strip()
            raw_text = re.sub(r'\n--- Page\s+\d+\s+---\s*$', '', raw_text).strip()
            raw_chars = len(raw_text)
            raw_effective_end_pos = pos + len(raw_text)
            raw_source_pages = self._pages_for_text_span(page_markers, pos, raw_effective_end_pos)
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
                "raw_effective_end_pos": raw_effective_end_pos,
                "has_next_gp": has_next_gp,
                "next_gp_title": next_title,
                "raw_chars_to_next_or_eof": raw_chars,
                "raw_source_pages": raw_source_pages,
                "source_pages": raw_source_pages,
                "stored_chars": min(raw_chars, max_gp_chars),
                "max_gp_chars": max_gp_chars,
                "rule_complete_trust_chars": rule_complete_trust_chars,
                "end_reason": end_reason,
                "pre_llm_end_reason": end_reason,
                "needs_llm_truncation": needs_llm,
                "boundary_suspect": raw_chars >= max_gp_chars,
            }
            if needs_llm:
                record = self._llm_trim_gp_record(record)
            final_text = str(record.get("final_text") or "")
            final_end_pos = pos + len(final_text)
            record["source_pages"] = self._pages_for_text_span(
                page_markers,
                pos,
                final_end_pos,
            )
            record["stored_chars"] = len(final_text)
            record["boundary_suspect"] = bool(record.get("boundary_suspect")) or len(final_text) >= max_gp_chars
            records.append(record)

        self.last_gp_records = records
        return records

    def extract_general_procedure_texts(self, pages: List[Dict]) -> Dict[str, str]:
        """Extract General Procedure text and return the existing public dict schema."""
        records = self.extract_general_procedure_records(pages)
        if not records:
            return {}

        gp_texts: Dict[str, str] = {}
        source_pages_by_gp: Dict[str, List[int]] = {}
        used_gp_keys = set()
        for record in records:
            key = self._make_unique_gp_key(str(record["key"]), used_gp_keys)
            text = record.get("final_text") or record.get("raw_text", "")
            gp_texts[key] = str(text)[:GP_TEMPLATE_SOURCE_CHAR_LIMIT]
            source_pages_by_gp[key] = list(record.get("source_pages") or [])

        split_texts = self._split_embedded_procedure_scopes(gp_texts)
        capped_texts: Dict[str, str] = {}
        used_final_keys = set()
        for key, text in split_texts.items():
            unique_key = self._make_unique_gp_key(str(key), used_final_keys)
            capped_texts[unique_key] = str(text)[:GP_TEMPLATE_SOURCE_CHAR_LIMIT]
        self.last_gp_source_pages = {
            key: list(source_pages_by_gp.get(key) or [])
            for key in capped_texts
        }
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

    def build_gp_templates(
        self,
        gp_texts: Dict[str, str],
        source_pages_by_gp: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, Dict]:
        """Call the LLM exactly once per GP and return canonical templates."""
        templates: Dict[str, Dict] = {}
        source_pages_by_gp = source_pages_by_gp or {}
        for gp_label, gp_text in (gp_texts or {}).items():
            print(f"    structuring GP: {gp_label}")
            try:
                response = self.client.chat.completions.create(
                    model=self.extract_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You extract canonical chemistry procedure JSON. Return valid JSON only.",
                        },
                        {
                            "role": "user",
                            "content": self.GP_TEMPLATE_PROMPT.format(
                                gp_label=gp_label,
                                gp_text=gp_text,
                            ),
                        },
                    ],
                    temperature=0.0,
                )
                parsed = self._parse_canonical_gp_response(
                    (response.choices[0].message.content or "").strip()
                )
                templates[gp_label] = self.validate_gp_template(
                    parsed,
                    gp_label=gp_label,
                    gp_text=gp_text,
                    source_pages=source_pages_by_gp.get(gp_label, []),
                )
                template = templates[gp_label]
                print(
                    f"      template ready: substrates={len(template['substrates'])}, "
                    f"catalysts={len(template['catalysts'])}, ligands={len(template['ligands'])}"
                )
            except Exception as exc:
                print(f"      GP template failed: {exc}")
                templates[gp_label] = {
                    "schema_version": GP_TEMPLATE_SCHEMA_VERSION,
                    "prompt_version": GP_TEMPLATE_PROMPT_VERSION,
                    "gp_id": gp_label,
                    "raw_text_sha256": hashlib.sha256(gp_text.encode("utf-8")).hexdigest(),
                    "status": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return templates

    @staticmethod
    def _parse_canonical_gp_response(raw: str) -> Dict:
        raw = (raw or "").strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        elif raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        data = json.loads(raw.strip())
        if not isinstance(data, dict):
            raise ValueError("GP template response must be an object")
        return data

    @staticmethod
    def _gp_name_key(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

    def validate_gp_template(
        self,
        payload: Dict,
        *,
        gp_label: str,
        gp_text: str,
        source_pages: Optional[List[int]] = None,
    ) -> Dict:
        """Validate and normalize the canonical single/multi-step GP union."""
        if not isinstance(payload, dict):
            raise ValueError("GP template must be an object")

        raw_step_count = payload.get("step_count")
        step_count = self._normalize_step_number(raw_step_count)
        is_multistep = raw_step_count is not None
        if is_multistep and (step_count is None or step_count < 2):
            raise ValueError("multi-step GP requires step_count >= 2")

        pages = []
        for page in source_pages or []:
            try:
                page_num = int(page)
            except (TypeError, ValueError):
                continue
            if page_num > 0:
                pages.append(page_num)

        template: Dict[str, Any] = {
            "schema_version": GP_TEMPLATE_SCHEMA_VERSION,
            "prompt_version": GP_TEMPLATE_PROMPT_VERSION,
            "gp_id": gp_label,
            "raw_text_sha256": hashlib.sha256(gp_text.encode("utf-8")).hexdigest(),
            "source_pages": sorted(set(pages)),
            "reaction_type": str(payload.get("reaction_type") or "unknown reaction").strip(),
        }
        if is_multistep:
            template["step_count"] = step_count

        def normalize_items(field: str) -> List[Dict]:
            values = payload.get(field) or []
            if not isinstance(values, list):
                raise ValueError(f"{field} must be a list")
            normalized: List[Dict] = []
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError(f"{field} items must be objects")
                name = str(value.get("name") or "").strip()
                if not name:
                    raise ValueError(f"{field} item is missing name")
                if field == "ligands":
                    allowed = {"name", "step"} if is_multistep else {"name"}
                    unexpected = set(value) - allowed
                    if unexpected:
                        raise ValueError(f"ligand item has unsupported fields: {sorted(unexpected)}")
                    item: Dict[str, Any] = {"name": name}
                else:
                    item = {"name": name, "amount": value.get("amount")}
                    if field == "substrates":
                        item["is_generic_class"] = bool(value.get("is_generic_class", False))
                if is_multistep:
                    step = self._normalize_step_number(value.get("step"))
                    if step is None or step > step_count:
                        raise ValueError(f"{field} item has invalid step")
                    item["step"] = step
                elif value.get("step") is not None:
                    raise ValueError(f"single-step {field} item must not have step")
                normalized.append(item)
            return normalized

        for field in ("substrates", "catalysts", "ligands", "other_components"):
            template[field] = normalize_items(field)

        deduped_ligands = []
        seen_ligands = set()
        for ligand in template["ligands"]:
            key = (self._gp_name_key(ligand.get("name")), ligand.get("step"))
            if key not in seen_ligands:
                seen_ligands.add(key)
                deduped_ligands.append(ligand)
        template["ligands"] = deduped_ligands

        ligand_names = {self._gp_name_key(item.get("name")) for item in template["ligands"]}
        template["other_components"] = [
            item for item in template["other_components"]
            if self._gp_name_key(item.get("name")) not in ligand_names
        ]

        raw_intermediates = payload.get("intermediates")
        if raw_intermediates is None:
            raw_intermediates = []
        if not isinstance(raw_intermediates, list):
            raise ValueError("intermediates must be a list")
        intermediates = []
        for value in raw_intermediates:
            if not isinstance(value, dict):
                raise ValueError("intermediate items must be objects")
            name = str(value.get("name") or "").strip()
            if not name:
                raise ValueError("intermediate item is missing name")
            if is_multistep:
                produced = self._normalize_step_number(value.get("produced_in_step"))
                consumed = self._normalize_step_number(value.get("consumed_in_step"))
                if produced is None or consumed is None or produced >= consumed or consumed > step_count:
                    raise ValueError(f"intermediate has invalid step relation: {value!r}")
                item: Dict[str, Any] = {
                    "name": name,
                    "produced_in_step": produced,
                    "consumed_in_step": consumed,
                }
                if value.get("description") not in (None, ""):
                    item["description"] = value.get("description")
                intermediates.append(item)
            else:
                if value.get("produced_in_step") is not None or value.get("consumed_in_step") is not None:
                    raise ValueError("single-step intermediate item must not have step relation")
                item = {"name": name}
                if value.get("description") not in (None, ""):
                    item["description"] = value.get("description")
                intermediates.append(item)
        template["intermediates"] = intermediates

        if is_multistep:
            intermediate_keys = {
                self._gp_name_key(item.get("name"))
                for item in intermediates
                if self._gp_name_key(item.get("name"))
            }
            for field in ("substrates",):
                duplicate_keys = {
                    self._gp_name_key(item.get("name"))
                    for item in template.get(field, [])
                    if self._gp_name_key(item.get("name")) in intermediate_keys
                }
                if duplicate_keys:
                    raise ValueError(
                        f"intermediates must not be duplicated in {field}: {sorted(duplicate_keys)}"
                    )

        raw_conditions = payload.get("conditions") or {}
        if not isinstance(raw_conditions, dict):
            raise ValueError("conditions must be an object")
        condition_keys = (
            "solvent", "solvent_amount", "temperature",
            "time", "atmosphere", "light_source", "wavelength",
        )
        conditions: Dict[str, Any] = {}
        for key in condition_keys:
            value = raw_conditions.get(key)
            if key == "solvent_amount" and value in (None, ""):
                value = raw_conditions.get("volume")
            if is_multistep:
                value = [] if value in (None, "") else value
                if not isinstance(value, list):
                    raise ValueError(f"multi-step condition {key} must be a list")
                normalized_values = []
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError(f"multi-step condition {key} items must be objects")
                    step = self._normalize_step_number(item.get("step"))
                    if step is None or step > step_count:
                        raise ValueError(f"multi-step condition {key} has invalid step")
                    if item.get("value") not in (None, ""):
                        normalized_values.append({"step": step, "value": item.get("value")})
                conditions[key] = normalized_values
            else:
                if isinstance(value, list):
                    raise ValueError(f"single-step condition {key} must be scalar")
                conditions[key] = None if value in (None, "") else value
        template["conditions"] = conditions
        template["procedure_details"] = payload.get("procedure_details") or []
        template["evidence"] = payload.get("evidence") or {}
        template["status"] = "valid"
        return template

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
        for field in ("substrates", "products", "catalysts", "ligands", "other_components", "additives", "reagents"):
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

    def _validate_single_step_compound_arrays(self, reaction: Dict) -> None:
        if reaction.get("step_count") is not None:
            return
        for field in ("substrates", "products", "catalysts", "ligands", "other_components", "additives", "reagents"):
            values = reaction.get(field)
            if values is None:
                continue
            if not isinstance(values, list):
                raise ValueError(f"single-step {field} must be a list")
            for item in values:
                if not isinstance(item, dict):
                    raise ValueError(f"single-step {field} items must be objects")

    def _collect_step_annotations(self, reaction: Dict) -> List[int]:
        """Return normalized step annotations already present in a reaction object."""
        steps: List[int] = []
        for field in ("substrates", "products", "catalysts", "ligands", "other_components", "additives", "reagents"):
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

        compound_fields = (
            "substrates", "products", "catalysts", "ligands",
            "other_components", "additives", "reagents",
        )
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

    def _normalize_condition_amount_key(self, reaction: Dict) -> Dict:
        """Canonicalize solvent quantity conditions to solvent_amount."""
        conditions = reaction.get("conditions")
        if not isinstance(conditions, dict):
            return reaction

        if "volume" in conditions:
            volume_value = conditions.pop("volume")
            solvent_amount_value = conditions.get("solvent_amount")
            if not self._has_meaningful_value(solvent_amount_value):
                conditions["solvent_amount"] = volume_value
        if "concentration" in conditions and not self._has_meaningful_value(
            conditions.get("concentration")
        ):
            conditions.pop("concentration", None)
        return reaction

    def _split_trailing_reported_symbol(self, item: Dict) -> None:
        """Move a trailing short compound label from name to symbol when safe.

        This is intentionally conservative: it only handles common SI labels such
        as "(34)", "(35')", "(SM15)", "(3a)", or "(7-(E))" at the very end of a
        name. It does not split chemistry parentheticals like "(R,R)-ligand" or
        "(E)-alkene" because those do not match the short-label pattern.
        """
        if not isinstance(item, dict):
            return
        if self._has_meaningful_value(item.get("symbol")):
            return
        name = item.get("name")
        if not isinstance(name, str):
            return
        match = re.fullmatch(
            r"\s*(?P<base>.+?)\s*\((?P<label>(?:[A-Za-z]{1,8}-)?\d+[A-Za-z]?'?|[A-Za-z]{1,8}\d+[A-Za-z]?'?|\d+-\([EZ]\))\)\s*",
            name,
        )
        if not match:
            return
        base = re.sub(r"\s+", " ", match.group("base")).strip()
        label = match.group("label").strip()
        if not base or not label:
            return
        item["name"] = base
        item["symbol"] = label

    def _normalize_compound_name_symbols(self, reaction: Dict) -> Dict:
        """Normalize name/symbol separation for all compound-role arrays."""
        for field in (
            "substrates",
            "products",
            "catalysts",
            "ligands",
            "other_components",
            "additives",
            "reagents",
            "intermediates",
        ):
            values = reaction.get(field)
            if not isinstance(values, list):
                continue
            for item in values:
                self._split_trailing_reported_symbol(item)
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
        reaction = self._normalize_condition_amount_key(reaction)
        reaction = self._normalize_compound_name_symbols(reaction)
        self._validate_single_step_compound_arrays(reaction)
        ligands = reaction.get("ligands")
        if ligands is None:
            reaction["ligands"] = []
        elif not isinstance(ligands, list):
            raise ValueError("ligands must be a list")
        else:
            clean_ligands = []
            for ligand in ligands:
                if not isinstance(ligand, dict) or not str(ligand.get("name") or "").strip():
                    continue
                clean = {"name": str(ligand["name"]).strip()}
                if self._has_meaningful_value(ligand.get("symbol")):
                    clean["symbol"] = str(ligand["symbol"]).strip()
                if reaction.get("step_count") is not None:
                    step = self._normalize_step_number(ligand.get("step"))
                    if step is None:
                        raise ValueError("multi-step ligand is missing step")
                    clean["step"] = step
                clean_ligands.append(clean)
            reaction["ligands"] = clean_ligands
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
                    {"role": "system", "content": self.STRUCTURED_REACTION_PROMPT},
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
                       gp_texts: Optional[Dict[str, str]] = None,
                       gp_templates: Optional[Dict[str, Dict]] = None) -> List[Dict]:
        result = self.stage2_extract_with_meta(
            chunk_text,
            chunk_label,
            registry=registry,
            gp_texts=gp_texts,
            gp_templates=gp_templates,
        )
        self.stage2_audit_recovered = int(
            getattr(self, "stage2_audit_recovered", 0) or 0
        ) + int(result.get("audit_recovered") or 0)
        return result.get("reactions") or []

    @staticmethod
    def _reaction_gp_context(gp_label: str, template: Dict) -> Dict:
        """Project a validated GP template onto the reaction-only context."""
        procedure_details = []
        for value in template.get("procedure_details") or []:
            if not isinstance(value, dict):
                continue
            item = {
                key: value.get(key)
                for key in ("sequence", "step", "detail")
                if value.get(key) is not None
            }
            if item:
                procedure_details.append(item)
        projected = {
            "gp_id": str(template.get("gp_id") or gp_label),
            "reaction_type": template.get("reaction_type"),
            "substrates": template.get("substrates") or [],
            "catalysts": template.get("catalysts") or [],
            "ligands": template.get("ligands") or [],
            "other_components": template.get("other_components") or [],
            "intermediates": template.get("intermediates") or [],
            "conditions": template.get("conditions") or {},
            "procedure_details": procedure_details,
        }
        if template.get("step_count") is not None:
            projected["step_count"] = template["step_count"]
        return projected

    def _fallback_router_from_selected_gp(
        self,
        selected_gp_texts: Dict[str, str],
        gp_selection_debug: Optional[Dict] = None,
        reason: str = "",
    ) -> Dict:
        selected_keys = list((selected_gp_texts or {}).keys())
        if selected_keys:
            route = "gp_only"
            jobs = [{
                "job_id": "gp_1",
                "mode": "gp",
                "gp_keys": selected_keys,
                "scope": "fallback GP extraction for selected procedures",
                "source_pages": [],
            }]
        else:
            route = "non_gp_only"
            jobs = [{
                "job_id": "non_gp_1",
                "mode": "non_gp",
                "gp_keys": [],
                "scope": "fallback standalone non-GP extraction",
                "source_pages": [],
            }]
        return {
            "route": route,
            "jobs": jobs,
            "confidence": "low",
            "reason": reason or "router fallback",
            "fallback": True,
            "gp_selection": dict(gp_selection_debug or {}),
        }

    def route_reaction_chunk(
        self,
        chunk_text: str,
        gp_texts: Optional[Dict[str, str]],
    ) -> Dict:
        """Use an LLM router to split a chunk into GP/non-GP extraction jobs."""
        valid_gp_texts = {
            key: text for key, text in (gp_texts or {}).items()
            if isinstance(text, str) and text.strip()
        }
        if not valid_gp_texts:
            router = self._fallback_router_from_selected_gp(
                {},
                {"mode": "none", "selected_gp_keys": []},
                reason="no GP candidates available",
            )
            router["fallback"] = False
            router["confidence"] = "high"
            return router

        if not getattr(self, "client", None):
            selected = self.select_gp_for_chunk(chunk_text, valid_gp_texts)
            return self._fallback_router_from_selected_gp(
                selected,
                getattr(self, "last_gp_selection_debug", {}) or {},
                reason="router LLM unavailable",
            )

        candidates = self._build_gp_resolution_candidates(valid_gp_texts)
        try:
            response = self.client.chat.completions.create(
                model=self.extract_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You route chemistry SI reaction chunks. Return valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": self.CHUNK_REACTION_ROUTER_PROMPT.format(
                            chunk_text=chunk_text[:5000],
                            gp_candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
                        ),
                    },
                ],
                temperature=0.0,
            )
            parsed = self._parse_compact_json_response(response.choices[0].message.content or "")
            router = self._normalize_reaction_router(parsed, valid_gp_texts)
            self.last_gp_selection_debug = {
                "mode": "router",
                "selected_gp_keys": sorted({
                    key
                    for job in router.get("jobs", [])
                    if job.get("mode") == "gp"
                    for key in job.get("gp_keys", [])
                }),
                "route": router.get("route"),
                "confidence": router.get("confidence"),
                "reason": router.get("reason"),
            }
            return router
        except Exception as exc:
            selected = self.select_gp_for_chunk(chunk_text, valid_gp_texts)
            router = self._fallback_router_from_selected_gp(
                selected,
                getattr(self, "last_gp_selection_debug", {}) or {},
                reason="router_error",
            )
            router["router_error"] = f"{type(exc).__name__}: {exc}"
            return router

    def _normalize_reaction_router(self, parsed, gp_texts: Dict[str, str]) -> Dict:
        if not isinstance(parsed, dict):
            raise ValueError("router returned non-object JSON")
        allowed_routes = {"gp_only", "non_gp_only", "mixed", "none"}
        route = str(parsed.get("route") or "").strip()
        if route not in allowed_routes:
            raise ValueError(f"invalid router route: {route}")

        raw_jobs = parsed.get("jobs") if isinstance(parsed.get("jobs"), list) else []
        jobs = []
        for index, raw_job in enumerate(raw_jobs, 1):
            if not isinstance(raw_job, dict):
                continue
            mode = str(raw_job.get("mode") or "").strip()
            if mode not in {"gp", "non_gp"}:
                continue
            gp_keys = raw_job.get("gp_keys") if isinstance(raw_job.get("gp_keys"), list) else []
            gp_keys = [
                str(key)
                for key in gp_keys
                if str(key) in gp_texts
            ]
            if mode == "gp" and not gp_keys:
                continue
            if mode == "non_gp":
                gp_keys = []
            jobs.append({
                "job_id": str(raw_job.get("job_id") or f"{mode}_{index}"),
                "mode": mode,
                "gp_keys": gp_keys,
                "scope": str(raw_job.get("scope") or ""),
                "source_pages": self._normalize_source_pages(raw_job.get("source_pages")),
            })

        if route == "none":
            jobs = []
        elif route == "gp_only":
            jobs = [job for job in jobs if job["mode"] == "gp"]
        elif route == "non_gp_only":
            jobs = [job for job in jobs if job["mode"] == "non_gp"]
        elif route == "mixed":
            has_gp = any(job["mode"] == "gp" for job in jobs)
            has_non_gp = any(job["mode"] == "non_gp" for job in jobs)
            if not (has_gp and has_non_gp):
                raise ValueError("mixed route requires both gp and non_gp jobs")

        if route != "none" and not jobs:
            raise ValueError("router returned no executable jobs")

        return {
            "route": route,
            "jobs": jobs,
            "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason"),
        }

    def _build_gp_block_for_job(
        self,
        job: Dict,
        gp_templates: Optional[Dict[str, Dict]],
    ) -> Tuple[str, List[str]]:
        gp_entries = []
        invalid_gp_labels = []
        for label in job.get("gp_keys") or []:
            template = (gp_templates or {}).get(label)
            if not isinstance(template, dict) or template.get("status") != "valid":
                invalid_gp_labels.append(label)
                continue
            reaction_context = self._reaction_gp_context(label, template)
            text = json.dumps(reaction_context, ensure_ascii=False, sort_keys=True, indent=2)
            gp_entries.append(f"=== {label} ===\n{text}")
        if invalid_gp_labels:
            return "", invalid_gp_labels
        if not gp_entries:
            return "", []
        return self.GP_INJECTION_TEMPLATE.format(gp_block="\n\n".join(gp_entries)), []

    def _stage2_prompt_for_job(self, job: Dict) -> Tuple[str, str]:
        mode = str(job.get("mode") or "")
        if mode == "mixed":
            return self.MIXED_REACTION_PROMPT, "mixed_gp_non_gp"
        if mode == "gp":
            return self.STRUCTURED_REACTION_PROMPT, "gp_template"
        return self.NON_GP_REACTION_PROMPT, "non_gp"

    @staticmethod
    def _chunk_has_target_cue(chunk_text: str) -> bool:
        text = chunk_text or ""
        return bool(re.search(
            r"(?i)(?:\b\d{1,3}(?:\.\d+)?\s*%(?:\s*\([^)]{1,80}\))?\s*(?:yield|ee|e\.e\.)\b|"
            r"\byield\s*\d{1,3}(?:\.\d+)?\s*%|"
            r"\ber\s*[:=]?\s*\d+\s*:\s*\d+)",
            text,
        ))

    @staticmethod
    def _reaction_has_product(reaction: Dict) -> bool:
        products = reaction.get("products")
        if not isinstance(products, list):
            return False
        return any(isinstance(product, dict) and product.get("name") for product in products)

    @staticmethod
    def _reaction_targets_all_empty(reaction: Dict) -> bool:
        targets = reaction.get("targets")
        if not isinstance(targets, dict):
            return True
        return not any(targets.get(key) for key in ("yield", "ee", "er"))

    def _detect_missing_gp_targets_for_retry(self, chunk_text: str, reactions: List[Dict], job: Dict) -> Optional[str]:
        if str(job.get("mode") or "") not in {"gp", "mixed"}:
            return None
        if not self._chunk_has_target_cue(chunk_text):
            return None
        missing_ids = []
        for reaction in reactions or []:
            if not isinstance(reaction, dict):
                continue
            if self._reaction_has_product(reaction) and self._reaction_targets_all_empty(reaction):
                missing_ids.append(str(reaction.get("id") or "<missing id>"))
        if not missing_ids:
            return None
        preview = ", ".join(missing_ids[:5])
        if len(missing_ids) > 5:
            preview += f", ... (+{len(missing_ids) - 5} more)"
        return (
            "Previous output missed targets. Extract percent-only yield/ee values from "
            "the concrete entry line before NMR/HPLC/HRMS text. Use \"81%\", not "
            "\"81% yield\" or \"81% (2.1 g) yield\". "
            f"Reactions with products but empty targets: {preview}."
        )

    def _detect_unresolved_generic_substrates_for_retry(self, reactions: List[Dict], job: Dict) -> Optional[str]:
        missing = []
        for reaction in reactions or []:
            if not isinstance(reaction, dict):
                continue
            product_names = self._specific_product_names(reaction)
            if len(product_names) != 1:
                continue
            for index, substrate in enumerate(reaction.get("substrates") or []):
                if not isinstance(substrate, dict):
                    continue
                name = str(substrate.get("name") or "").strip()
                if not self._is_generic_substrate_name(name):
                    continue
                source = str(substrate.get("resolution_source") or "").strip()
                method = str(substrate.get("resolution_method") or "").strip()
                if source == "product_name" or method == "gp_product_to_substrate_mapping":
                    continue
                missing.append(
                    f"{reaction.get('id') or '<missing id>'} substrate[{index}]={name!r}"
                )
        if not missing:
            return None
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += f", ... (+{len(missing) - 5} more)"
        return (
            "Previous output left a generic substrate unresolved although a unique product name is available. "
            "For every generic substrate, either emit the full high-confidence product_name resolution metadata "
            "or keep the generic name only when the mapping is ambiguous. If chemically valid and one-to-one, "
            "write the concrete substrate name directly in substrates[].name. "
            f"Unresolved generic substrates: {preview}."
        )

    @staticmethod
    def _has_meaningful_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return any(SIExtractor._has_meaningful_value(item) for item in value)
        if isinstance(value, dict):
            return any(SIExtractor._has_meaningful_value(item) for item in value.values())
        return True

    def _template_for_gp_job_reaction(
        self,
        reaction: Dict,
        job: Dict,
        gp_templates: Optional[Dict[str, Dict]],
    ) -> Tuple[Optional[str], Optional[Dict], List[str]]:
        if not gp_templates:
            return None, None, []
        gp_keys = [str(key) for key in (job.get("gp_keys") or []) if str(key)]
        gp_key, template = self._gp_template_for_reaction(reaction, gp_templates)
        missing = []
        if not reaction.get("_gp_source"):
            missing.append("_gp_source")
        if not gp_key and len(gp_keys) == 1:
            gp_key = gp_keys[0]
            candidate = gp_templates.get(gp_key)
            template = candidate if isinstance(candidate, dict) else None
        if not gp_key or not isinstance(template, dict) or template.get("status") != "valid":
            return gp_key, None, missing
        return gp_key, template, missing

    def _detect_missing_gp_template_fields_for_retry(
        self,
        reactions: List[Dict],
        job: Dict,
        gp_templates: Optional[Dict[str, Dict]],
    ) -> Optional[str]:
        mode = str(job.get("mode") or "")
        if mode not in {"gp", "mixed"} or not gp_templates:
            return None
        missing_by_reaction = []
        for reaction in reactions or []:
            if not isinstance(reaction, dict):
                continue
            if mode == "mixed" and not reaction.get("_gp_source"):
                continue
            _, template, missing_fields = self._template_for_gp_job_reaction(
                reaction, job, gp_templates
            )
            if not template:
                if missing_fields:
                    missing_by_reaction.append(
                        f"{reaction.get('id') or '<missing id>'}: {', '.join(missing_fields)}"
                    )
                continue
            for field in ("substrates", "catalysts", "ligands", "other_components", "intermediates"):
                if self._has_meaningful_value(template.get(field)) and not self._has_meaningful_value(reaction.get(field)):
                    missing_fields.append(field)
            template_conditions = template.get("conditions")
            reaction_conditions = reaction.get("conditions")
            if isinstance(template_conditions, dict) and self._has_meaningful_value(template_conditions):
                if not isinstance(reaction_conditions, dict):
                    missing_fields.append("conditions")
                else:
                    for key, value in template_conditions.items():
                        if self._has_meaningful_value(value) and not self._has_meaningful_value(reaction_conditions.get(key)):
                            missing_fields.append(f"conditions.{key}")
            if template.get("reaction_type") and not reaction.get("reaction_type"):
                missing_fields.append("reaction_type")
            if template.get("step_count") is not None:
                if reaction.get("step_count") != template.get("step_count"):
                    missing_fields.append("step_count")
            if missing_fields:
                unique_fields = list(dict.fromkeys(missing_fields))
                missing_by_reaction.append(
                    f"{reaction.get('id') or '<missing id>'}: {', '.join(unique_fields)}"
                )
        if not missing_by_reaction:
            return None
        preview = "; ".join(missing_by_reaction[:5])
        if len(missing_by_reaction) > 5:
            preview += f"; ... (+{len(missing_by_reaction) - 5} more)"
        return (
            "Previous output omitted shared GP template fields. For every GP-referenced entry, "
            "output a complete reaction by copying applicable substrates, catalysts, ligands, "
            "other_components, conditions, intermediates, reaction_type, and step schema from "
            "the supplied canonical GP template, then apply entry-specific overrides. "
            "Do not treat products or targets as GP-template fields; products and yield/ee/er "
            "must come from the concrete entry text. "
            f"Missing GP-derived fields: {preview}."
        )

    def _stage2_user_content_for_job(
        self,
        chunk_text: str,
        chunk_label: str,
        job: Dict,
        gp_block: str,
        previous_schema_error: Optional[str] = None,
        previous_target_error: Optional[str] = None,
    ) -> str:
        mode = str(job.get("mode") or "")
        if mode == "gp":
            instruction = (
                "Extract all qualifying reactions in this chunk that explicitly rely on "
                "the supplied canonical General Procedure template(s). "
                "Do not extract standalone non-GP reactions in this job.\n"
            )
        elif mode == "mixed":
            instruction = (
                "Extract all qualifying reactions in this mixed chunk. Use the supplied "
                "canonical General Procedure template(s) only for entries that explicitly "
                "rely on them and mark those records with _gp_source. Extract standalone "
                "or downstream transformations as non-GP reactions without inheriting GP "
                "fields and without _gp_source. Do not output the same concrete reaction "
                "twice as both GP and non-GP.\n"
            )
        else:
            instruction = (
                "Extract all standalone qualifying reactions in this chunk that do not rely on "
                "any General Procedure. Skip GP-referenced entries in this job.\n"
            )
        retry_feedback = ""
        if previous_schema_error:
            retry_feedback = (
                "\nPrevious output failed schema validation: "
                f"{previous_schema_error}\n"
                "Fix only the JSON schema shape while preserving the extracted chemistry. "
                "Compound arrays must contain objects, not strings, for both single-step "
                "and multi-step reactions. Split each string into {\"name\": ..., "
                "\"amount\": ...}; keep yield only in targets. For multi-step reactions, "
                "compound objects must include integer step, intermediates must use "
                "produced_in_step/consumed_in_step, and conditions must be an object. "
                "In mixed GP-referenced reactions using a multi-step GP template, every "
                "substrate/product/catalyst/ligand/other_component object must include "
                "integer step. Final reported products usually use step=step_count. "
                "For single-step standalone literature-procedure entries, remove "
                "step_count and do not use item-level step.\n"
            )
        if previous_target_error:
            retry_feedback += f"\n{previous_target_error}\n"
        return (
            f"Processing chunk: {chunk_label}\n"
            f"Extraction job: {job.get('job_id')} ({mode})\n"
            f"{instruction}"
            f"{retry_feedback}"
            "If a full chemical name is not explicitly present in this text, "
            "keep the short label/code in symbol and do not invent a full name, "
            "including generic substrate names that might later be resolved from a reported product. "
            "Product-derived generic-substrate resolution is handled after extraction. "
            "Registry-based name completion is handled after extraction.\n"
            f"{gp_block}"
            f"\nText content:\n{chunk_text}"
        )

    def _run_stage2_job(
        self,
        chunk_text: str,
        chunk_label: str,
        job: Dict,
        gp_block: str,
        allowed_page_nums: Optional[List[int]],
        gp_templates: Optional[Dict[str, Dict]] = None,
        allow_non_gp_split_fallback: bool = False,
    ) -> Dict:
        system_prompt, prompt_mode = self._stage2_prompt_for_job(job)
        attempts = []
        previous_schema_error = None
        previous_target_error = None
        for attempt in range(3):
            raw_preview = ""
            try:
                response = self.client.chat.completions.create(
                    model=self.extract_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": self._stage2_user_content_for_job(
                                chunk_text,
                                chunk_label,
                                job,
                                gp_block,
                                previous_schema_error=previous_schema_error,
                                previous_target_error=previous_target_error,
                            ),
                        },
                    ],
                    temperature=0.1,
                )
                gpt_response = response.choices[0].message.content or ""
                raw_preview = gpt_response[:2000]
                if not gpt_response.strip():
                    attempts.append({
                        "attempt": attempt + 1,
                        "exception_type": "EmptyResponse",
                        "message": "empty response",
                        "raw_response_preview": raw_preview,
                        "prompt_mode": prompt_mode,
                    })
                    print(f"  [WARN] Stage2 attempt {attempt+1}/3: empty response ({chunk_label}, {job.get('job_id')})")
                    continue
                reactions = self.sanitize_reactions_schema(
                    self.parse_and_validate_json(gpt_response),
                    allowed_page_nums=allowed_page_nums,
                )
                template_error = self._detect_missing_gp_template_fields_for_retry(
                    reactions, job, gp_templates
                )
                target_error = self._detect_missing_gp_targets_for_retry(chunk_text, reactions, job)
                quality_errors = [error for error in (template_error, target_error) if error]
                if quality_errors and attempt < 2:
                    if len(quality_errors) > 1:
                        exception_type = "ExtractionQualityRetry"
                    elif template_error:
                        exception_type = "MissingGPTemplateFields"
                    else:
                        exception_type = "MissingTargets"
                    quality_error = "\n".join(quality_errors)
                    attempts.append({
                        "attempt": attempt + 1,
                        "exception_type": exception_type,
                        "message": quality_error[:1000],
                        "raw_response_preview": raw_preview,
                        "prompt_mode": prompt_mode,
                    })
                    previous_schema_error = None
                    previous_target_error = quality_error[:1000]
                    print(f"  [WARN] Stage2 attempt {attempt+1}/3 quality retry ({chunk_label}, {job.get('job_id')}): {quality_error}")
                    continue
                previous_target_error = None
                missing = self.stage2_audit_missing_reactions(
                    chunk_text,
                    chunk_label,
                    reactions,
                    gp_block=gp_block,
                )
                if missing:
                    print(f"  [Stage2 audit] recovered {len(missing)} omitted reactions ({chunk_label}, {job.get('job_id')})")
                    reactions = self.merge_results([reactions, missing])
                return {
                    "reactions": reactions,
                    "audit_recovered": len(missing),
                    "error": None,
                    "job_log": {
                        "job_id": job.get("job_id"),
                        "mode": job.get("mode"),
                        "prompt_mode": prompt_mode,
                        "gp_keys": list(job.get("gp_keys") or []),
                        "reaction_count": len(reactions),
                        "reaction_ids": [
                            str(reaction.get("id"))
                            for reaction in reactions
                            if isinstance(reaction, dict) and reaction.get("id")
                        ],
                        "attempts": attempts,
                    },
                }
            except Exception as e:
                attempts.append({
                    "attempt": attempt + 1,
                    "exception_type": type(e).__name__,
                    "message": str(e)[:1000],
                    "raw_response_preview": raw_preview,
                    "prompt_mode": prompt_mode,
                })
                previous_schema_error = str(e)[:1000]
                previous_target_error = None
                print(f"  [WARN] Stage2 attempt {attempt+1}/3 failed ({chunk_label}, {job.get('job_id')}): {e}")

        error = "stage2_failed_after_retries"
        fallback = None
        if job.get("mode") == "non_gp" and allow_non_gp_split_fallback:
            fallback = self._retry_non_gp_job_by_page(chunk_text, chunk_label, job)
            if fallback.get("reactions"):
                return {
                    "reactions": fallback.get("reactions") or [],
                    "audit_recovered": int(fallback.get("audit_recovered") or 0),
                    "error": None,
                    "job_log": {
                        "job_id": job.get("job_id"),
                        "mode": job.get("mode"),
                        "prompt_mode": prompt_mode,
                        "gp_keys": [],
                        "reaction_count": len(fallback.get("reactions") or []),
                        "reaction_ids": [
                            str(reaction.get("id"))
                            for reaction in fallback.get("reactions") or []
                            if isinstance(reaction, dict) and reaction.get("id")
                        ],
                        "attempts": attempts,
                        "fallback": fallback.get("job_logs") or [],
                        "fallback_recovered_count": len(fallback.get("reactions") or []),
                    },
                }

        return {
            "reactions": [],
            "audit_recovered": 0,
            "error": error,
            "job_log": {
                "job_id": job.get("job_id"),
                "mode": job.get("mode"),
                "prompt_mode": prompt_mode,
                "gp_keys": list(job.get("gp_keys") or []),
                "reaction_count": 0,
                "reaction_ids": [],
                "attempts": attempts,
                "fallback": (fallback or {}).get("job_logs") or [],
            },
        }

    def _split_stage2_text_by_page(self, chunk_text: str) -> List[Tuple[str, str]]:
        pattern = re.compile(r'(?m)^---\s*Page\s+(\d+)\s*---\s*$')
        matches = list(pattern.finditer(chunk_text or ""))
        if not matches:
            return []
        pages = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(chunk_text)
            page_text = chunk_text[match.start():end].strip()
            pages.append((match.group(1), page_text))
        candidates = [(f"Page {page}", text) for page, text in pages if text]
        for index in range(len(pages) - 1):
            page_a, text_a = pages[index]
            page_b, text_b = pages[index + 1]
            joined = f"{text_a}\n{text_b}".strip()
            if joined:
                candidates.append((f"Pages {page_a}-{page_b}", joined))
        return candidates

    def _retry_non_gp_job_by_page(self, chunk_text: str, chunk_label: str, job: Dict) -> Dict:
        reactions_by_candidate = []
        job_logs = []
        audit_recovered = 0
        for suffix, sub_text in self._split_stage2_text_by_page(chunk_text):
            allowed_pages = self._source_page_numbers_from_chunk_text(sub_text)
            sub_job = dict(job)
            sub_job["job_id"] = f"{job.get('job_id')}_fallback_{len(job_logs)+1}"
            result = self._run_stage2_job(
                sub_text,
                f"{chunk_label} {suffix}",
                sub_job,
                gp_block="",
                allowed_page_nums=allowed_pages,
                allow_non_gp_split_fallback=False,
            )
            job_log = result.get("job_log") or {}
            job_log["fallback_source"] = suffix
            job_logs.append(job_log)
            audit_recovered += int(result.get("audit_recovered") or 0)
            if result.get("reactions"):
                reactions_by_candidate.append(result.get("reactions") or [])
        merged = self.merge_results(reactions_by_candidate) if reactions_by_candidate else []
        return {
            "reactions": merged,
            "audit_recovered": audit_recovered,
            "job_logs": job_logs,
        }

    def stage2_extract_with_meta(self, chunk_text: str, chunk_label: str,
                                 registry: Optional[Dict[str, str]] = None,
                                 gp_texts: Optional[Dict[str, str]] = None,
                                 gp_templates: Optional[Dict[str, Dict]] = None) -> Dict:
        """
        用精确模型从分块中提取反应数据

        Args:
            chunk_text: 分块文本
            chunk_label: 分块标签（用于日志）
            registry: symbol→name 映射表（可选）
            gp_texts: 原始GP文本 dict（可选），格式: {"GP标签": "GP原文..."}
        """
        allowed_page_nums = self._source_page_numbers_from_chunk_text(chunk_text)
        if not hasattr(self, "_gp_selection_lock"):
            self._gp_selection_lock = threading.Lock()
        with self._gp_selection_lock:
            selected_gp_texts = self.select_gp_for_chunk(chunk_text, gp_texts)
            gp_selection_debug = dict(getattr(self, "last_gp_selection_debug", {}) or {})

        all_job_reactions = []
        job_logs = []
        errors = []
        audit_recovered = 0
        selected_gp_keys = [
            key
            for key in (selected_gp_texts or {}).keys()
            if isinstance(key, str) and key
        ]
        if selected_gp_keys:
            dispatch_mode = "gp_template_forced_mixed"
            executable_jobs = [{
                "job_id": "mixed_1",
                "mode": "mixed",
                "gp_keys": selected_gp_keys,
                "source_pages": allowed_page_nums,
            }]
        else:
            dispatch_mode = "non_gp_only"
            executable_jobs = [{
                "job_id": "non_gp_1",
                "mode": "non_gp",
                "gp_keys": [],
                "source_pages": allowed_page_nums,
            }]
        dispatch_info = {
            "dispatch_mode": dispatch_mode,
            "selected_gp_keys": selected_gp_keys,
            "router_disabled": True,
        }

        for job in executable_jobs:
            gp_block = ""
            if job.get("mode") in {"gp", "mixed"}:
                gp_block, invalid_gp_labels = self._build_gp_block_for_job(job, gp_templates)
                if invalid_gp_labels:
                    gp_selection_debug["invalid_gp_templates"] = invalid_gp_labels
                    error = "invalid_gp_template: " + ", ".join(invalid_gp_labels)
                    job_logs.append({
                        "job_id": job.get("job_id"),
                        "mode": job.get("mode"),
                        "prompt_mode": "mixed_gp_non_gp" if job.get("mode") == "mixed" else "gp_template",
                        "gp_keys": list(job.get("gp_keys") or []),
                        "reaction_count": 0,
                        "reaction_ids": [],
                        "error": error,
                    })
                    dispatch_info["invalid_gp_templates"] = invalid_gp_labels
                    return {
                        "reactions": [],
                        "audit_recovered": 0,
                        "gp_selection_debug": gp_selection_debug,
                        "dispatch": dispatch_info,
                        "jobs": job_logs,
                        "error": error,
                    }

            result = self._run_stage2_job(
                chunk_text,
                chunk_label,
                job,
                gp_block=gp_block,
                allowed_page_nums=allowed_page_nums,
                gp_templates=gp_templates,
            )
            job_log = result.get("job_log") or {}
            if result.get("error"):
                job_log["error"] = result.get("error")
                errors.append(f"{job.get('job_id')}: {result.get('error')}")
            job_logs.append(job_log)
            audit_recovered += int(result.get("audit_recovered") or 0)
            if result.get("reactions"):
                all_job_reactions.append(result.get("reactions") or [])

        reactions = self.merge_results(all_job_reactions) if all_job_reactions else []
        if errors and not reactions:
            print(f"  [ERROR] Stage2 最终失败 ({chunk_label})")
            error_value = "stage2_failed_after_retries"
        elif errors:
            error_value = "; ".join(errors)
        else:
            error_value = None

        return {
            "reactions": reactions,
            "audit_recovered": audit_recovered,
            "gp_selection_debug": gp_selection_debug,
            "dispatch": dispatch_info,
            "jobs": job_logs,
            "error": error_value,
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

    def _reaction_id_prefix(self, reaction: Dict) -> str:
        """Return the stable id prefix used for final sequential numbering."""
        gp_source = str(reaction.get("_gp_source") or "").strip()
        if gp_source:
            return gp_source
        return "NonGP"

    def _renumber_reaction_ids(self, reactions: List[Dict]) -> List[Dict]:
        """Make final reaction ids unique and sequential without filtering reactions."""
        counters = {}
        renumbered = []
        for reaction in reactions:
            if not isinstance(reaction, dict):
                renumbered.append(reaction)
                continue
            new_reaction = dict(reaction)
            original_id = str(new_reaction.get("id") or "").strip()
            if original_id and not new_reaction.get("_original_id"):
                new_reaction["_original_id"] = original_id
            prefix = self._reaction_id_prefix(new_reaction)
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

    def _gp_template_for_reaction(
        self,
        reaction: Dict,
        gp_templates: Dict[str, Dict],
    ) -> Tuple[Optional[str], Optional[Dict]]:
        explicit = str(reaction.get("_gp_source") or "").strip()
        if explicit in gp_templates:
            template = gp_templates.get(explicit)
            return explicit, template if isinstance(template, dict) else None
        identifiers = " ".join(
            str(reaction.get(key) or "") for key in ("id", "_original_id")
        )
        for gp_key, template in gp_templates.items():
            if gp_key in identifiers or self._text_contains_gp_alias(identifiers, gp_key):
                return gp_key, template if isinstance(template, dict) else None
        return None, None

    @staticmethod
    def _merge_gp_override(base: Any, override: Any) -> Any:
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override
        merged = dict(base)
        for key, value in override.items():
            if value not in (None, "", [], {}):
                merged[key] = value
        return merged

    def apply_gp_templates(
        self,
        reactions: List[Dict],
        gp_templates: Dict[str, Dict],
    ) -> List[Dict]:
        """Materialize authoritative GP shared fields into concrete reactions."""
        if not gp_templates:
            return reactions
        materialized = []
        for reaction in reactions:
            if not isinstance(reaction, dict):
                materialized.append(reaction)
                continue
            gp_key, template = self._gp_template_for_reaction(reaction, gp_templates)
            if not gp_key:
                materialized.append(reaction)
                continue
            if not isinstance(template, dict) or template.get("status") != "valid":
                raise ValueError(f"Reaction references invalid GP template: {gp_key}")

            effective = dict(reaction)
            effective["_gp_source"] = gp_key
            effective["_gp_template_sha256"] = template.get("raw_text_sha256")
            overrides = effective.pop("_entry_overrides", {})
            if not isinstance(overrides, dict):
                overrides = {}

            if template.get("reaction_type"):
                effective["reaction_type"] = template["reaction_type"]
            if not effective.get("substrates") and template.get("substrates"):
                effective["substrates"] = [dict(item) for item in template["substrates"]]
            if "intermediates" in template and not effective.get("intermediates"):
                effective["intermediates"] = [
                    dict(item) for item in template.get("intermediates", [])
                ]

            for field in ("catalysts", "ligands", "other_components"):
                value = [dict(item) for item in template.get(field, [])]
                if isinstance(overrides.get(field), list):
                    value = overrides[field]
                effective[field] = value

            conditions = dict(template.get("conditions") or {})
            if isinstance(overrides.get("conditions"), dict):
                conditions = self._merge_gp_override(conditions, overrides["conditions"])
            effective["conditions"] = conditions

            if template.get("step_count") is not None:
                effective["step_count"] = template["step_count"]
                for field, default_step in (
                    ("substrates", 1),
                    ("products", int(template["step_count"])),
                ):
                    for item in effective.get(field) or []:
                        if isinstance(item, dict) and item.get("step") is None:
                            item["step"] = default_step
            else:
                effective.pop("step_count", None)
                effective = self._single_step_schema(effective)

            effective.pop("additives", None)
            effective.pop("reagents", None)
            materialized.append(effective)
        return materialized

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
        self.last_substrate_name_resolution_reviews = []
        self.last_substrate_name_resolution_stats = {"resolved": 0, "reviewed": 0}
        self.last_generic_substrate_resolution_reviews = []
        self.last_generic_substrate_resolution_stats = {
            "candidates": 0,
            "llm_calls": 0,
            "candidate_substrates": 0,
            "candidate_products": 0,
            "resolved": 0,
            "reviewed": 0,
        }

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
                "schema": "chunk_extraction_log_v2",
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
                    "stage2_error_count": sum(
                        1 for item in chunk_extraction_log
                        if item.get("stage2", {}).get("error")
                    ),
                    "invalid_gp_template_chunk_count": sum(
                        1 for item in chunk_extraction_log
                        if str(item.get("stage2", {}).get("error") or "").startswith("invalid_gp_template:")
                    ),
                    "stage2_job_error_count": sum(
                        1
                        for item in chunk_extraction_log
                        for job in item.get("stage2", {}).get("jobs", [])
                        if job.get("error")
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
                "general_procedure_templates": gp_templates,
                "entity_context_path": entity_context.get("_context_path"),
                "gp_template_log_path": entity_context.get("gp_template_log_path"),
                "chunk_extraction_log_path": str(chunk_log_path),
                "stats": stats_snapshot,
                "substrate_name_resolution_reviews": list(
                    getattr(self, "last_substrate_name_resolution_reviews", []) or []
                ),
                "reactions": reactions,
                "metadata": entity_context.get("metadata"),
            }

        registry = entity_context.get("name_registry") or entity_context.get("symbol_name_mapping") or {}
        gp_texts = entity_context.get("general_procedures") or {}
        gp_templates = entity_context.get("general_procedure_templates") or {}
        file_stats['registry_size'] = len(registry)
        file_stats['gp_templates'] = sum(
            1 for template in gp_templates.values()
            if isinstance(template, dict) and template.get("status") == "valid"
        )

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
                        gp_templates=gp_templates,
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
                    if result.get("router"):
                        log_item["stage2"]["router"] = result.get("router")
                    if result.get("dispatch"):
                        log_item["stage2"]["dispatch"] = result.get("dispatch")
                    if result.get("jobs"):
                        log_item["stage2"]["jobs"] = result.get("jobs")
                    if result.get("error"):
                        log_item["stage2"]["error"] = result.get("error")
                    if result.get("gp_selection_debug"):
                        log_item["stage2"]["gp_selection"] = result.get("gp_selection_debug")
                all_chunk_results.append(reactions)
            _write_chunk_log()
        else:
            for chunk, _screen in relevant_chunks:
                label = f"{pdf_name} Chunk{chunk['chunk_id']} Pages[{chunk['page_range']}]"
                stage2_result = self.stage2_extract_with_meta(
                    chunk['text'],
                    label,
                    gp_texts=gp_texts,
                    gp_templates=gp_templates,
                )
                reactions = stage2_result.get("reactions") or []
                self.stage2_audit_recovered += int(stage2_result.get("audit_recovered") or 0)
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
                    if stage2_result.get("router"):
                        log_item["stage2"]["router"] = stage2_result.get("router")
                    if stage2_result.get("dispatch"):
                        log_item["stage2"]["dispatch"] = stage2_result.get("dispatch")
                    if stage2_result.get("jobs"):
                        log_item["stage2"]["jobs"] = stage2_result.get("jobs")
                    if stage2_result.get("error"):
                        log_item["stage2"]["error"] = stage2_result.get("error")
                    if stage2_result.get("gp_selection_debug"):
                        log_item["stage2"]["gp_selection"] = stage2_result.get("gp_selection_debug")
                all_chunk_results.append(reactions)
                _write_chunk_log()
                partial_merged = self.sanitize_reactions_schema(self.merge_results(all_chunk_results))
                partial_merged = self.validate_substrate_name_resolutions(partial_merged)
                _atomic_write_json(
                    output_file,
                    _build_output_payload(
                        partial_merged,
                        is_partial=True,
                        completed_stage2_chunks=len(all_chunk_results),
                    ),
                )

        stage2_errors = [
            {
                "chunk_id": item.get("chunk_id"),
                "page_range": item.get("page_range"),
                "error": item.get("stage2", {}).get("error"),
            }
            for item in chunk_extraction_log
            if item.get("stage2", {}).get("error")
        ]
        file_stats["stage2_error_count"] = len(stage2_errors)
        file_stats["invalid_gp_template_chunk_count"] = sum(
            1 for item in stage2_errors
            if str(item.get("error") or "").startswith("invalid_gp_template:")
        )
        file_stats["stage2_errors"] = stage2_errors

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
        merged = self.resolve_generic_substrates_from_products(merged)
        generic_resolution_stats = getattr(self, "last_generic_substrate_resolution_stats", {}) or {}
        file_stats["generic_substrate_resolution_candidates"] = generic_resolution_stats.get("candidates", 0)
        file_stats["generic_substrate_resolution_llm_calls"] = generic_resolution_stats.get("llm_calls", 0)
        file_stats["generic_substrate_resolution_candidate_substrates"] = generic_resolution_stats.get("candidate_substrates", 0)
        file_stats["generic_substrate_resolution_candidate_products"] = generic_resolution_stats.get("candidate_products", 0)
        merged = self.validate_substrate_name_resolutions(merged)
        substrate_resolution_stats = getattr(self, "last_substrate_name_resolution_stats", {}) or {}
        file_stats["substrate_names_resolved_from_product"] = substrate_resolution_stats.get("resolved", 0)
        file_stats["generic_substrates_resolved_from_product"] = substrate_resolution_stats.get("resolved", 0)
        file_stats["substrate_name_resolution_review_count"] = substrate_resolution_stats.get("reviewed", 0)
        file_stats["generic_substrate_resolution_review_count"] = substrate_resolution_stats.get("reviewed", 0)
        scaffold_mapping = entity_context.get("scaffold_substituent_mapping") or {}
        if scaffold_mapping and merged:
            merged = self.enrich_reactions_with_scaffold_mapping(merged, scaffold_mapping)
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

    def _is_generic_substrate_name(self, value: str) -> bool:
        """Conservatively identify class-level substrate placeholders."""
        text = self._normalize_registry_compare_name(value)
        text = re.sub(r"^(?:the|a|an)\s+", "", text).strip()
        if not text:
            return True
        generic_nouns = (
            "substrate", "starting material", "aniline", "amine", "amide",
            "alkene", "alkyne", "aldehyde", "ketone", "imine", "ester",
            "carboxylic acid", "alcohol", "phenol", "aryl halide",
        )
        if text in generic_nouns:
            return True
        qualifier = r"(?:corresponding|appropriate|desired|substituted|generic)"
        noun_pattern = "|".join(re.escape(noun) for noun in generic_nouns)
        if re.fullmatch(rf"{qualifier}\s+(?:{noun_pattern})(?:\s+derivative)?", text):
            return True
        if re.fullmatch(rf"(?:{noun_pattern})\s+derivative", text):
            return True
        if re.fullmatch(r"substrate(?:\s+(?:amide|amine|alkene|alkyne|\w{1,5}))?", text):
            return True
        return False

    def _specific_product_names(self, reaction: Dict) -> List[str]:
        names = []
        for product in reaction.get("products") or []:
            if isinstance(product, dict):
                name = str(product.get("name") or "").strip()
            else:
                name = str(product or "").strip()
            if (
                name
                and not self._is_generic_substrate_name(name)
                and not self._is_placeholder_name(name)
                and not self._is_label_only_product_name(name)
            ):
                names.append(name)
        return names

    def _is_label_only_product_name(self, value: str) -> bool:
        """Return true for product labels without a concrete chemical name."""
        text = self._normalize_registry_compare_name(value)
        if not text:
            return True
        label_prefixes = (
            "product", "compound", "entry", "example", "substrate",
            "oxindole", "amide", "material",
        )
        prefix_pattern = "|".join(re.escape(prefix) for prefix in label_prefixes)
        if re.fullmatch(rf"(?:{prefix_pattern})\s+[a-z]?\d+[a-z]?", text):
            return True
        if re.fullmatch(r"[a-z]?\d+[a-z]?", text):
            return True
        return False

    def _generic_substrate_resolution_review(
        self,
        reaction: Dict,
        substrate_index: Optional[int],
        original_name: str,
        product_names: List[str],
        reason: str,
        detail: str = "",
    ) -> Dict:
        review = {
            "reaction_id": str(reaction.get("id") or ""),
            "substrate_index": substrate_index,
            "original_name": original_name,
            "candidate_name": None,
            "product_names": product_names,
            "reason": reason,
        }
        if detail:
            review["detail"] = detail
        return review

    def _generic_substrate_resolution_candidate(self, reaction: Dict) -> Tuple[Optional[Dict], List[Dict]]:
        """Build a compact LLM payload for generic substrates with concrete product evidence."""
        reviews = []
        products = reaction.get("products") or []
        raw_product_names = [
            str(product.get("name") if isinstance(product, dict) else product or "").strip()
            for product in products
        ]
        raw_product_names = [name for name in raw_product_names if name]
        product_names = [
            name for name in raw_product_names
            if (
                not self._is_generic_substrate_name(name)
                and not self._is_placeholder_name(name)
                and not self._is_label_only_product_name(name)
            )
        ]

        generic_substrates = []
        for index, substrate in enumerate(reaction.get("substrates") or []):
            if not isinstance(substrate, dict):
                continue
            name = str(substrate.get("name") or "").strip()
            if not self._is_generic_substrate_name(name):
                continue
            if substrate.get("resolution_source") or substrate.get("resolution_method"):
                continue
            generic_substrates.append((index, substrate))

        if not generic_substrates:
            return None, reviews
        if not product_names:
            reason = "product_not_unique_or_not_specific"
            if raw_product_names and all(self._is_label_only_product_name(name) for name in raw_product_names):
                reason = "label_only_product_name"
            for index, substrate in generic_substrates:
                reviews.append(
                    self._generic_substrate_resolution_review(
                        reaction,
                        index,
                        str(substrate.get("name") or ""),
                        raw_product_names,
                        reason,
                    )
                )
            return None, reviews

        payload = {
            "reaction_id": str(reaction.get("id") or ""),
            "reaction_type": str(reaction.get("reaction_type") or ""),
            "gp_source": str(reaction.get("_gp_source") or ""),
            "substrates": [
                {
                    "index": index,
                    "name": str(substrate.get("name") or ""),
                    "symbol": substrate.get("symbol"),
                    "amount": substrate.get("amount"),
                    "step": substrate.get("step"),
                }
                for index, substrate in generic_substrates
            ],
            "products": [{"name": name} for name in product_names],
        }
        return payload, reviews

    def _call_generic_substrate_resolution_llm(self, payload: Dict) -> Dict:
        response = self.client.chat.completions.create(
            model=self.extract_model,
            messages=[
                {"role": "system", "content": self.GENERIC_SUBSTRATE_RESOLUTION_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        raw_preview = content[:500]
        try:
            parsed = self._parse_json_with_trailing_text(content)
        except Exception as exc:
            raise ValueError(
                "generic substrate resolution JSON parse failed: "
                f"{type(exc).__name__}: {exc}; raw_response_preview={raw_preview!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                "generic substrate resolution response must be a JSON object; "
                f"raw_response_preview={raw_preview!r}"
            )
        if not isinstance(parsed.get("resolutions"), list):
            raise ValueError(
                "generic substrate resolution response must contain a resolutions list; "
                f"raw_response_preview={raw_preview!r}"
            )
        return parsed

    def resolve_generic_substrates_from_products(self, reactions: List[Dict]) -> List[Dict]:
        """Use a focused LLM node to resolve generic substrates from concrete product names."""
        resolved_reactions = []
        reviews = []
        stats = {
            "candidates": 0,
            "llm_calls": 0,
            "candidate_substrates": 0,
            "candidate_products": 0,
            "resolved": 0,
            "reviewed": 0,
        }

        for reaction in reactions or []:
            if not isinstance(reaction, dict):
                resolved_reactions.append(reaction)
                continue
            new_reaction = json.loads(json.dumps(reaction, ensure_ascii=False))
            payload, candidate_reviews = self._generic_substrate_resolution_candidate(new_reaction)
            reviews.extend(candidate_reviews)
            if not payload:
                resolved_reactions.append(new_reaction)
                continue

            stats["candidates"] += 1
            stats["candidate_substrates"] += len(payload.get("substrates") or [])
            stats["candidate_products"] += len(payload.get("products") or [])
            try:
                stats["llm_calls"] += 1
                llm_result = self._call_generic_substrate_resolution_llm(payload)
            except Exception as exc:
                product_names = [str(product.get("name") or "") for product in payload.get("products") or []]
                detail = f"{type(exc).__name__}: {exc}"[:500]
                for substrate in payload.get("substrates") or []:
                    reviews.append(
                        self._generic_substrate_resolution_review(
                            new_reaction,
                            substrate.get("index"),
                            str(substrate.get("name") or ""),
                            product_names,
                            "llm_resolution_error",
                            detail=detail,
                        )
                    )
                resolved_reactions.append(new_reaction)
                continue

            resolutions = llm_result.get("resolutions")
            if not isinstance(resolutions, list):
                resolutions = []
            substrates = new_reaction.get("substrates") or []
            payload_product_names = [
                str(product.get("name") or "").strip()
                for product in payload.get("products") or []
                if str(product.get("name") or "").strip()
            ]
            payload_product_set = {
                self._normalize_registry_compare_name(name)
                for name in payload_product_names
            }
            for resolution in resolutions:
                if not isinstance(resolution, dict):
                    continue
                substrate_index = resolution.get("substrate_index")
                if not isinstance(substrate_index, int) or substrate_index < 0 or substrate_index >= len(substrates):
                    reviews.append(
                        self._generic_substrate_resolution_review(
                            new_reaction,
                            substrate_index if isinstance(substrate_index, int) else None,
                            "",
                            payload_product_names,
                            "invalid_substrate_index",
                        )
                    )
                    continue
                substrate = substrates[substrate_index]
                if not isinstance(substrate, dict):
                    continue
                if not resolution.get("can_resolve"):
                    reviews.append(
                        self._generic_substrate_resolution_review(
                            new_reaction,
                            substrate_index,
                            str(substrate.get("name") or ""),
                            payload_product_names,
                            "llm_could_not_resolve",
                            detail=str(resolution.get("reason") or "")[:500],
                        )
                    )
                    continue
                resolved_name = str(resolution.get("resolved_name") or "").strip()
                if not resolved_name:
                    reviews.append(
                        self._generic_substrate_resolution_review(
                            new_reaction,
                            substrate_index,
                            str(substrate.get("name") or ""),
                            payload_product_names,
                            "empty_resolved_name",
                        )
                    )
                    continue
                original_name = str(resolution.get("original_name") or substrate.get("name") or "").strip()
                evidence = resolution.get("resolution_evidence")
                if not isinstance(evidence, dict):
                    evidence = {}
                evidence_product = str(evidence.get("product_name") or "").strip()
                if not evidence_product and len(payload_product_names) == 1:
                    evidence_product = payload_product_names[0]
                    evidence["product_name"] = evidence_product
                normalized_evidence = self._normalize_registry_compare_name(evidence_product)
                if not evidence_product or normalized_evidence not in payload_product_set:
                    reviews.append(
                        self._generic_substrate_resolution_review(
                            new_reaction,
                            substrate_index,
                            original_name,
                            payload_product_names,
                            "missing_or_invalid_evidence_product",
                            detail=str(resolution.get("reason") or evidence.get("reason") or "")[:500],
                        )
                    )
                    continue

                substrate["name"] = resolved_name
                substrate["original_name"] = original_name
                substrate["resolution_source"] = "product_name"
                substrate["resolution_method"] = "gp_product_to_substrate_mapping"
                substrate["resolution_confidence"] = str(
                    resolution.get("resolution_confidence") or "high"
                ).strip() or "high"
                substrate["resolution_evidence"] = evidence
                stats["resolved"] += 1
            new_reaction["substrates"] = substrates
            resolved_reactions.append(new_reaction)

        stats["reviewed"] = len(reviews)
        self.last_generic_substrate_resolution_reviews = reviews
        self.last_generic_substrate_resolution_stats = stats
        return resolved_reactions

    def validate_substrate_name_resolutions(self, reactions: List[Dict]) -> List[Dict]:
        """Accept only auditable, high-confidence product-derived completions."""
        validated = []
        reviews = list(getattr(self, "last_generic_substrate_resolution_reviews", []) or [])
        existing_review_keys = {
            (review.get("reaction_id"), review.get("substrate_index"))
            for review in reviews
            if isinstance(review, dict)
        }
        resolved_count = 0
        resolution_keys = (
            "original_name", "resolution_source", "resolution_method",
            "resolution_confidence", "resolution_evidence",
        )

        for reaction in reactions or []:
            if not isinstance(reaction, dict):
                validated.append(reaction)
                continue
            new_reaction = dict(reaction)
            product_names = self._specific_product_names(new_reaction)
            normalized_products = {
                self._normalize_registry_compare_name(name): name
                for name in product_names
            }
            new_substrates = []
            for index, substrate in enumerate(new_reaction.get("substrates") or []):
                if not isinstance(substrate, dict):
                    new_substrates.append(substrate)
                    continue
                item = dict(substrate)
                source = str(item.get("resolution_source") or "").strip()
                method = str(item.get("resolution_method") or "").strip()
                is_product_resolution = (
                    source == "product_name"
                    or method == "gp_product_to_substrate_mapping"
                )
                attempted_review = False

                if is_product_resolution:
                    original_name = str(item.get("original_name") or "").strip()
                    candidate_name = str(item.get("name") or "").strip()
                    confidence = str(item.get("resolution_confidence") or "").strip().casefold()
                    evidence = item.get("resolution_evidence")
                    if isinstance(evidence, dict):
                        evidence_product = str(evidence.get("product_name") or "").strip()
                    else:
                        evidence_product = str(evidence or "").strip()
                    normalized_evidence = self._normalize_registry_compare_name(evidence_product)

                    reason = ""
                    if not self._is_generic_substrate_name(original_name):
                        reason = "original_name_is_not_generic"
                    elif not candidate_name or self._is_generic_substrate_name(candidate_name):
                        reason = "candidate_name_is_not_specific"
                    elif confidence != "high":
                        reason = "resolution_confidence_is_not_high"
                    elif source != "product_name" or method != "gp_product_to_substrate_mapping":
                        reason = "invalid_product_resolution_metadata"
                    elif not normalized_evidence or normalized_evidence not in normalized_products:
                        reason = "evidence_product_not_found_in_reaction"
                    elif self._normalize_registry_compare_name(candidate_name) in normalized_products:
                        reason = "candidate_copies_complete_product_name"

                    if reason:
                        attempted_review = True
                        reviews.append({
                            "reaction_id": str(new_reaction.get("id") or ""),
                            "substrate_index": index,
                            "original_name": original_name,
                            "candidate_name": candidate_name,
                            "product_names": product_names,
                            "reason": reason,
                        })
                        item["name"] = original_name or "substrate"
                        for key in resolution_keys:
                            item.pop(key, None)
                    else:
                        item["original_name"] = original_name
                        item["resolution_source"] = "product_name"
                        item["resolution_method"] = "gp_product_to_substrate_mapping"
                        item["resolution_confidence"] = "high"
                        item["resolution_evidence"] = {"product_name": normalized_products[normalized_evidence]}
                        resolved_count += 1

                if (
                    not attempted_review
                    and not is_product_resolution
                    and self._is_generic_substrate_name(str(item.get("name") or ""))
                    and product_names
                ):
                    review_key = (str(new_reaction.get("id") or ""), index)
                    if review_key not in existing_review_keys:
                        reviews.append({
                            "reaction_id": str(new_reaction.get("id") or ""),
                            "substrate_index": index,
                            "original_name": str(item.get("name") or ""),
                            "candidate_name": None,
                            "product_names": product_names,
                            "reason": "no_high_confidence_resolution_proposed",
                        })
                new_substrates.append(item)
            new_reaction["substrates"] = new_substrates
            validated.append(new_reaction)

        self.last_substrate_name_resolution_reviews = reviews
        self.last_substrate_name_resolution_stats = {
            "resolved": resolved_count,
            "reviewed": len(reviews),
        }
        return validated

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
        allow_generic_name: bool = False,
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
        product_inferred = (
            str(new_item.get("resolution_source") or "") == "product_name"
            or str(new_item.get("resolution_method") or "") == "gp_product_to_substrate_mapping"
        )
        if (
            product_inferred
            or self._should_registry_overwrite_name(raw_name, symbol, registry, normalized_registry)
            or (allow_generic_name and self._is_generic_substrate_name(raw_name))
        ):
            original_name = str(new_item.get("original_name") or raw_name).strip()
            new_item["name"] = name
            new_item["symbol"] = raw_symbol or symbol
            new_item["resolution_source"] = "name_registry"
            new_item["resolution_method"] = "same_paper_symbol"
            new_item["resolution_confidence"] = "high"
            if allow_generic_name and self._is_generic_substrate_name(original_name):
                new_item["original_name"] = original_name
            new_item.pop("resolution_evidence", None)
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
        fields = [
            'substrates', 'products', 'intermediates', 'catalysts',
            'other_components', 'additives', 'reagents',
        ]
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
                            allow_generic_name=(field == "substrates"),
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
        self.last_substrate_name_resolution_reviews = []
        self.last_substrate_name_resolution_stats = {"resolved": 0, "reviewed": 0}
        self.last_generic_substrate_resolution_reviews = []
        self.last_generic_substrate_resolution_stats = {
            "candidates": 0,
            "llm_calls": 0,
            "candidate_substrates": 0,
            "candidate_products": 0,
            "resolved": 0,
            "reviewed": 0,
        }
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
                gp_templates = self.build_gp_templates(
                    gp_texts,
                    source_pages_by_gp=getattr(self, "last_gp_source_pages", {}) or {},
                )
                print(f"  找到 {len(gp_texts)} 个 GP 段落")
                file_stats['gp_templates'] = sum(
                    1 for template in gp_templates.values()
                    if isinstance(template, dict) and template.get("status") == "valid"
                )
                
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
                gp_templates = {}
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
                reactions = self.stage2_extract(
                    chunk['text'], label, gp_texts=gp_texts, gp_templates=gp_templates
                )
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
            merged = self.resolve_generic_substrates_from_products(merged)
            generic_resolution_stats = getattr(self, "last_generic_substrate_resolution_stats", {}) or {}
            file_stats["generic_substrate_resolution_candidates"] = generic_resolution_stats.get("candidates", 0)
            file_stats["generic_substrate_resolution_llm_calls"] = generic_resolution_stats.get("llm_calls", 0)
            file_stats["generic_substrate_resolution_candidate_substrates"] = generic_resolution_stats.get("candidate_substrates", 0)
            file_stats["generic_substrate_resolution_candidate_products"] = generic_resolution_stats.get("candidate_products", 0)
            merged = self.validate_substrate_name_resolutions(merged)
            substrate_resolution_stats = getattr(self, "last_substrate_name_resolution_stats", {}) or {}
            file_stats["substrate_names_resolved_from_product"] = substrate_resolution_stats.get("resolved", 0)
            file_stats["generic_substrates_resolved_from_product"] = substrate_resolution_stats.get("resolved", 0)
            file_stats["substrate_name_resolution_review_count"] = substrate_resolution_stats.get("reviewed", 0)
            file_stats["generic_substrate_resolution_review_count"] = substrate_resolution_stats.get("reviewed", 0)
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
                "general_procedure_templates": gp_templates,
                "stats": file_stats,
                "substrate_name_resolution_reviews": list(
                    getattr(self, "last_substrate_name_resolution_reviews", []) or []
                ),
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
