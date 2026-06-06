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
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

# 确保能导入同目录下的pdf_to_gpt_extractor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_to_gpt_extractor import PDFReactionExtractor, PDF_LIBRARY


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
    GP_TITLE_PATTERNS = [
        r'(?i)(general\s+procedure)(?:\s+for\s+synthesis\s+of\s+([\w\d\s,\-–and]+))?\s*([A-Z])?\b',
        r'(?i)(representative\s+procedure)\s*[A-Z]?',
        r'(?i)(typical\s+procedure)\s*[A-Z]?',
        r'(?i)(standard\s+procedure)\s*[A-Z]?',
        r'(?i)(standard\s+conditions)\s*[A-Z]?',
        r'(?i)(experimental\s+procedure)\s+for',
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
        'following the', 'as described', 'following general',
    ]

    # =====================================================================
    # 未使用代码 (已弃用: 正则匹配Registry)
    # =====================================================================
    # NAME_PATTERNS 常量已弃用，现在使用GPT分块提取Registry
    # def _find_registry_pages() 和 _extract_registry_regex() 也已弃用
    
    # Stage1 screening prompt
    SCREEN_PROMPT = """You are a chemistry data screener. 
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

    STAGE2_AUDIT_PROMPT = """You are auditing a chemistry SI reaction extraction.
Compare the source text against the current extracted reaction JSON.

Return ONLY a JSON object:
{"missing_reactions": [<reaction objects>]}

Rules:
- Do not rewrite reactions already present in current_extraction.
- Add only reaction entries that are clearly present in the source text but missing from current_extraction.
- Product characterization entries after a GP are valid reaction entries when they report an isolated product and yield.
- "Prepared according to General Procedure A/B using ..." entries are valid reaction entries.
- For GP-referenced entries, use the supplied GP context for substrates/reagents/conditions, but do not copy product names into substrates.
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

    def __init__(self, api_key: Optional[str] = None,
                 pages_per_chunk: int = 5,
                 screen_model: str = "gpt-5-mini",
                 extract_model: str = "gpt-5-mini",
                 base_url: str = "https://oneapi.xty.app/v1",
                 enable_stage2_audit: bool = True):
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
        self.stage2_audit_recovered = 0
        self.last_registry_validation_stats = {
            "registry_raw_count": 0,
            "registry_grounded_count": 0,
            "registry_removed_count": 0,
            "registry_removed_entries": [],
        }
        self._last_registry_chunk_validation_stats = dict(self.last_registry_validation_stats)
        self.stats = {
            'total_pages': 0,
            'filtered_pages': 0,
            'total_chunks': 0,
            'screened_chunks': 0,
            'extracted_chunks': 0,
            'skipped_chunks': 0,
        }
    
    # =====================================================================
    # 第一层：按页提取文本
    # =====================================================================

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

        # 检查是否需要跳过（参考文献、致谢等）
        for pattern in self.SKIP_PATTERNS:
            if re.search(pattern, text_lower):
                return False

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

        # 阈值：总分 >= 5 视为相关页
        return (quant_score + general_score) >= 5

    def filter_relevant_pages(self, pages: List[Dict]) -> List[Dict]:
        """
        过滤页面，只保留化学相关页

        Returns:
            过滤后的页列表
        """
        filtered = [p for p in pages if self.is_page_relevant(p['text'])]
        return filtered

    # =====================================================================
    # Stage 0: Name Registry — 符号→完整化学名称映射
    # =====================================================================

    def _chunk_pages_for_registry(self, pages: List[Dict], pages_per_chunk: int = 5) -> List[Dict]:
        """将页面按指定数量分块（用于registry提取）
        
        Args:
            pages: 页列表
            pages_per_chunk: 每块页数
            
        Returns:
            [{"chunk_id": 1, "text": "...", "page_nums": [1,2,...]}, ...]
        """
        chunks = []
        for i in range(0, len(pages), pages_per_chunk):
            chunk_pages = pages[i:i + pages_per_chunk]
            page_nums = [p['page_num'] for p in chunk_pages]
            chunk_text = "\n\n".join(
                f"--- Page {p['page_num']} ---\n{p['text']}"
                for p in chunk_pages
            )
            chunks.append({
                "chunk_id": len(chunks) + 1,
                "page_nums": page_nums,
                "text": chunk_text,
            })
        return chunks

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

        策略：GPT分块提取 → 合并去重 → 清理

        Args:
            pages: 按页提取的文本列表
            max_scan_pages: 最多扫描前N页
            pages_per_chunk: 每块页数

        Returns:
            {"1a": "(E)-3-(4-methoxyphenyl)-...", "2a": "...", "C1": "..."}
        """
        if not pages:
            return {}
        
        scan_pages = pages[:max_scan_pages]
        
        # --- Step 1: 页面分块 ---
        print(f"  [Registry Step 1] 页面分块: {len(scan_pages)}页 分 {pages_per_chunk}页/块")
        chunks = self._chunk_pages_for_registry(scan_pages, pages_per_chunk)
        print(f"  [Registry Step 2] 分为 {len(chunks)} 个分块，分别调用GPT提取...")
        
        # --- Step 2: GPT分块提取 ---
        all_registries = []
        validation_stats = {
            "registry_raw_count": 0,
            "registry_grounded_count": 0,
            "registry_removed_count": 0,
            "registry_removed_entries": [],
        }
        for chunk in chunks:
            print(f"    处理分块{chunk['chunk_id']}: 页 {chunk['page_nums']}")
            registry = self.extract_registry_with_gpt(chunk['text'])
            chunk_stats = getattr(self, "_last_registry_chunk_validation_stats", {}) or {}
            validation_stats["registry_raw_count"] += int(chunk_stats.get("registry_raw_count", 0) or 0)
            validation_stats["registry_grounded_count"] += int(chunk_stats.get("registry_grounded_count", 0) or 0)
            validation_stats["registry_removed_count"] += int(chunk_stats.get("registry_removed_count", 0) or 0)
            validation_stats["registry_removed_entries"].extend(
                chunk_stats.get("registry_removed_entries", []) or []
            )
            if registry:
                print(f"      提取到 {len(registry)} 条映射")
                all_registries.append(registry)
            else:
                print(f"      未提取到映射")
        
        # --- Step 3: 合并去重 ---
        print(f"  [Registry Step 3] 合并 {len(all_registries)} 个分块的结果...")
        merged_registry = self._merge_registry_results(all_registries)
        
        # --- Step 4: 清理通用术语映射 ---
        print(f"  [Registry Step 4] 清理通用术语...")
        registry = self._clean_registry(merged_registry)
        validation_stats["registry_removed_entries"] = validation_stats["registry_removed_entries"][:20]
        self.last_registry_validation_stats = validation_stats

        return registry

    # 泛指术语集合（不能作为化学名称）
    _GENERIC_TERMS = {
        'photocatalyst', 'catalyst', 'catalyst_name', 'ligand', 'chiral ligand',
        'solvent', 'reagent', 'additive', 'product', 'substrate', 'derivative',
        'side product', 'desired product', 'main product', 'intermediate',
        'alkene', 'alcohol', 'ketone', 'ester', 'aldehyde', 'amine', 'amide',
        'acid', 'ether', 'aromatic', 'olefin', 'enone', 'epoxide', 'heterocycle',
        'diene', 'enol', 'hydrocarbon', 'compound', 'material', 'species',
        'molecule', 'fragment', 'moiety', 'group', 'ring', 'chain',
        'ru(bpy)3', 'ir(ppy)3',
    }

    # 泛指描述词前缀（用于检测 "descriptor + symbol" 模式）
    _GENERIC_PREFIXES = [
        'alkene', 'alcohol', 'ketone', 'ester', 'aldehyde', 'amine', 'amide',
        'acid', 'ether', 'aromatic', 'olefin', 'enone', 'epoxide', 'heterocycle',
        'diene', 'enol', 'hydrocarbon', 'compound', 'substrate', 'product',
        'catalyst', 'photocatalyst', 'ligand', 'chiral ligand', 'reagent',
        'additive', 'solvent', 'intermediate', 'derivative', 'material',
        'fragment', 'moiety',
    ]

    def _clean_registry(self, registry: Dict[str, str]) -> Dict[str, str]:
        """
        删除映射到泛指术语或不合规名称的条目

        过滤规则:
        1. name 是泛指术语（如 alkene, catalyst）
        2. name 是 "泛指词 + 代号" 模式（如 alkene 2s, aldehyde 5b）
        3. name 末尾包含 symbol（说明可能是泛指+代号）
        4. name 长度太短（< 15 字符，大概率不是完整化学名）
        5. name 等于 symbol 本身
        """
        # 预编译 "descriptor + symbol" 模式正则
        # 匹配：泛指词 + 可选括号 + 符号（如 "alkene 2s", "alcohol (1a)", "ketone 5b"）
        generic_prefix_pattern = '|'.join(re.escape(p) for p in self._GENERIC_PREFIXES)
        desc_symbol_re = re.compile(
            rf'^\s*(?:{generic_prefix_pattern})\s*\(?\s*[\w\d]{{1,5}}\s*\)?$',
            re.IGNORECASE
        )

        cleaned = {}
        removed = []

        for symbol, name in registry.items():
            name_str = str(name).strip()
            name_lower = name_str.lower()

            # 规则1: 泛指术语
            if name_lower in self._GENERIC_TERMS:
                removed.append((symbol, name, "泛指术语"))
                continue

            # 规则5: name 等于 symbol
            if name_str == str(symbol).strip():
                removed.append((symbol, name, "name=symbol"))
                continue

            # 规则2: "泛指词 + 代号" 模式
            if desc_symbol_re.match(name_str):
                removed.append((symbol, name, "泛指词+代号模式"))
                continue

            # 规则3: name 以 symbol 结尾且前面是泛指词
            # 如 name="alkene 2s", symbol="2s"
            if str(symbol).strip() in name_str:
                # 去掉 symbol 后的部分是否是泛指词
                without_symbol = name_str.replace(str(symbol).strip(), '').strip()
                if without_symbol.lower() in {p.lower() for p in self._GENERIC_PREFIXES}:
                    removed.append((symbol, name, "泛指词+symbol"))
                    continue

            # 规则4: 长度太短（完整化学名通常 > 15 字符）
            if len(name_str) < 15:
                removed.append((symbol, name, f"太短({len(name_str)}字符)"))
                continue

            cleaned[symbol] = name

        if removed:
            removed_str = ", ".join(f"'{s}' -> '{n}' [{reason}]" for s, n, reason in removed)
            print(f"  [Registry Cleanup] 移除 {len(removed)} 条不合规映射: {removed_str}")

        return cleaned

    _REGISTRY_DEFINITION_TERMS = {
        "compound",
        "substrate",
        "product",
        "ligand",
        "catalyst",
        "denoted",
        "named",
        "abbreviated",
        "corresponding",
    }
    _REGISTRY_RESULT_TERMS = {
        "afforded",
        "gave",
        "yielded",
        "obtained",
        "isolated",
    }

    def _normalize_registry_text(self, text: str) -> str:
        """Normalize source text for registry grounding checks."""
        if text is None:
            return ""
        normalized = str(text).lower()
        normalized = normalized.replace("\u00ad", "")
        normalized = re.sub(r"[\u2010-\u2015\u2212]", "-", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"\s*-\s*", "-", normalized)
        return normalized.strip()

    def _find_normalized_spans(self, source_text: str, needle: str) -> List[tuple]:
        source_norm = self._normalize_registry_text(source_text)
        needle_norm = self._normalize_registry_text(needle)
        if not source_norm or not needle_norm:
            return []
        return [(m.start(), m.end()) for m in re.finditer(re.escape(needle_norm), source_norm)]

    def _find_symbol_spans(self, source_text: str, symbol: str) -> List[tuple]:
        source_norm = self._normalize_registry_text(source_text)
        symbol_norm = self._normalize_registry_text(symbol)
        if not source_norm or not symbol_norm:
            return []
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(symbol_norm)}(?![a-z0-9])")
        return [(m.start(), m.end()) for m in pattern.finditer(source_norm)]

    def _is_definition_window(self, window: str) -> bool:
        return any(re.search(rf"\b{re.escape(term)}\b", window) for term in self._REGISTRY_DEFINITION_TERMS)

    def _is_result_window(self, window: str) -> bool:
        return any(re.search(rf"\b{re.escape(term)}\b", window) for term in self._REGISTRY_RESULT_TERMS)

    def _is_strong_symbol_name_binding(self, symbol: str, name: str, source_text: str) -> bool:
        source_norm = self._normalize_registry_text(source_text)
        name_spans = self._find_normalized_spans(source_text, name)
        symbol_spans = self._find_symbol_spans(source_text, symbol)
        if not source_norm or not name_spans or not symbol_spans:
            return False

        for name_start, name_end in name_spans:
            for symbol_start, symbol_end in symbol_spans:
                if name_end <= symbol_start:
                    separator = source_norm[name_end:symbol_start]
                    if len(separator) <= 20 and re.fullmatch(r"[\s\(\)\[\]\{\}:,;=\-]*", separator):
                        return True
                elif symbol_end <= name_start:
                    separator = source_norm[symbol_end:name_start]
                    prefix = source_norm[max(0, symbol_start - 40):symbol_start]
                    if len(separator) <= 20:
                        has_definition_separator = bool(re.search(r"[:=]", separator))
                        has_definition_prefix = bool(
                            re.search(r"\b(?:compound|substrate|product|ligand|catalyst)\s*$", prefix)
                        )
                        has_named_separator = bool(
                            re.search(r"\b(?:is|named|denoted|abbreviated(?: as)?)\b", separator)
                        )
                        if has_definition_separator or has_definition_prefix or has_named_separator:
                            return True
        return False

    def _is_weak_symbol_name_binding(self, symbol: str, name: str, source_text: str) -> bool:
        source_norm = self._normalize_registry_text(source_text)
        name_spans = self._find_normalized_spans(source_text, name)
        symbol_spans = self._find_symbol_spans(source_text, symbol)
        if not source_norm or not name_spans or not symbol_spans:
            return False

        for name_start, name_end in name_spans:
            for symbol_start, symbol_end in symbol_spans:
                distance = max(symbol_start, name_start) - min(symbol_end, name_end)
                if distance > 300:
                    continue
                window_start = max(0, min(name_start, symbol_start) - 80)
                window_end = min(len(source_norm), max(name_end, symbol_end) + 80)
                window = source_norm[window_start:window_end]
                if self._is_result_window(window):
                    continue
                if self._is_definition_window(window):
                    return True
        return False

    def _validate_registry_against_source(self, registry: Dict[str, str], source_text: str):
        grounded = {}
        removed = []

        for symbol, name in (registry or {}).items():
            symbol_str = str(symbol).strip()
            name_str = str(name).strip()

            if not symbol_str:
                removed.append({"symbol": symbol_str, "name": name_str, "reason": "empty_symbol"})
                continue
            if not self._find_symbol_spans(source_text, symbol_str):
                removed.append({"symbol": symbol_str, "name": name_str, "reason": "symbol_not_in_source"})
                continue
            if not self._find_normalized_spans(source_text, name_str):
                removed.append({"symbol": symbol_str, "name": name_str, "reason": "name_not_in_source"})
                continue
            if self._is_strong_symbol_name_binding(symbol_str, name_str, source_text):
                grounded[symbol_str] = name_str
                continue
            if self._is_weak_symbol_name_binding(symbol_str, name_str, source_text):
                grounded[symbol_str] = name_str
                continue

            removed.append({"symbol": symbol_str, "name": name_str, "reason": "symbol_name_not_nearby"})

        return grounded, removed

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

    # 以下方法已弃用，现在使用GPT分块提取Registry
    # def _find_registry_pages(self, pages: List[Dict]) -> List[Dict]:
    #     """找出包含 symbol→name 模式的页面"""
    #     ...
    # def _extract_registry_regex(self, text: str) -> Dict[str, str]:
    #     """用正则从文本中提取 symbol→name 映射"""
    #     ...

    def extract_registry_with_gpt(self, text: str) -> Dict[str, str]:
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

            response = self.client.chat.completions.create(
                model=self.screen_model,
                messages=[
                    {"role": "system",
                     "content": "You are a chemistry data extractor. Always respond with valid JSON only."},
                    {"role": "user",
                     "content": f"{self.REGISTRY_PROMPT}\n\n{text}"}
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content.strip()
            raw_registry = self._parse_registry_response(raw)
            grounded_registry, removed_entries = self._validate_registry_against_source(raw_registry, text)
            self._last_registry_chunk_validation_stats = {
                "registry_raw_count": len(raw_registry),
                "registry_grounded_count": len(grounded_registry),
                "registry_removed_count": len(removed_entries),
                "registry_removed_entries": removed_entries[:20],
            }
            if removed_entries:
                removed_preview = ", ".join(
                    f"{entry.get('symbol')}[{entry.get('reason')}]" for entry in removed_entries[:5]
                )
                print(f"      [Registry Grounding] removed {len(removed_entries)} ungrounded mappings: {removed_preview}")
            return grounded_registry
        except Exception as e:
            print(f"  [WARN] GPT registry提取出错: {e}")
            self._last_registry_chunk_validation_stats = {
                "registry_raw_count": 0,
                "registry_grounded_count": 0,
                "registry_removed_count": 0,
                "registry_removed_entries": [],
            }
            return {}

    def _parse_registry_response(self, raw: str) -> Dict[str, str]:
        """解析GPT返回的registry JSON，解析时过滤明显不合规的条目"""
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
                    name_lower = name.lower()

                    # 基本校验：必须是字符串且长度>=5
                    if not isinstance(v, str) or len(name) < 5:
                        continue

                    # name 不能等于 symbol
                    if name == sym:
                        continue

                    # name 不能是泛指术语
                    if name_lower in self._GENERIC_TERMS:
                        continue

                    # name 不能以泛指描述词开头
                    if any(name_lower.startswith(p.lower() + ' ') for p in self._GENERIC_PREFIXES):
                        continue

                    result[sym] = name
                return result
        except json.JSONDecodeError:
            pass
        return {}

    # =====================================================================
    # Stage -1: General Procedure 文本提取
    # =====================================================================

    def _make_gp_key(self, title: str, gp_counter: Dict[str, int]) -> str:
        """
        根据 GP 标题生成 key
        例: 'General procedure for synthesis of 1-38:' → 'GeneralProcedure_1-38'
            'General Procedure A' → 'GeneralProcedureA'
            无标识时 → 'GeneralProcedureA', 'GeneralProcedureB', ...
        """
        # 尝试提取 scope (数字范围)
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
        letter_match = re.search(r'procedure\s+([A-Z])\b', title, re.IGNORECASE)
        if letter_match:
            return f"GeneralProcedure{letter_match.group(1).upper()}"

        # 无标识，按顺序编号
        if 'GeneralProcedure' not in gp_counter:
            gp_counter['GeneralProcedure'] = 0
        gp_counter['GeneralProcedure'] += 1
        idx = gp_counter['GeneralProcedure']
        suffix = chr(ord('A') + idx - 1) if idx <= 26 else str(idx)
        return f"GeneralProcedure{suffix}"

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

    def extract_general_procedure_texts(self, pages: List[Dict]) -> Dict[str, str]:
        """
        从全文提取所有 General Procedure 段落的原始文本

        Returns:
            {
                "GeneralProcedure_1-38": "A 25 mL Schlenk flask...",
                "GeneralProcedureA": "To a flame-dried vial...",
            }
        """
        # 拼接全文
        full_text = "\n".join(
            f"--- Page {p['page_num']} ---\n{p['text']}"
            for p in pages
        )

        # 修复跨行化学名
        full_text = re.sub(r'-\n\s*', '', full_text)
        full_text = re.sub(r'\n\s+(?=[a-z(])', ' ', full_text)

        # 找到所有 GP 标题匹配
        all_matches = []
        for pattern in self.GP_TITLE_PATTERNS:
            for m in re.finditer(pattern, full_text):
                all_matches.append((m.start(), m.group(0).strip()))

        # 按位置排序
        all_matches.sort(key=lambda x: x[0])

        # 过滤 GP 引用（产物条目中的 "according to..."）
        filtered = []
        for pos, title in all_matches:
            context_after = full_text[pos:pos + 300]
            if self._is_gp_definition(context_after):
                filtered.append((pos, title))

        if not filtered:
            return {}

        # 提取每个 GP 的文本（到下一个标题或上限）
        gp_texts = {}
        gp_counter = {}
        max_gp_chars = 4000

        for i, (pos, title) in enumerate(filtered):
            # 确定结束位置
            if i + 1 < len(filtered):
                end_pos = filtered[i + 1][0]
            else:
                end_pos = len(full_text)

            text = full_text[pos:end_pos].strip()
            # if len(text) > max_gp_chars:
            #     text = text[:max_gp_chars]

            key = self._make_gp_key(title, gp_counter)

            # 如果 key 已存在（罕见），合并文本
            if key in gp_texts:
                gp_texts[key] += "\n\n" + text
            else:
                gp_texts[key] = text

            if len(gp_texts[key]) > max_gp_chars:
                gp_texts[key] = gp_texts[key][:max_gp_chars]
        return self._split_embedded_procedure_scopes(gp_texts)

    def _split_embedded_procedure_scopes(self, gp_texts: Dict[str, str]) -> Dict[str, str]:
        """Split one General Procedure section into scoped Procedure A/B entries."""
        split_texts = {}
        heading_re = re.compile(
            r'(?:^|\n)\s*(?:\d+\)\s*)?Procedure\s+([A-Z])\s*'
            r'\((?:for\s+)?([^)]+?\b(?:products?|compounds?)\s+'
            r'([A-Za-z]?\d+[A-Za-z]*)\s*[-\u2010-\u2015]\s*([A-Za-z]?\d+[A-Za-z]*))\)',
            re.IGNORECASE,
        )

        for base_key, text in gp_texts.items():
            matches = list(heading_re.finditer(text))
            if len(matches) < 2:
                split_texts[base_key] = text
                continue

            for idx, match in enumerate(matches):
                letter = match.group(1).upper()
                lo = match.group(3)
                hi = match.group(4)
                start = match.start()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                scoped_text = text[start:end].strip()
                split_texts[f"GeneralProcedure{letter}_{lo}-{hi}"] = scoped_text

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
                    max_tokens=1000,
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
                max_tokens=300,
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

    def sanitize_reaction_schema(self, reaction: Dict) -> Dict:
        """Keep only the current LangGraph reaction schema surface."""
        if not isinstance(reaction, dict):
            return reaction

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

    def sanitize_reactions_schema(self, reactions: List[Dict]) -> List[Dict]:
        return [
            self.sanitize_reaction_schema(reaction)
            for reaction in reactions
            if isinstance(reaction, dict)
        ]

    def stage2_audit_missing_reactions(
        self,
        chunk_text: str,
        chunk_label: str,
        current_reactions: List[Dict],
        registry_block: str = "",
        gp_block: str = "",
    ) -> List[Dict]:
        """Ask the LLM to extract only reactions omitted from the first Stage2 pass."""
        if not self.enable_stage2_audit:
            return []

        try:
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
                            f"{registry_block}"
                            f"{gp_block}"
                            f"\ncurrent_extraction:\n{current_json}\n"
                            f"\nsource_text:\n{chunk_text}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=8000,
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
            return self.sanitize_reactions_schema(missing)
        except Exception as e:
            print(f"  [WARN] Stage2 audit failed ({chunk_label}): {e}")
            return []

    def stage2_extract(self, chunk_text: str, chunk_label: str,
                       registry: Optional[Dict[str, str]] = None,
                       gp_texts: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        用精确模型从分块中提取反应数据

        Args:
            chunk_text: 分块文本
            chunk_label: 分块标签（用于日志）
            registry: symbol→name 映射表（可选）
            gp_texts: 原始GP文本 dict（可选），格式: {"GP标签": "GP原文..."}
        """
        # 构建 registry 注入块
        registry_block = ""
        if registry:
            mapping_lines = "\n".join(
                f"  {sym} = {name}"
                for sym, name in registry.items()
            )
            registry_block = f"""

KNOWN SYMBOL-TO-NAME MAPPINGS (use these to replace symbols with full names):
{mapping_lines}

IMPORTANT: When you see a symbol like '1a' in a reaction table, replace it with 
the full chemical name from the mapping above. Keep both the symbol and name if useful.
"""

        # 构建 GP 注入块 (使用原始GP文本，不过滤，全量注入)
        gp_block = ""
        if gp_texts:
            gp_entries = []
            for label, text in gp_texts.items():
                if isinstance(text, str):
                    # 截断过长的GP文本
                    text_truncated = text[:800] + "..." if len(text) > 800 else text
                    entry = f"""=== {label} ===
{text_truncated}"""
                    gp_entries.append(entry)
            
            if gp_entries:
                gp_block = self.GP_INJECTION_TEMPLATE.format(
                    gp_block="\n\n".join(gp_entries)
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
                                f"{registry_block}"
                                f"{gp_block}"
                                f"\nText content:\n{chunk_text}"
                            )
                        }
                    ],
                    temperature=0.1,
                    max_tokens=16000,
                )
                gpt_response = response.choices[0].message.content or ""
                if not gpt_response.strip():
                    print(f"  [WARN] Stage2 attempt {attempt+1}/3: empty response ({chunk_label})")
                    continue
                reactions = self.sanitize_reactions_schema(
                    self.parse_and_validate_json(gpt_response)
                )
                missing = self.stage2_audit_missing_reactions(
                    chunk_text,
                    chunk_label,
                    reactions,
                    registry_block=registry_block,
                    gp_block=gp_block,
                )
                if missing:
                    self.stage2_audit_recovered += len(missing)
                    print(f"  [Stage2 audit] recovered {len(missing)} omitted reactions ({chunk_label})")
                    return self.merge_results([reactions, missing])
                return reactions
            except Exception as e:
                print(f"  [WARN] Stage2 attempt {attempt+1}/3 failed ({chunk_label}): {e}")

        print(f"  [ERROR] Stage2 最终失败 ({chunk_label})")
        return []

    # =====================================================================
    # 结果合并与去重
    # =====================================================================

    def merge_results(self, all_reactions: List[List[Dict]]) -> List[Dict]:
        """
        合并多个分块的提取结果，去重

        去重策略：
        1. 优先按 id 字段去重
        2. 无id时按 (substrates, products, yield, ee) 组合去重
        """
        merged = []
        seen_ids = set()
        seen_signatures = set()

        for chunk_reactions in all_reactions:
            if not chunk_reactions:
                continue
            for reaction in chunk_reactions:
                if not isinstance(reaction, dict):
                    continue

                # 策略1：按id去重
                rid = reaction.get('id')
                if rid:
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    merged.append(reaction)
                    continue

                # 策略2：按内容签名去重
                sig = self._reaction_signature(reaction)
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                merged.append(reaction)

        return merged

    def _reaction_signature(self, reaction: Dict) -> str:
        """生成反应的唯一签名用于去重"""
        subs = json.dumps(reaction.get('substrates', ''), sort_keys=True, ensure_ascii=False)
        prods = json.dumps(reaction.get('products', ''), sort_keys=True, ensure_ascii=False)
        target = json.dumps(
            reaction.get('targets', reaction.get('target', '')),
            sort_keys=True,
            ensure_ascii=False,
        )
        return f"{subs}|{prods}|{target}"

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
                if gp_key in rid:
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
        self.stage2_audit_recovered = 0

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
        chunks = self.chunk_pages(relevant_pages)
        file_stats['total_chunks'] = len(chunks)

        print(f"[ReactionExtractionAgent] Stage1 screening with {self.screen_model}...")
        screen_results = []
        for chunk in chunks:
            label = f"{pdf_name} Chunk{chunk['chunk_id']} Pages[{chunk['page_range']}]"
            screen_results.append(self.stage1_screen(chunk['text'], label))

        relevant_chunks = [
            (chunk, screen)
            for chunk, screen in zip(chunks, screen_results)
            if screen.get('has_reactions')
        ]
        file_stats['screened_pass'] = len(relevant_chunks)
        file_stats['screened_fail'] = len(chunks) - len(relevant_chunks)
        if not relevant_chunks:
            print("  [SKIP] Stage1 found no reaction chunks")
            return None

        print(f"[ReactionExtractionAgent] Stage2 extraction with {self.extract_model}...")
        all_chunk_results = []
        for chunk, _screen in relevant_chunks:
            label = f"{pdf_name} Chunk{chunk['chunk_id']} Pages[{chunk['page_range']}]"
            chunk_registry = self._filter_registry_for_chunk(chunk['text'], registry)
            reactions = self.stage2_extract(
                chunk['text'],
                label,
                registry=chunk_registry,
                gp_texts=gp_texts,
            )
            all_chunk_results.append(reactions)

        merged = self.merge_results(all_chunk_results)
        if registry and merged:
            merged = self.align_names_in_reactions(merged, registry)
        scaffold_mapping = entity_context.get("scaffold_substituent_mapping") or {}
        if scaffold_mapping and merged:
            merged = self.enrich_reactions_with_scaffold_mapping(merged, scaffold_mapping)
        if gp_texts and merged:
            merged = self.apply_gp_conditions_fallback(merged, gp_texts)
        merged = self.sanitize_reactions_schema(merged)
        file_stats['stage2_audit_recovered'] = self.stage2_audit_recovered

        output_file = Path(output_path) if output_path else output_dir / f"{Path(pdf_path).stem}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "source": str(pdf_path),
            "extracted_at": datetime.now().isoformat(),
            "total_reactions": len(merged),
            "name_registry": registry,
            "general_procedures": {
                k: v[:300] + "..." if isinstance(v, str) and len(v) > 300 else v
                for k, v in gp_texts.items()
            } if gp_texts else {},
            "entity_context_path": entity_context.get("_context_path"),
            "stats": file_stats,
            "reactions": merged,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

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

    def align_names_in_reactions(self, reactions: List[Dict],
                                 registry: Dict[str, str]) -> List[Dict]:
        """Replace registry symbols in extracted reaction entities."""
        if not registry:
            return reactions

        aligned = []
        replace_count = 0
        for reaction in reactions:
            if not isinstance(reaction, dict):
                aligned.append(reaction)
                continue

            new_reaction = dict(reaction)
            for field in ['substrates', 'products', 'catalysts', 'additives', 'reagents']:
                items = new_reaction.get(field)
                if not items:
                    continue

                if isinstance(items, list):
                    new_items = []
                    for item in items:
                        if isinstance(item, str) and item in registry:
                            new_items.append(registry[item])
                            replace_count += 1
                        elif isinstance(item, dict):
                            new_item = dict(item)
                            name_val = new_item.get('name', '')
                            if name_val in registry:
                                new_item['name'] = registry[name_val]
                                new_item['symbol'] = name_val
                                replace_count += 1
                            new_items.append(new_item)
                        else:
                            new_items.append(item)
                    new_reaction[field] = new_items
                elif isinstance(items, str) and items in registry:
                    new_reaction[field] = registry[items]
                    new_reaction[f'{field}_symbol'] = items
                    replace_count += 1

            aligned.append(new_reaction)

        if replace_count > 0:
            print(f"  [Registry] symbol replacements: {replace_count}")
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
                "registry_grounded_count": self.last_registry_validation_stats.get("registry_grounded_count", len(registry)),
                "registry_removed_count": self.last_registry_validation_stats.get("registry_removed_count", 0),
                "registry_removed_entries": self.last_registry_validation_stats.get("registry_removed_entries", [])[:20],
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
            chunks = self.chunk_pages(relevant_pages)
            file_stats['total_chunks'] = len(chunks)
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
            relevant_chunks = [
                (chunk, screen)
                for chunk, screen in zip(chunks, screen_results)
                if screen['has_reactions']
            ]
            file_stats['screened_pass'] = len(relevant_chunks)
            file_stats['screened_fail'] = len(chunks) - len(relevant_chunks)
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
                chunk_registry = self._filter_registry_for_chunk(chunk['text'], registry)

                print(f"  块{chunk['chunk_id']}: "
                      f"registry {len(registry)}→{len(chunk_registry)}, "
                      f"GP 全量注入 {len(gp_texts)} 个")

                # 传入原始 gp_texts（全部GP），不过滤
                reactions = self.stage2_extract(chunk['text'], label, registry=chunk_registry, gp_texts=gp_texts)
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
