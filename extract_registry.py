import json
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pdf_to_gpt_extractor import PDFReactionExtractor, PDF_LIBRARY


# REGISTRY_PROMPT = """Extract symbol-to-chemical-name mappings from chemistry text.
# CORE PRINCIPLE: Distinguish short symbols (like "1a", "C1") from chemical formulas.
# - Short symbols (digits+letters or letter+digits) should be mapped to full chemical names.
# - Chemical formulas should be mapped to themselves (or their IUPAC names if provided), NOT to generic terms like "photocatalyst".
# Look for these patterns:
# 1. "full_chemical_name 1a" or "full_chemical_name (1a)" where 1a is a symbol
# 2. "1a: full_chemical_name" or "1a, full_chemical_name"
# 3. Catalyst/ligand symbols like C1, P1, L1 mapped to names
# Return ONLY a JSON object mapping symbols to full chemical names:
# {"1a": "full_chemical_name", "2b": "full_name", "C1": "catalyst_name"}
# Rules:
# - Symbols are short identifiers: digits+letters (1a, 2b, 3c) or letter+digits (C1, P1, L1)
# - Symbol length must be <= 5 characters. Examples: "1a" (2 chars), "3ab" (3 chars), "C1" (2 chars), "L10" (3 chars). NOT: "4CzIPN" (6 chars - too long, treat as chemical formula)
# - For identifiers that look like chemical formulas (contain parentheses, mixed case, or complex structures, or length > 5), keep them as names. Do NOT map them to generic functional terms.
# - Full names are IUPAC or descriptive chemical names
# - If a symbol appears with both a short abbreviation and full name, use the full name
# - Only include mappings for substrates, products, catalysts, ligands, additives
# - Do NOT include yield numbers, conditions, or amounts in the name
# """
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


class RegistryExtractor(PDFReactionExtractor):
    """专门用于提取化学符号→化学名称映射表的提取器"""

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

    def __init__(self, api_key: Optional[str] = None, screen_model: str = "gpt-4o-mini"):
        """初始化Registry提取器

        Args:
            api_key: OpenAI API密钥
            screen_model: GPT模型用于registry提取
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("请提供OpenAI API密钥或设置环境变量OPENAI_API_KEY")

        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://oneapi.xty.app/v1",
                api_key=self.api_key
            )
        except Exception as e:
            raise Exception(f"OpenAI客户端初始化失败: {e}")

        self.screen_model = screen_model

    def extract_text_by_pages(self, pdf_path: str) -> List[Dict]:
        """按页提取PDF文本

        Returns:
            List of Dict with keys: page_num, text
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

    def chunk_pages(self, pages: List[Dict], pages_per_chunk: int = 5) -> List[Dict]:
        """将页面按指定数量分块
        
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

    def extract_name_registry(self, pages: List[Dict], max_scan_pages: int = 60, pages_per_chunk: int = 5) -> Dict[str, str]:
        """提取 symbol→full_chemical_name 映射表
        
        策略：直接使用GPT分块提取，然后合并去重
        
        Args:
            pages: 按页提取的文本列表
            max_scan_pages: 最多扫描前N页
            pages_per_chunk: 每块页数
            
        Returns:
            {"1a": "full_chemical_name", "C1": "catalyst_name"}
        """
        if not pages:
            return {}
            
        scan_pages = pages[:max_scan_pages]
        
        print(f"  [Step 1] 页面分块: {len(scan_pages)}页 分 {pages_per_chunk}页/块")
        chunks = self.chunk_pages(scan_pages, pages_per_chunk)
        print(f"  [Step 2] 分为 {len(chunks)} 个分块，分别调用GPT提取...")
        
        all_registries = []
        for chunk in chunks:
            print(f"    处理分块{chunk['chunk_id']}: 页 {chunk['page_nums']}")
            registry = self.extract_registry_with_gpt(chunk['text'])
            if registry:
                print(f"      提取到 {len(registry)} 条映射")
                all_registries.append(registry)
            else:
                print(f"      未提取到映射")
        
        print(f"  [Step 3] 合并 {len(all_registries)} 个分块的结果...")
        merged_registry = self.merge_registries(all_registries)
        
        print(f"  [Step 4] 清理通用术语...")
        cleaned_registry = self._clean_registry(merged_registry)
        
        return cleaned_registry

    def merge_registries(self, registries: List[Dict[str, str]]) -> Dict[str, str]:
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

    

    def _clean_registry(self, registry: Dict[str, str]) -> Dict[str, str]:
        """删除映射到泛指术语或不合规名称的条目

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

    def extract_registry_with_gpt(self, text: str) -> Dict[str, str]:
        """用 gpt-4o-mini 从文本中提取 symbol→name 映射（正则的fallback）

        Args:
            text: 要分析的文本（通常是PDF前N页）

        Returns:
            {"1a": "full_name", "C1": "..."}
        """
        try:
            if len(text) > 15000:
                text = text[:15000]

            response = self.client.chat.completions.create(
                model=self.screen_model,
                messages=[
                    {"role": "system", "content": "You are a chemistry data extractor. Always respond with valid JSON only."},
                    {"role": "user", "content": f"{REGISTRY_PROMPT}\n\n{text}"}
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content.strip()
            return self._parse_registry_response(raw)
        except Exception as e:
            print(f"[WARN] GPT registry提取出错: {e}")
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

    def process_single_pdf(self, pdf_path: str, output_dir: str, 
                        max_scan_pages: int = 60, pages_per_chunk: int = 5) -> Optional[Dict]:
        """处理单个PDF文件，提取名称映射表

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            max_scan_pages: 最多扫描前N页
            pages_per_chunk: 每块页数

        Returns:
            {"pdf_name": "...", "registry": {...}}
        """
        pdf_name = Path(pdf_path).name
        print(f"\n{'='*60}")
        print(f"处理: {pdf_name}")
        print(f"{'='*60}")

        try:
            print("[Step 1] 按页提取文本...")
            pages = self.extract_text_by_pages(pdf_path)
            print(f"  提取到 {len(pages)} 页文本")

            if not pages:
                print("[SKIP] PDF无文本内容（可能是纯图像PDF）")
                return None

            print(f"[Step 2] 提取 symbol→name 注册表 (max_pages={max_scan_pages}, chunk={pages_per_chunk})...")
            registry = self.extract_name_registry(
                pages, 
                max_scan_pages=max_scan_pages,
                pages_per_chunk=pages_per_chunk
            )
            
            print(f"\n{'='*60}")
            print(f"提取完成: {len(registry)} 条映射")
            print(f"{'='*60}")
            
            if registry:
                # 显示前5条
                for sym, name in list(registry.items())[:5]:
                    display_name = name[:50] + "..." if len(name) > 50 else name
                    print(f"  {sym} → {display_name}")
                if len(registry) > 5:
                    print(f"  ... 还有 {len(registry) - 5} 条")
            else:
                print("  [WARN] 未提取到名称映射")

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            result = {
                "pdf_name": pdf_name,
                "registry": registry
            }

            json_path = output_path / f"{Path(pdf_path).stem}_registry.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            print(f"[SAVE] Registry保存到: {json_path}")

            return result

        except Exception as e:
            print(f"[ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def batch_process(self, input_path: str, output_dir: str, 
                  max_scan_pages: int = 60, pages_per_chunk: int = 5) -> Dict:
        """批量处理PDF文件或文件夹

        Args:
            input_path: PDF文件路径或文件夹路径
            output_dir: 输出目录
            max_scan_pages: 最多扫描前N页
            pages_per_chunk: 每块页数

        Returns:
            {"success": [...], "failed": [...], "total": N}
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = []
        if input_path.is_file():
            if input_path.suffix.lower() == '.pdf':
                pdf_files = [input_path]
        elif input_path.is_dir():
            pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            print(f"未找到PDF文件: {input_path}")
            return {"success": [], "failed": [], "total": 0}

        print(f"找到 {len(pdf_files)} 个PDF文件")
        print(f"配置: max_scan_pages={max_scan_pages}, pages_per_chunk={pages_per_chunk}")

        success = []
        failed = []

        for pdf_file in pdf_files:
            result = self.process_single_pdf(
                str(pdf_file), 
                str(output_dir),
                max_scan_pages=max_scan_pages,
                pages_per_chunk=pages_per_chunk
            )
            if result is not None:
                success.append(pdf_file.name)
            else:
                failed.append(pdf_file.name)

        print(f"\n=== 处理完成: {len(success)}/{len(pdf_files)} 成功 ===")
        if failed:
            print(f"失败的文件: {failed}")

        return {
            "success": success,
            "failed": failed,
            "total": len(pdf_files)
        }


def main():
    parser = argparse.ArgumentParser(
        description="CHEMICAL SYMBOL TO NAME 映射提取工具 (GPT-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python extract_registry.py --input paper.pdf --output ./output --api_key YOUR_KEY
  python extract_registry.py --input ./supporting_information --output ./output --api_key YOUR_KEY
  python extract_registry.py --input ./SI_folder --output ./output --api_key YOUR_KEY --max_pages 30 --pages_per_chunk 8

环境变量:
  OPENAI_API_KEY: OpenAI API密钥（必需）

API消耗估算:
  每个PDF ~$0.06 (60页, gpt-4o-mini)
"""
    )
    parser.add_argument("--input", required=True,
                        help="PDF文件或文件夹路径")
    parser.add_argument("--output", required=True,
                        help="输出目录")
    parser.add_argument("--api_key", default=None,
                        help="OpenAI API密钥 (默认从OPENAI_API_KEY环境变量读取)")
    parser.add_argument("--max_pages", type=int, default=60,
                        help="最大扫描页数 (默认: 60)")
    parser.add_argument("--pages_per_chunk", type=int, default=5,
                        help="每块页数 (默认: 5)")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("错误: 请提供OpenAI API密钥")
        print("  方法1: 设置环境变量 $env:OPENAI_API_KEY='your-api-key'")
        print("  方法2: 使用参数 --api_key YOUR_KEY")
        return

    extractor = RegistryExtractor(
        api_key=api_key,
        screen_model="gpt-4o-mini"
    )

    result = extractor.batch_process(
        args.input, 
        args.output, 
        max_scan_pages=args.max_pages,
        pages_per_chunk=args.pages_per_chunk
    )
    print(f"\n结果: {result}")


if __name__ == "__main__":
    main()