"""
PDF文献数据抽取工具
使用OpenAI GPT API从化学文献PDF中提取反应数据
"""

import base64
import json
import os
from pathlib import Path
from typing import Optional, Dict, List
import openai
from openai import OpenAI

# 尝试导入python-dotenv来加载.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()  # 自动加载.env文件
except ImportError:
    pass  # 如果没有安装python-dotenv，继续使用环境变量

# 尝试导入PDF处理库（优先使用pdfplumber，更稳定快速）
try:
    import pdfplumber
    PDF_LIBRARY = "pdfplumber"
except ImportError:
    try:
        import PyPDF2
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        PDF_LIBRARY = None
        print("警告: 未安装PDF处理库。请运行: pip install pdfplumber 或 pip install PyPDF2")

# 尝试导入 pdf2image 用于图片分块（适用于扫描版/图像型 PDF）
try:
    from pdf2image import convert_from_path
    PDF_IMAGE_AVAILABLE = True
except ImportError:
    PDF_IMAGE_AVAILABLE = False
    print("提示: 未安装 pdf2image，图像分块将不可用。如需支持扫描版 PDF，请运行: pip install pdf2image poppler-utils")


class PDFReactionExtractor:
    """PDF文献反应数据提取器"""
    
    # 数据抽取Prompt模板
    EXTRACTION_PROMPT = """You are a chemistry literature extraction specialist.
Extract every qualifying paragraph/prose reaction entry from any SI section. Ignore tables, figures, captions, and analytical-only text.

Rules
- Extract every qualifying reaction entry in source order with no omissions. Only use reported data.
- Different ee/yield = separate records even if substrates match.
- Keep numbers exactly as written (units, significant figures, ranges).
  Use null for missing yield, ee, or er fields.
- Classify catalyst complexes/precatalysts under catalysts, ligands under ligands, and all other reacting materials under other_components. Do NOT drop any.
- ligands items contain name only for single-step reactions; multi-step ligands contain name and step only. Never put ligand loading in ligands.

Coverage-first extraction:
Internally identify every extractable paragraph/prose reaction entry in source order before writing the final JSON.

Page provenance:
- Every reaction object must include "source_pages": [positive PDF page numbers].
- Read page numbers only from the "--- Page N ---" markers surrounding the specific reaction entry in source_text.
- Include only pages containing concrete reaction entry evidence: its specific substrate/product, operation, or reported result.
- Do not include a General Procedure definition page merely because GP context was supplied or referenced.
- If the concrete entry spans pages, include every evidence page once in ascending order.
- If the page cannot be determined, use "source_pages": []. Never invent a page number.

An extractable prose reaction entry is any sentence or paragraph that contains:
- a specific compound/product/substrate name, label, code, or symbol, and
- wording indicating preparation, synthesis, isolation, furnishing, affording, obtaining, or reaction under/according to a procedure, and
- at least one reported target value: isolated yield, ee, or er.

Yield alone is sufficient. ee and er are optional.

For a consecutive series of similar product entries, extract every entry in order.
Do not skip middle entries in a repeated series.
If entries have the same wording pattern but different compound symbols/names/yields, each one is a separate reaction record.

Product characterization entries are valid reaction entries when they report a specific isolated product and yield, even if most following text is analytical data.
Ignore NMR, HRMS, HPLC, spectra, exact mass, melting point, optical rotation, and analytical details after the yield; they are not separate reactions.

General Procedure text is context only:
- Do not output a standalone reaction for the GP paragraph itself unless it reports a specific product/substrate and yield, ee, or er.
- Use GP text only to fill shared reagents, catalysts, solvents, and conditions for later prose entries that reference that procedure.

Multi-step reactions and General Procedures:
- Before extracting GP-referenced entries, determine whether the referenced GP describes one chemical transformation or multiple sequential chemical transformations. Do not output this decision separately.
- A multi-step reaction or GP contains two or more sequential chemical transformations leading to one final reported product/result. Keep that sequence as ONE reaction object.
- A synthetic step requires a chemical transformation, not just an operational step.
- Do not count workup or purification as a synthetic step: quench, extraction, washing, drying, concentration, filtration, and chromatography are not separate steps unless the source explicitly performs another chemical transformation.
- Expressions such as "over two steps", "used directly in the next step", "without isolation/further purification", or "the crude product/residue was subjected to another transformation" are evidence to review; they are not sufficient by themselves unless the text describes multiple chemical transformations.
- If a supplied GP is multi-step, every concrete entry that references that GP must inherit the multi-step schema.
- For a multi-step reaction only, add integer "step_count" and integer "step" to every item in substrates, products, catalysts, ligands, and other_components.
- If any substrate, product, catalyst, additive, reagent, intermediate, or condition uses a step field, the reaction must include integer "step_count".
- Assign each chemical to the step where it is actually used or formed. Do not flatten chemicals from different steps into an unlabelled list.
- For multi-step conditions, every condition field must be a list of {"step": N, "value": "reported value"} objects. Preserve separate solvent, volume, atmosphere, light source, wavelength, temperature, and time values by step.
- Normalize condition fields by meaning:
  * solvent records solvent identity only, without quantities, equivalents, or procedural roles.
  * volume records solvent quantities and short role notes for separate portions, addition solutions, suspensions, dilutions, or reaction mixtures.
  * When the same solvent appears in multiple portions in one synthetic step, keep the solvent identity once and keep the distinct quantities/purposes in volume.
  * Do not duplicate the same quantity in both solvent and volume.
  * Exclude workup, extraction, washing, and chromatography solvents from reaction conditions unless the text uses them as the reaction medium.
- Add "intermediates" for multi-step reactions. Include an intermediate only when the source explicitly gives its chemical name or symbol/code. Each intermediate must contain name, symbol, amount, produced_in_step, and consumed_in_step.
- Do not invent an intermediate identity from phrases such as "crude product", "residue obtained above", or "corresponding intermediate". If no intermediate is explicitly identified, use "intermediates": [].
- An intermediate belongs only in intermediates; do not duplicate it in substrates or products.
- Keep an overall reported yield exactly scoped as written, for example "66% yield over two steps" rather than "66%".
- Single-step reactions must keep the ordinary schema below: do not add step_count, step annotations, or intermediates.

Do NOT extract from tables, optimization tables, screening tables, entry tables, figure captions, or tabular lists.
Ignore table content completely even if it contains yield, ee, er, conditions, substrates, or entry numbers.

General Procedure entries
Many SI docs have a GP paragraph followed by individual entries specifying only compound, yield, ee/er, and sometimes time.
Product characterization entries after a GP are separate reaction entries when they report a specific product and isolated yield, ee, or er.
1. conditions — Entry-specified reagents/solvents/conditions ALWAYS override GP conditions.
2. substrates — Use GP substrate symbols/ranges. Do NOT set substrates = products.
   For compound-range GPs (e.g. "1a-19a"): use general class name (e.g. "alkene derivative"), keep symbol.
   Check KNOWN SYMBOL-TO-NAME MAPPINGS to infer substrate chemical class. Do NOT use product IUPAC as substrate.
3. id — GP entries: "GeneralProcedure{scope}-Entry{N}" where scope = number (e.g. "1-38") OR letter (e.g. "A", "B").
   Other sections: "{SectionName}-Entry{N}".
4. When text references a GP (e.g. "Following GP5", "Following General Procedure A"), use that GP's scope in id.
   DO NOT use "SubstrateScope-Entry{N}" for GP-referenced reactions.
5. For characterization-only product entries, infer substrates/reagents/conditions from the referenced GP context.
   The product header supplies products[].name/symbol only; it must never be copied into substrates[].
   If the exact substrate identity is not stated in the entry, use the GP's generic substrate class/range instead of guessing.

Reaction type
- Include reaction_type for every record.
- Use the most specific reaction class supported by title, section, GP, substrates, products, catalysts, reagents, or conditions.
- Do not use catalyst names, labels, or broad context labels as reaction_type; if unclear, use "unknown reaction".

Output
Output only valid JSON matching the requested task.
For normal extraction, return only a JSON array of reaction objects.
For audit tasks, follow the audit prompt and return only the requested JSON object.
Never include coverage lists, explanations, headings, bullets, prose prefaces, Markdown fences, or text outside JSON.

For normal extraction, each entry:
{
  "id": "...",
  "source_pages": [1],
  "reaction_type": "...",
  "substrates": [{"name": "...", "symbol": "...", "amount": "..."}],
  "products": [{"name": "...", "symbol": "...", "amount": "..."}],
  "catalysts": [{"name": "...", "symbol": "...", "amount": "..."}],
  "ligands": [{"name": "..."}],
  "other_components": [{"name": "...", "symbol": "...", "amount": "..."}],
  "conditions": {"solvent": "...", "volume": "...", "concentration": "...", "atmosphere": "...", "light_source": "...", "wavelength": "...", "temperature": "...", "time": "..."},
  "targets": {"yield": "...", "ee": "...%", "er": "..."}
}

For a multi-step entry, extend that object as follows:
{
  "step_count": 2,
  "substrates": [{"name": "...", "symbol": "...", "amount": "...", "step": 1}],
  "products": [{"name": "...", "symbol": "...", "amount": "...", "step": 2}],
  "catalysts": [{"name": "...", "symbol": "...", "amount": "...", "step": 1}],
  "ligands": [{"name": "...", "step": 1}],
  "other_components": [{"name": "...", "symbol": "...", "amount": "...", "step": 2}],
  "intermediates": [{"name": "...", "symbol": "...", "amount": "...", "produced_in_step": 1, "consumed_in_step": 2}],
  "conditions": {
    "solvent": [{"step": 1, "value": "..."}, {"step": 2, "value": "..."}],
    "volume": [{"step": 1, "value": "..."}, {"step": 2, "value": "..."}],
    "temperature": [{"step": 1, "value": "..."}, {"step": 2, "value": "..."}],
    "time": [{"step": 1, "value": "..."}, {"step": 2, "value": "..."}]
  }
}

name/symbol rules
- "name" = full IUPAC or descriptive chemical name. "symbol" = short identifier (e.g. "1a", "5b").
  If no full name found, use symbol for both.
- SI pages often have a compound header with full IUPAC name, then a results line with short descriptor.
  ALWAYS use the FULL name from the header as "name", NOT the short descriptor.
  Example: Header="...cyclobutane-1-carbaldehyde (5b)", Results="aldehyde 5b [10.4 mg, ...]"
  → {"name": "...cyclobutane-1-carbaldehyde", "symbol": "5b", "amount": "10.4 mg"}
  NOT {"name": "aldehyde 5b", "symbol": "5b"}. Scan entire section for longest, most complete name.
- Same name rules apply to substrates, products, catalysts, ligands, and other_components.
- products[].amount = mass/volume ONLY (e.g. "511 mg"), NEVER yield.
- yield/ee/er MUST be inside "targets", NOT top-level. Extract er when reported.
- Use null for unreported fields."""
    
    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://hk.xty.app/v1"):
        """
        初始化提取器
        
        Args:
            api_key: OpenAI API密钥，如果不提供则从环境变量OPENAI_API_KEY读取
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("请提供OpenAI API密钥或设置环境变量OPENAI_API_KEY")
        
        self.base_url = base_url
        self.client = OpenAI(
                            base_url=self.base_url,
                            api_key=self.api_key
                            )

    def extract_images_from_pdf(self, pdf_path: str, output_dir: Optional[str] = None,
                                dpi: int = 150, poppler_path: Optional[str] = None) -> List[str]:
        """
        将 PDF 页转换为图片路径列表（适合扫描版 PDF 或需要视觉模型时）
        """
        if not PDF_IMAGE_AVAILABLE:
            raise ImportError("未安装 pdf2image，无法进行图片分块。请运行: pip install pdf2image poppler-utils")

        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        output_root = Path(output_dir) if output_dir else Path("tmp_pdf_images")
        output_root.mkdir(parents=True, exist_ok=True)

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            output_folder=output_root,
            fmt="png",
            poppler_path=poppler_path
        )

        image_paths: List[str] = []
        for idx, img in enumerate(images, 1):
            img_path = output_root / f"page_{idx}.png"
            img.save(img_path, "PNG")
            image_paths.append(str(img_path))

        return image_paths

    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        """将图片文件编码为 base64 字符串"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        从PDF文件中提取文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的文本内容
        """
        if PDF_LIBRARY is None:
            raise ImportError("未安装PDF处理库。请运行: pip install PyPDF2 或 pip install pdfplumber")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        text = ""
        
        if PDF_LIBRARY == "PyPDF2":
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        elif PDF_LIBRARY == "pdfplumber":
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        return text
    
    
    def call_gpt(self, pdf_text: str, model: str = "gpt-5-mini",
                 temperature: float = 0.1, max_tokens: int = 16000,
                 chunk_hint: str = "") -> str:
        """调用GPT API进行数据抽取"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert chemistry literature data extraction specialist. "
                            "Extract every qualifying reaction from the text with no omissions. "
                            "Always use full IUPAC names for the 'name' field and short symbols "
                            "(e.g. '3aa', '1a') for the 'symbol' field."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Processing chunk: {chunk_hint}\n"
                            f"Extract ALL qualifying reactions from the text below. Do not omit any.\n"
                            f"\n\n{self.EXTRACTION_PROMPT}\n\nText content:\n{pdf_text}"
                        )
                    }
                ],
                temperature=temperature,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"调用GPT API时出错: {str(e)}")

    def call_gpt_with_image(self, image_path: str, text_prompt: str,
                            model: str = "gpt-5-mini", temperature: float = 0.1,
                            chunk_hint: str = "") -> str:
        """调用GPT多模态接口，传入单张图片"""
        base64_image = self.encode_image_base64(image_path)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"当前分块: {chunk_hint}\n{text_prompt}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"调用GPT多模态API时出错: {str(e)}")


    def _strip_markdown_fence(self, text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()

    def _parse_json_with_trailing_text(self, text: str):
        text = self._strip_markdown_fence(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as original_error:
            stripped = text.lstrip()
            if not stripped or stripped[0] not in "[{":
                raise original_error

            decoder = json.JSONDecoder()
            try:
                data, end = decoder.raw_decode(stripped)
            except json.JSONDecodeError:
                raise original_error

            if stripped[end:].strip():
                return data
            raise original_error

    def parse_and_validate_json(self, json_str: str) -> List[Dict]:
        """
        解析和验证JSON输出
        
        Args:
            json_str: GPT返回的JSON字符串
            
        Returns:
            解析后的反应数据列表
        """
        # 移除可能的Markdown代码块标记
        json_str = json_str.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        try:
            data = self._parse_json_with_trailing_text(json_str)
            
            # 确保是列表 - 处理多种可能的包装格式
            if isinstance(data, dict):
                # 尝试提取reactions键
                if 'reactions' in data:
                    data = data['reactions']
                # 尝试提取任何列表值
                elif len(data) > 0:
                    for key, value in data.items():
                        if isinstance(value, list) and len(value) > 0:
                            # 检查是否是反应列表
                            if isinstance(value[0], dict) and 'id' in value[0]:
                                data = value
                                break
            
            # 确保是列表
            if not isinstance(data, list):
                data = [data] if isinstance(data, dict) else []
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析JSON输出: {str(e)}\n原始输出:\n{json_str}")
    
    def extract_from_pdf(self, pdf_path: str, output_path: Optional[str] = None,
                        model: str = "gpt-5-mini", temperature: float = 0.1) -> List[Dict]:
        """
        完整的PDF数据提取流程（无分块版本）
        
        Args:
            pdf_path: PDF文件路径
            output_path: 输出JSON文件路径（可选）
            model: GPT模型
            temperature: 温度参数
            
        Returns:
            提取的反应数据列表
        """
        print(f"正在读取PDF文件: {pdf_path}")
        
        # 提取PDF文本
        pdf_text = self.extract_text_from_pdf(pdf_path)
        print(f"PDF文本长度: {len(pdf_text)} 字符")
        
        # 调用GPT API - 增加max_tokens以支持更多数据
        print(f"正在调用GPT API ({model}) 提取反应数据...")
        gpt_response = self.call_gpt(
            pdf_text,
            model=model,
            temperature=temperature,
            chunk_hint="完整PDF"
        )

        # 调试：打印GPT返回的原始内容长度和前500字符
        print(f"\n[DEBUG] GPT原始返回长度: {len(gpt_response)} 字符")
        print(f"[DEBUG] GPT返回前500字符:\n{gpt_response[:500]}\n")

        # 解析JSON数据
        print("解析JSON数据...")
        reactions = self.parse_and_validate_json(gpt_response)
        
        print(f"成功提取 {len(reactions)} 条反应数据")
        
        # 如果指定了输出路径，保存结果
        if output_path:
            output_data = {
                "source": str(pdf_path),
                "extracted_at": __import__('datetime').datetime.now().isoformat(),
                "total_reactions": len(reactions),
                "reactions": reactions
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"数据已保存到: {output_path}")
        
        return reactions


def main():
    """主函数示例"""
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python pdf_to_gpt_extractor.py <PDF文件路径> [输出JSON路径]")
        print("\n示例:")
        print("  python pdf_to_gpt_extractor.py paper.pdf extracted_data.json")
        print("\n环境变量:")
        print("  OPENAI_API_KEY: OpenAI API密钥（必需）")
        return
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 从环境变量获取API密钥
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量 OPENAI_API_KEY")
        print("在PowerShell中设置: $env:OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # 创建提取器
        extractor = PDFReactionExtractor(api_key=api_key)
        
        # 执行提取
        reactions = extractor.extract_from_pdf(
            pdf_path=pdf_path,
            output_path=output_path,
            model="gpt-5-mini",
            temperature=0.1
        )
        
        print("\n提取完成!")
        print(f"共提取 {len(reactions)} 条反应数据")
        
        # 显示前3条数据示例
        if reactions:
            print("\n前3条数据示例:")
            for i, reaction in enumerate(reactions[:3], 1):
                print(f"\n反应 {i}: {reaction.get('id', 'N/A')}")
                print(f"  类型: {reaction.get('reaction_type', 'N/A')}")
                print(f"  产率: {reaction.get('target', {}).get('yield', 'N/A')}")
                print(f"  ee值: {reaction.get('target', {}).get('ee', 'N/A')}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
