import importlib.util
import json
import sys
from pathlib import Path


EXTRACT_DIR = Path(__file__).resolve().parents[2]


def load_module(name, relative_path):
    if str(EXTRACT_DIR) not in sys.path:
        sys.path.insert(0, str(EXTRACT_DIR))
    module_path = EXTRACT_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_structure_cache_imports_legacy_files(tmp_path):
    mod = load_module("multimodal_structure_enrichment_for_test", "multimodal_structure_enrichment.py")
    cache_dir = tmp_path / "intermediate" / "cache"
    authoritative = cache_dir / "compound_structure_parse_cache.json"
    write_json(
        cache_dir / "multimodal_structure_parse_cache.json",
        {
            "N-(2-bromophenyl)acetamide": {
                "parseable": True,
                "scaffold": "acetamide",
                "substituents": ["N-(2-bromophenyl)"],
            }
        },
    )
    write_json(
        cache_dir / "substrate_parse_cache.json",
        {
            "n-(2-BROMOPHENYL)acetamide": {
                "parseable": True,
                "scaffold": "acetamide",
                "substituents": ["N-(2-bromophenyl)"],
            },
            "Product A": {
                "parseable": True,
                "scaffold": "benzene",
                "substituents": ["4-fluoro"],
            },
        },
    )

    cache = mod.load_cache(authoritative)

    assert len(cache) == 2
    assert cache[mod.cache_key("N-(2-bromophenyl)acetamide")]["scaffold"] == "acetamide"
    assert cache[mod.cache_key("product a")]["scaffold"] == "benzene"


def test_product_enrichment_only_sets_scaffold(tmp_path, monkeypatch):
    mod = load_module("multimodal_structure_enrichment_for_product_test", "multimodal_structure_enrichment.py")
    input_path = tmp_path / "filtered" / "text" / "paper.json"
    output_dir = tmp_path / "filtered" / "structure_enriched"
    cache_path = tmp_path / "intermediate" / "cache" / "compound_structure_parse_cache.json"
    report_path = tmp_path / "filtered" / "reports" / "structure_enrichment_report.json"
    write_json(
        input_path,
        {
            "reactions": [
                {
                    "substrates": [
                        {
                            "name": "Known substrate",
                            "parseable": True,
                            "scaffold": "amide",
                            "substituents": ["N-aryl"],
                        }
                    ],
                    "products": [{"name": "Product A"}],
                }
            ]
        },
    )
    calls = []

    def fake_call_structure_llm(batch, client, model, role="substrate", max_retries=3):
        calls.append((role, list(batch)))
        assert role == "product"
        return {
            mod.cache_key("Product A"): {
                "parseable": True,
                "scaffold": "benzene",
                "substituents": ["should not be applied"],
                "structure_parse_source": "llm",
                "structure_parse_status": "parsed",
                "structure_parse_scope": "product_scaffold",
            }
        }

    monkeypatch.setattr(mod, "call_structure_llm", fake_call_structure_llm)

    report = mod.enrich_multimodal_structure(
        text_reaction_paths=[input_path],
        chemeagle_reaction_paths=[],
        output_dir=output_dir,
        cache_path=cache_path,
        report_path=report_path,
        client=object(),
        model="test-model",
        batch_size=30,
        roles=("substrates", "products"),
    )

    enriched = read_json(output_dir / "text" / "paper.json")
    product = enriched["reactions"][0]["products"][0]
    assert calls == [("product", ["Product A"])]
    assert product["scaffold"] == "benzene"
    assert product["substituents"] == []
    assert report["substrate_llm_identifiers"] == 0
    assert report["product_scaffold_llm_identifiers"] == 1
    assert report["skipped_existing_structure"] == 1


def test_q1q2_split_cache_only_does_not_need_api_key(tmp_path, monkeypatch):
    split = load_module("split_by_substrate_for_test", "split_by_substrate.py")
    input_path = tmp_path / "filtered" / "merged" / "all_filtered_reactions.json"
    output_dir = tmp_path / "filtered" / "benchmark"
    cache_path = tmp_path / "intermediate" / "cache" / "compound_structure_parse_cache.json"
    write_json(
        input_path,
        {
            "reactions": [
                {
                    "id": "r1",
                    "substrates": [
                        {
                            "name": "Known substrate",
                            "parseable": True,
                            "scaffold": "amide",
                            "substituents": ["N-aryl"],
                        }
                    ],
                    "products": [{"name": "Product A"}],
                    "conditions": {"solvent": "DCM"},
                    "targets": {"yield": "80%"},
                }
            ]
        },
    )
    write_json(
        cache_path,
        {
            "Product A": {
                "parseable": True,
                "scaffold": "benzene",
                "substituents": ["ignored"],
            }
        },
    )

    def fail_openai(*args, **kwargs):
        raise AssertionError("OpenAI must not be constructed in cache-only split")

    monkeypatch.setattr(split, "OpenAI", fail_openai)

    result = split.split_reactions(
        input_path=input_path,
        output_dir=output_dir,
        cache_path=cache_path,
        api_key=None,
        allow_llm=False,
    )

    report = read_json(output_dir / "q1q2_split_report.json")
    q1 = read_json(output_dir / "Q1_substrate_to_condition.json")
    assert result["llm_enabled"] is False
    assert report["cache_only"] is True
    assert report["missing_product_scaffold_count"] == 0
    assert q1[0]["products"][0]["scaffold"] == "benzene"
