from batch_si_extractor import SIExtractor


def test_reaction_prompts_describe_name_symbol_separation():
    for prompt in (
        SIExtractor.STRUCTURED_REACTION_PROMPT,
        SIExtractor.NON_GP_REACTION_PROMPT,
        SIExtractor.MIXED_REACTION_PROMPT,
    ):
        assert "name stores the chemical name only" in prompt
        assert "symbol stores the reported label" in prompt
        assert "carboxylic acid (34)" in prompt
        assert "dibromovinyl ... (35')" in prompt
        assert "methanol (36)" in prompt


def test_sanitize_splits_trailing_reported_product_symbol():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = {
        "id": "r1",
        "source_pages": [34],
        "reaction_type": "test",
        "substrates": [],
        "products": [
            {"name": "(1aR,1a1R,9bS)-4-oxo-carboxylic acid (34)", "symbol": None}
        ],
        "catalysts": [],
        "ligands": [],
        "other_components": [],
        "conditions": {},
        "targets": {"yield": "68%", "ee": None, "er": None},
    }

    sanitized = extractor.sanitize_reaction_schema(reaction)

    assert sanitized["products"][0]["name"] == "(1aR,1a1R,9bS)-4-oxo-carboxylic acid"
    assert sanitized["products"][0]["symbol"] == "34"


def test_sanitize_splits_prime_and_code_symbols():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = {
        "id": "r1",
        "source_pages": [35],
        "reaction_type": "test",
        "substrates": [{"name": "starting material (SM15)"}],
        "products": [
            {"name": "reported dibromovinyl compound (35')"},
            {"name": "reported methanol (36)"},
        ],
        "catalysts": [],
        "ligands": [{"name": "reported ligand (L1)"}],
        "other_components": [{"name": "reported reagent (3a)"}],
        "conditions": {},
        "targets": {"yield": "71%", "ee": None, "er": None},
    }

    sanitized = extractor.sanitize_reaction_schema(reaction)

    assert sanitized["substrates"][0]["name"] == "starting material"
    assert sanitized["substrates"][0]["symbol"] == "SM15"
    assert sanitized["products"][0]["name"] == "reported dibromovinyl compound"
    assert sanitized["products"][0]["symbol"] == "35'"
    assert sanitized["products"][1]["name"] == "reported methanol"
    assert sanitized["products"][1]["symbol"] == "36"
    assert sanitized["ligands"][0] == {"name": "reported ligand", "symbol": "L1"}
    assert sanitized["other_components"][0]["name"] == "reported reagent"
    assert sanitized["other_components"][0]["symbol"] == "3a"


def test_sanitize_does_not_split_chemical_parentheticals_or_existing_symbol():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = {
        "id": "r1",
        "source_pages": [1],
        "reaction_type": "test",
        "substrates": [
            {"name": "(E)-hex-4-enoic acid"},
            {"name": "product name (34)", "symbol": "existing"},
        ],
        "products": [{"name": "(R,R)-QuinoxP*"}],
        "catalysts": [],
        "ligands": [],
        "other_components": [],
        "conditions": {},
        "targets": {"yield": "80%", "ee": None, "er": None},
    }

    sanitized = extractor.sanitize_reaction_schema(reaction)

    assert sanitized["substrates"][0] == {"name": "(E)-hex-4-enoic acid"}
    assert sanitized["substrates"][1] == {"name": "product name (34)", "symbol": "existing"}
    assert sanitized["products"][0] == {"name": "(R,R)-QuinoxP*"}


def test_sanitize_splits_multistep_intermediate_symbol():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = {
        "id": "r1",
        "source_pages": [1, 2],
        "reaction_type": "test",
        "step_count": 2,
        "substrates": [{"name": "starting material", "step": 1}],
        "intermediates": [
            {
                "name": "intermediate from step 1 (7-(E))",
                "produced_in_step": 1,
                "consumed_in_step": 2,
            }
        ],
        "products": [{"name": "final product (36)", "step": 2}],
        "catalysts": [],
        "ligands": [],
        "other_components": [{"name": "reagent", "step": 2}],
        "conditions": {},
        "targets": {"yield": "60%", "ee": None, "er": None},
    }

    sanitized = extractor.sanitize_reaction_schema(reaction)

    assert sanitized["intermediates"][0]["name"] == "intermediate from step 1"
    assert sanitized["intermediates"][0]["symbol"] == "7-(E)"
    assert sanitized["products"][0]["name"] == "final product"
    assert sanitized["products"][0]["symbol"] == "36"
