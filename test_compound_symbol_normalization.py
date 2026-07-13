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


def test_registry_alignment_resolves_ligand_label_and_preserves_step():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    ligand_name = (
        "(1R,2R)-1,2-bis(2-bromophenyl)-N1,N2-"
        "diethylethane-1,2-diamine"
    )
    reactions = [
        {"id": "r1", "ligands": [{"name": "L6", "step": 2}]}
    ]

    aligned = extractor.align_names_in_reactions(reactions, {"L6": ligand_name})

    assert aligned[0]["ligands"] == [
        {
            "name": ligand_name,
            "symbol": "L6",
            "step": 2,
            "resolution_source": "name_registry",
            "resolution_method": "same_paper_symbol",
            "resolution_confidence": "high",
        }
    ]


def test_registry_alignment_leaves_unknown_ligand_unchanged():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    ligand = {"name": "unknown ligand", "step": 2}

    aligned = extractor.align_names_in_reactions(
        [{"id": "r1", "ligands": [ligand]}],
        {"L6": "registered ligand"},
    )

    assert aligned[0]["ligands"] == [ligand]


def test_registry_alignment_reports_ligand_name_conflict_without_overwrite():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    extracted_name = "explicitly extracted ligand name"
    registry_name = "registered ligand name"

    aligned = extractor.align_names_in_reactions(
        [
            {
                "id": "r1",
                "ligands": [
                    {"name": extracted_name, "symbol": "L6", "step": 2}
                ],
            }
        ],
        {"L6": registry_name},
    )

    assert aligned[0]["ligands"] == [
        {
            "name": extracted_name,
            "symbol": "L6",
            "step": 2,
            "registry_name": registry_name,
            "registry_match_status": "conflict",
            "registry_conflict_reason": (
                "extracted_name_differs_from_registry_symbol_name"
            ),
        }
    ]


def test_sanitize_preserves_resolved_ligand_identity_but_drops_amount():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    ligand_name = (
        "(1R,2R)-1,2-bis(2-bromophenyl)-N1,N2-"
        "diethylethane-1,2-diamine"
    )
    reaction = {
        "id": "r1",
        "source_pages": [32],
        "reaction_type": "test",
        "step_count": 3,
        "substrates": [{"name": "substrate", "step": 1}],
        "products": [{"name": "product", "step": 3}],
        "catalysts": [],
        "ligands": [
            {
                "name": ligand_name,
                "symbol": "L6",
                "amount": "10.0 mg, 0.024 mmol, 12 mol%",
                "step": 2,
                "resolution_source": "name_registry",
                "resolution_method": "same_paper_symbol",
                "resolution_confidence": "high",
            }
        ],
        "other_components": [],
        "intermediates": [],
        "conditions": {},
        "targets": {"yield": "76%", "ee": "94%", "er": None},
    }

    sanitized = extractor.sanitize_reaction_schema(reaction)

    assert sanitized["ligands"] == [
        {"name": ligand_name, "symbol": "L6", "step": 2}
    ]
