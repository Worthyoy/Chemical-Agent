from batch_si_extractor import SIExtractor


def test_reaction_prompts_describe_name_symbol_separation():
    for prompt in (
        SIExtractor.STRUCTURED_REACTION_PROMPT,
        SIExtractor.NON_GP_REACTION_PROMPT,
        SIExtractor.MIXED_REACTION_PROMPT,
    ):
        assert "name stores the chemical name only" in prompt
        assert "symbol stores the reported label" in prompt
        assert "Chemical formulas, counterions, coordination fragments" in prompt
        assert "Do not infer symbol merely because" in prompt
        assert "carboxylic acid (34)" in prompt
        assert "dibromovinyl ... (35')" in prompt
        assert "methanol (36)" in prompt


def test_all_text_reaction_prompts_require_ligand_amount_or_null():
    for prompt in (
        SIExtractor.STRUCTURED_REACTION_PROMPT,
        SIExtractor.NON_GP_REACTION_PROMPT,
        SIExtractor.MIXED_REACTION_PROMPT,
    ):
        assert "ligand" in prompt.casefold()
        assert "amount=null" in prompt
        assert "preformed" in prompt.casefold()
        assert (
            "Do not infer a separate ligand amount" in prompt
            or "Do not copy a preformed catalyst-complex amount" in prompt
        )


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


def test_sanitize_splits_numeric_labels_but_preserves_unconfirmed_letter_codes():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = {
        "id": "r1",
        "source_pages": [35],
        "reaction_type": "test",
        "substrates": [
            {"name": "starting material (SM15)"},
            {"name": "numbered starting material (3a)"},
        ],
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

    assert sanitized["substrates"][0] == {"name": "starting material (SM15)"}
    assert sanitized["substrates"][1]["name"] == "numbered starting material"
    assert sanitized["substrates"][1]["symbol"] == "3a"
    assert sanitized["products"][0]["name"] == "reported dibromovinyl compound"
    assert sanitized["products"][0]["symbol"] == "35'"
    assert sanitized["products"][1]["name"] == "reported methanol"
    assert sanitized["products"][1]["symbol"] == "36"
    assert sanitized["ligands"][0] == {
        "name": "reported ligand (L1)",
        "amount": None,
    }
    assert sanitized["other_components"][0] == {"name": "reported reagent (3a)"}


def test_sanitize_preserves_formula_counterions_and_coordination_fragments():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reaction = {
        "id": "r1",
        "source_pages": [1],
        "reaction_type": "test",
        "substrates": [],
        "products": [],
        "catalysts": [
            {
                "name": "Ir[dF(CF3)ppy]2(dtbbpy)(PF6)",
                "symbol": None,
                "amount": "5.5 mg, 0.0050 mmol",
            },
            {"name": "reported metal complex (BF4)"},
        ],
        "ligands": [],
        "other_components": [{"name": "reported salt (ClO4)"}],
        "conditions": {},
        "targets": {"yield": None, "ee": None, "er": None},
    }

    sanitized = extractor.sanitize_reaction_schema(reaction)

    assert sanitized["catalysts"][0] == {
        "name": "Ir[dF(CF3)ppy]2(dtbbpy)(PF6)",
        "symbol": None,
        "amount": "5.5 mg, 0.0050 mmol",
    }
    assert sanitized["catalysts"][1] == {"name": "reported metal complex (BF4)"}
    assert sanitized["other_components"][0] == {"name": "reported salt (ClO4)"}


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
        {
            "id": "r1",
            "ligands": [
                {
                    "name": "L6",
                    "amount": "10.0 mg, 0.024 mmol, 12 mol%",
                    "step": 2,
                }
            ],
        }
    ]

    aligned = extractor.align_names_in_reactions(reactions, {"L6": ligand_name})

    assert aligned[0]["ligands"] == [
        {
            "name": ligand_name,
            "symbol": "L6",
            "amount": "10.0 mg, 0.024 mmol, 12 mol%",
            "step": 2,
            "resolution_source": "name_registry",
            "resolution_method": "same_paper_symbol",
            "resolution_confidence": "high",
        }
    ]


def test_registry_alignment_splits_matching_trailing_letter_code():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    full_name = "7-Methyl-1-(pent-4-enoyl)-1H-indole-3-carbaldehyde"

    aligned = extractor.align_names_in_reactions(
        [{"id": "r1", "products": [{"name": f"{full_name} (SM15)"}]}],
        {"SM15": full_name},
    )

    product = aligned[0]["products"][0]
    assert product["name"] == full_name
    assert product["symbol"] == "SM15"
    assert product["registry_match_status"] == "verified"


def test_registry_alignment_does_not_strip_counterion_on_name_mismatch():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    catalyst_name = "Ir[dF(CF3)ppy]2(dtbbpy)(PF6)"

    aligned = extractor.align_names_in_reactions(
        [{"id": "r1", "catalysts": [{"name": catalyst_name}]}],
        {"PF6": "hexafluorophosphate"},
    )

    assert aligned[0]["catalysts"] == [{"name": catalyst_name}]


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


def test_sanitize_preserves_resolved_ligand_identity_and_amount():
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
        {
            "name": ligand_name,
            "amount": "10.0 mg, 0.024 mmol, 12 mol%",
            "symbol": "L6",
            "step": 2,
        }
    ]


def test_registry_alignment_resolves_underspecified_substrate_series_label():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    registry_name = "Methyl substituted enamine"

    aligned = extractor.align_names_in_reactions(
        [
            {
                "id": "GeneralProcedureF-Entry11",
                "substrates": [
                    {
                        "name": "3",
                        "symbol": "3k",
                        "amount": "0.2 mmol, 1.0 equiv.",
                    }
                ],
            }
        ],
        {"3k": registry_name},
    )

    substrate = aligned[0]["substrates"][0]
    assert substrate == {
        "name": registry_name,
        "symbol": "3k",
        "amount": "0.2 mmol, 1.0 equiv.",
        "original_name": "3",
        "resolution_source": "name_registry",
        "resolution_method": "same_paper_symbol",
        "resolution_confidence": "high",
        "resolution_status": "resolved_from_registry",
        "resolution_evidence": {
            "match_type": "underspecified_series_label",
            "reported_name": "3",
            "reported_symbol": "3k",
            "registry_symbol": "3k",
        },
    }
    assert extractor.last_registry_resolution_stats["underspecified_series_labels"] == 1
    assert extractor._generic_substrate_resolution_candidate(aligned[0]) is None


def test_registry_alignment_replaces_prior_product_inference_and_stale_conflict():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    registry_name = "Methyl substituted enamine"
    aligned = extractor.align_names_in_reactions(
        [
            {
                "id": "GeneralProcedureF-Entry11",
                "substrates": [
                    {
                        "name": registry_name,
                        "symbol": "3k",
                        "original_name": "3",
                        "registry_name": registry_name,
                        "registry_match_status": "conflict",
                        "registry_conflict_reason": (
                            "extracted_name_differs_from_registry_symbol_name"
                        ),
                        "identity_status": "generic",
                        "resolution_status": "resolved_from_product",
                        "resolution_source": "product_name",
                        "resolution_method": "product_name_to_substrate_mapping",
                        "resolution_evidence": {"product_name": "product 4k"},
                    }
                ],
            }
        ],
        {"3k": registry_name},
    )

    substrate = aligned[0]["substrates"][0]
    assert substrate["name"] == registry_name
    assert substrate["symbol"] == "3k"
    assert substrate["original_name"] == "3"
    assert substrate["resolution_source"] == "name_registry"
    assert substrate["resolution_status"] == "resolved_from_registry"
    assert substrate["resolution_evidence"]["match_type"] == (
        "underspecified_series_label"
    )
    assert "registry_name" not in substrate
    assert "registry_match_status" not in substrate
    assert "registry_conflict_reason" not in substrate
    assert "identity_status" not in substrate


def test_registry_alignment_accepts_only_missing_letter_suffix():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    reactions = [
        {
            "id": "series-safety",
            "substrates": [
                {"name": "compound 3", "symbol": "3aa"},
                {"name": "3", "symbol": "3a"},
                {"name": "3", "symbol": "30"},
                {"name": "3j", "symbol": "3k"},
                {"name": "3", "symbol": "3z"},
            ],
        }
    ]
    registry = {
        "3aa": "compound three aa",
        "3a": "compound three a",
        "30": "compound thirty",
        "3k": "compound three k",
    }

    first = extractor.align_names_in_reactions(reactions, registry)
    substrates = first[0]["substrates"]
    assert substrates[0]["name"] == "compound three aa"
    assert substrates[0]["original_name"] == "compound 3"
    assert substrates[1]["name"] == "compound three a"
    assert substrates[1]["original_name"] == "3"
    assert substrates[2]["name"] == "3"
    assert substrates[2]["registry_match_status"] == "conflict"
    assert substrates[3]["name"] == "3j"
    assert substrates[3]["registry_match_status"] == "conflict"
    assert substrates[4] == {"name": "3", "symbol": "3z"}

    second = extractor.align_names_in_reactions(first, registry)
    assert second == first
