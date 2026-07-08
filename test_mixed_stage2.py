from batch_si_extractor import SIExtractor


def _valid_gp_template(label):
    return {
        "status": "valid",
        "gp_id": label,
        "reaction_type": "template reaction",
        "substrates": [{"name": "template substrate"}],
        "catalysts": [{"name": "template catalyst"}],
        "ligands": [{"name": "template ligand"}],
        "other_components": [{"name": "template reagent"}],
        "conditions": {"solvent": "MeCN", "time": "1 h"},
    }


def test_gp_template_chunks_force_single_mixed_job_without_router():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    extractor.route_reaction_chunk = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("router should not be called")
    )
    extractor.select_gp_for_chunk = lambda *_args, **_kwargs: {
        "GeneralProcedureA": "A",
        "GeneralProcedureB": "B",
    }
    calls = []

    def _fake_run(chunk_text, chunk_label, job, gp_block, **_kwargs):
        calls.append({"job": dict(job), "gp_block": gp_block})
        return {
            "reactions": [{"id": "mixed_result", "products": [{"name": "product"}]}],
            "audit_recovered": 0,
            "error": None,
            "job_log": {
                "job_id": job.get("job_id"),
                "mode": job.get("mode"),
                "prompt_mode": "mixed_gp_non_gp",
                "gp_keys": list(job.get("gp_keys") or []),
                "reaction_count": 1,
                "reaction_ids": ["mixed_result"],
                "attempts": [],
            },
        }

    extractor._run_stage2_job = _fake_run

    result = extractor.stage2_extract_with_meta(
        "--- Page 1 ---\nGP and standalone reactions, 81% yield.",
        "chunk",
        gp_texts={"GeneralProcedureA": "A", "GeneralProcedureB": "B"},
        gp_templates={
            "GeneralProcedureA": _valid_gp_template("GeneralProcedureA"),
            "GeneralProcedureB": _valid_gp_template("GeneralProcedureB"),
        },
    )

    assert len(calls) == 1
    assert calls[0]["job"]["job_id"] == "mixed_1"
    assert calls[0]["job"]["mode"] == "mixed"
    assert calls[0]["job"]["gp_keys"] == ["GeneralProcedureA", "GeneralProcedureB"]
    assert "=== GeneralProcedureA ===" in calls[0]["gp_block"]
    assert "=== GeneralProcedureB ===" in calls[0]["gp_block"]
    assert result["dispatch"]["dispatch_mode"] == "gp_template_forced_mixed"
    assert result["dispatch"]["router_disabled"] is True
    assert result["jobs"][0]["prompt_mode"] == "mixed_gp_non_gp"
    assert len(result["reactions"]) == 1


def test_mixed_prompt_and_user_content_include_downstream_boundary_rules():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    prompt, prompt_mode = extractor._stage2_prompt_for_job({"mode": "mixed"})
    user_content = extractor._stage2_user_content_for_job(
        "--- Page 34 ---\ntext",
        "chunk",
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["GeneralProcedureC"]},
        gp_block="GENERAL PROCEDURE CONTEXT\n=== GeneralProcedureC ===\n{}",
    )

    assert prompt is extractor.MIXED_REACTION_PROMPT
    assert prompt_mode == "mixed_gp_non_gp"
    assert "standalone/downstream reaction" in prompt
    assert "Do not output the same concrete reaction twice as both GP and non-GP" in prompt
    assert "GP templates provide shared reaction_type" in prompt
    assert "reported literature procedure" in prompt
    assert "published procedure" in prompt
    assert "previously reported method" in prompt
    assert "Extract all qualifying entries in source order" in prompt
    assert '"source_pages":[18]' in prompt
    assert '"symbol":"29"' in prompt
    assert '"amount":"454 mg"' in prompt
    assert '"_gp_source":"GeneralProcedureC"' in prompt
    assert '"step_count":2' in prompt
    assert '"step":2' in prompt
    assert '"produced_in_step":1' in prompt
    assert '"consumed_in_step":2' in prompt
    assert "final isolated product" in prompt
    assert "step=step_count" in prompt
    assert "do not inherit GP" in prompt
    assert "standalone or downstream transformations as non-GP reactions" in user_content
    assert "without inheriting GP fields" in user_content
    assert "GENERAL PROCEDURE CONTEXT" in user_content


def test_mixed_template_retry_checks_only_reactions_with_gp_source():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    job = {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["GeneralProcedureA"]}
    gp_templates = {"GeneralProcedureA": _valid_gp_template("GeneralProcedureA")}
    reactions = [
        {
            "id": "standalone",
            "products": [{"name": "standalone product"}],
            "targets": {"yield": "70%"},
        },
        {
            "id": "gp_missing",
            "_gp_source": "GeneralProcedureA",
            "products": [{"name": "gp product"}],
            "targets": {"yield": "80%"},
        },
    ]

    message = extractor._detect_missing_gp_template_fields_for_retry(
        reactions,
        job,
        gp_templates,
    )

    assert message
    assert "gp_missing" in message
    assert "standalone" not in message


def test_mixed_target_retry_is_enabled():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    message = extractor._detect_missing_gp_targets_for_retry(
        "reported product, 81% yield. 1H NMR...",
        [{"id": "r1", "products": [{"name": "reported product"}], "targets": {}}],
        {"job_id": "mixed_1", "mode": "mixed"},
    )

    assert message
    assert "Previous output missed targets" in message


def test_mixed_schema_retry_mentions_multistep_step_repair_and_single_step_literature_procedure():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    user_content = extractor._stage2_user_content_for_job(
        "--- Page 11 ---\ntext",
        "chunk",
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["GeneralProcedureB"]},
        gp_block="GENERAL PROCEDURE CONTEXT\n=== GeneralProcedureB ===\n{}",
        previous_schema_error=(
            "multi-step products item has invalid step: "
            "{'name': 'product', 'symbol': 'SM1', 'amount': '578 mg'}"
        ),
    )

    assert "Final reported products usually use step=step_count" in user_content
    assert "single-step standalone literature-procedure entries, remove step_count" in user_content
    assert "every substrate/product/catalyst/ligand/other_component object must include integer step" in user_content


def test_no_gp_template_chunks_use_non_gp_prompt_without_gp_block():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    extractor.route_reaction_chunk = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("router should not be called")
    )
    extractor.select_gp_for_chunk = lambda *_args, **_kwargs: {}
    calls = []

    def _fake_run(chunk_text, chunk_label, job, gp_block, **_kwargs):
        calls.append({"job": dict(job), "gp_block": gp_block})
        return {
            "reactions": [],
            "audit_recovered": 0,
            "error": None,
            "job_log": {
                "job_id": job.get("job_id"),
                "mode": job.get("mode"),
                "prompt_mode": job.get("mode"),
                "gp_keys": list(job.get("gp_keys") or []),
                "reaction_count": 0,
                "reaction_ids": [],
                "attempts": [],
            },
        }

    extractor._run_stage2_job = _fake_run

    result = extractor.stage2_extract_with_meta(
        "--- Page 2 ---\ntext",
        "non gp chunk",
        gp_texts={},
        gp_templates={},
    )

    assert len(calls) == 1
    assert calls[0]["job"]["mode"] == "non_gp"
    assert calls[0]["job"]["job_id"] == "non_gp_1"
    assert calls[0]["gp_block"] == ""
    assert result["dispatch"]["dispatch_mode"] == "non_gp_only"
    assert result["dispatch"]["router_disabled"] is True
