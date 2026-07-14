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


def _empty_gp_template(label="EmptyGP"):
    return {
        "status": "valid",
        "gp_id": label,
        "reaction_type": "title-only GP",
        "substrates": [],
        "catalysts": [],
        "ligands": [],
        "other_components": [],
        "intermediates": [],
        "conditions": {"solvent": None, "time": None},
    }


def test_gp_template_inheritance_info_distinguishes_empty_and_useful_templates():
    empty_info = SIExtractor._gp_template_inheritance_info(_empty_gp_template())
    useful_info = SIExtractor._gp_template_inheritance_info(_valid_gp_template("UsefulGP"))
    time_info = SIExtractor._gp_template_inheritance_info({
        "status": "valid",
        "gp_id": "TimeGP",
        "conditions": {"time": "12 h"},
    })
    reaction_type_only_info = SIExtractor._gp_template_inheritance_info({
        "status": "valid",
        "gp_id": "TitleGP",
        "reaction_type": "some reaction",
        "procedure_details": [{"detail": "text"}],
    })

    assert empty_info == {"inheritable": False, "score": 0, "reasons": []}
    assert useful_info["inheritable"] is True
    assert "substrates" in useful_info["reasons"]
    assert "catalysts" in useful_info["reasons"]
    assert any(reason.startswith("conditions.") for reason in useful_info["reasons"])
    assert time_info["inheritable"] is True
    assert "conditions.time" in time_info["reasons"]
    assert reaction_type_only_info["inheritable"] is False


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
    assert result["dispatch"]["dispatch_mode"] == "mixed_with_gp_templates"
    assert result["dispatch"]["router_disabled"] is True
    assert result["dispatch"]["non_gp_prompt_disabled"] is True
    assert result["jobs"][0]["prompt_mode"] == "mixed_gp_non_gp"
    assert len(result["reactions"]) == 1


def test_non_inheritable_gp_templates_are_not_injected_into_mixed_job():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    extractor.select_gp_for_chunk = lambda *_args, **_kwargs: {
        "EmptyGP": "title-only GP text",
        "UsefulGP": "useful GP text",
    }
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
                "prompt_mode": "mixed_gp_non_gp",
                "gp_keys": list(job.get("gp_keys") or []),
                "reaction_count": 0,
                "reaction_ids": [],
                "attempts": [],
            },
        }

    extractor._run_stage2_job = _fake_run

    result = extractor.stage2_extract_with_meta(
        "--- Page 9 ---\nproduct series, 88% yield.",
        "chunk",
        gp_texts={"EmptyGP": "title-only GP text", "UsefulGP": "useful GP text"},
        gp_templates={
            "EmptyGP": _empty_gp_template("EmptyGP"),
            "UsefulGP": _valid_gp_template("UsefulGP"),
        },
    )

    assert calls[0]["job"]["mode"] == "mixed"
    assert calls[0]["job"]["gp_keys"] == ["UsefulGP"]
    assert "=== UsefulGP ===" in calls[0]["gp_block"]
    assert "=== EmptyGP ===" not in calls[0]["gp_block"]
    assert result["dispatch"]["selected_gp_keys_raw"] == ["EmptyGP", "UsefulGP"]
    assert result["dispatch"]["selected_gp_keys_injected"] == ["UsefulGP"]
    assert result["dispatch"]["skipped_non_inheritable_gp_templates"] == ["EmptyGP"]
    assert result["dispatch"]["gp_template_inheritance"]["EmptyGP"]["inheritable"] is False
    assert result["dispatch"]["gp_template_inheritance"]["UsefulGP"]["inheritable"] is True


def test_all_non_inheritable_selected_gp_templates_dispatch_as_non_gp():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    extractor.select_gp_for_chunk = lambda *_args, **_kwargs: {
        "EmptyGP": "title-only GP text",
    }
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
                "prompt_mode": "mixed_gp_non_gp",
                "gp_keys": list(job.get("gp_keys") or []),
                "reaction_count": 0,
                "reaction_ids": [],
                "attempts": [],
            },
        }

    extractor._run_stage2_job = _fake_run

    result = extractor.stage2_extract_with_meta(
        "--- Page 9 ---\nproduct series, 88% yield.",
        "chunk",
        gp_texts={"EmptyGP": "title-only GP text"},
        gp_templates={"EmptyGP": _empty_gp_template("EmptyGP")},
    )

    assert calls[0]["job"]["job_id"] == "mixed_1"
    assert calls[0]["job"]["mode"] == "mixed"
    assert calls[0]["job"]["gp_keys"] == []
    assert calls[0]["gp_block"] == ""
    assert result["dispatch"]["dispatch_mode"] == "mixed_without_gp_templates"
    assert result["dispatch"]["non_gp_prompt_disabled"] is True
    assert result["dispatch"]["selected_gp_keys_raw"] == ["EmptyGP"]
    assert result["dispatch"]["selected_gp_keys_injected"] == []
    assert result["dispatch"]["skipped_non_inheritable_gp_templates"] == ["EmptyGP"]
    assert result["dispatch"]["all_selected_gp_templates_non_inheritable"] is True


def test_forced_mixed_job_receives_gp_selection_debug():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    def _select_gp(_chunk_text, _gp_texts):
        extractor.last_gp_selection_debug = {
            "mode": "llm_resolved_no_reference",
            "selected_gp_keys": ["GeneralProcedureUnlabeled1"],
        }
        return {"GeneralProcedureUnlabeled1": "GP text"}

    extractor.select_gp_for_chunk = _select_gp
    calls = []

    def _fake_run(chunk_text, chunk_label, job, gp_block, **kwargs):
        calls.append(kwargs)
        return {
            "reactions": [],
            "audit_recovered": 0,
            "error": None,
            "job_log": {
                "job_id": job.get("job_id"),
                "mode": job.get("mode"),
                "prompt_mode": "mixed_gp_non_gp",
                "gp_keys": list(job.get("gp_keys") or []),
                "reaction_count": 0,
                "reaction_ids": [],
                "attempts": [],
            },
        }

    extractor._run_stage2_job = _fake_run

    result = extractor.stage2_extract_with_meta(
        "--- Page 9 ---\nproduct series, 88% yield.",
        "chunk",
        gp_texts={"GeneralProcedureUnlabeled1": "GP text"},
        gp_templates={"GeneralProcedureUnlabeled1": _valid_gp_template("GeneralProcedureUnlabeled1")},
    )

    assert calls[0]["gp_selection_debug"]["mode"] == "llm_resolved_no_reference"
    assert result["gp_selection_debug"]["mode"] == "llm_resolved_no_reference"


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
    assert "GP role/step topology > entry surface wording" in prompt
    assert "within the same role" in prompt
    assert "concrete entry text override > canonical GP template default" not in prompt
    assert '"with X", "charged with X", or "using X" does not by itself establish substrate role' in prompt
    assert "must not replace a template external substrate" in prompt
    assert "generated within the sequence and consumed later is an internal intermediate" in prompt
    assert "do not duplicate X across substrates and intermediates" in prompt
    assert '"reactive species from label"' in prompt
    assert "otherwise do not guess the symbol assignment" in prompt
    assert "Change the template role topology only when the concrete entry explicitly changes the reaction boundary" in prompt
    assert "Do not remove unrelated template components merely because the entry does not repeat them" in prompt
    assert "class-level identity, role description, short label, or otherwise incomplete identity" in prompt
    assert "do not guess from textual similarity" in prompt
    assert "Do not keep both a generic template item and its more specific concrete-entry form" in prompt
    assert '"name":"catalyst label"' in prompt
    assert '"name":"full reported catalyst identity"' in prompt
    assert "organocatalyst 2d" not in prompt
    assert "reported literature procedure" in prompt
    assert "published procedure" in prompt
    assert "previously reported method" in prompt
    assert "product characterization entries" in prompt
    assert "same GP section or product series" in prompt
    assert 'does not explicitly say "according to General Procedure"' in prompt
    assert "set _gp_source" in prompt
    assert "copy the GP template shared fields" in prompt
    assert "multiple supplied GP templates" in prompt
    assert "meaningful shared reaction fields" in prompt
    assert "Do not use a supplied GP template as _gp_source if it has no meaningful shared fields" in prompt
    assert "another supplied template provides the actual substrates/conditions" in prompt
    assert "empty or title-only template" in prompt
    assert "literature/published/standalone procedure entry does not inherit GP" in prompt
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
    assert "GP role/step topology > entry surface wording" in user_content
    assert "within the same role" in user_content
    assert "concrete entry text override > canonical GP template default" not in user_content
    assert "charged with X, or using X does not by itself make X an external substrate" in user_content
    assert "Never replace an external template substrate" in user_content
    assert "keep internally generated species in intermediates" in user_content
    assert "Replace a matching generic, label-level, or incomplete template component" in user_content
    assert "preserve unrelated template fields that the entry does not mention" in user_content
    assert "GENERAL PROCEDURE CONTEXT" in user_content


def test_mixed_prompt_has_role_first_lineage_and_reaction_boundary_examples():
    prompt = SIExtractor.MIXED_REACTION_PROMPT

    assert "Template: substrate class A at step 1" in prompt
    assert 'Entry: "procedure followed with B7 and reactive X derived from A3"' in prompt
    assert "substrates contain A with symbol A3 at step 1" in prompt
    assert "substrate class B at step 2" in prompt
    assert "X is not a substrate" in prompt
    assert "pre-prepared or isolated X" in prompt
    assert "extract that new reaction with X as its external substrate" in prompt
    assert "Merely saying \"X from A7\" or \"X derived from A7\" does not omit the upstream step" in prompt
    assert "silently reconcile roles in this order" in prompt
    assert "Do not output this reasoning or extra audit fields" in prompt


def test_gp_injection_template_uses_role_first_precedence():
    template = SIExtractor.GP_INJECTION_TEMPLATE

    assert "authoritative for the reaction boundary, chemical roles" in template
    assert "GP role/step topology > entry surface wording" in template
    assert "Within an already matched role" in template
    assert "pre-prepared or isolated material" in template
    assert "GP-generated intermediates" in template


def test_mixed_prompt_requires_role_and_cross_page_target_self_check():
    prompt = SIExtractor.MIXED_REACTION_PROMPT

    assert "supplied from outside the current reaction sequence" in prompt
    assert "generated within the supplied GP sequence" in prompt
    assert "no internally generated material or alias is duplicated" in prompt
    assert "every page supplying a non-null yield, ee, er, or dr value" in prompt
    assert "included in source_pages" in prompt


def test_mixed_user_content_mentions_no_reference_gp_selection_when_applicable():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    user_content = extractor._stage2_user_content_for_job(
        "--- Page 9 ---\nproduct series entries, 88% yield.",
        "chunk",
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["GeneralProcedureUnlabeled1"]},
        gp_block="GENERAL PROCEDURE CONTEXT\n=== GeneralProcedureUnlabeled1 ===\n{}",
        gp_selection_debug={
            "mode": "llm_resolved_no_reference",
            "selected_gp_keys": ["GeneralProcedureUnlabeled1"],
        },
    )

    assert "GP selection determined that this chunk belongs to the supplied GP" in user_content
    assert "entries may not explicitly name the GP" in user_content
    assert "GP-continuation reactions" in user_content
    assert "separate literature, published, or standalone procedure" in user_content


def test_mixed_user_content_mentions_multiple_gp_template_choice_for_no_reference_selection():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    user_content = extractor._stage2_user_content_for_job(
        "--- Page 9 ---\nproduct series entries, 88% yield.",
        "chunk",
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["EmptyTitleGP", "UsefulGP"]},
        gp_block="GENERAL PROCEDURE CONTEXT\n=== EmptyTitleGP ===\n{}\n=== UsefulGP ===\n{}",
        gp_selection_debug={
            "mode": "llm_resolved_no_reference",
            "selected_gp_keys": ["EmptyTitleGP", "UsefulGP"],
        },
    )

    assert "Multiple GP templates may be supplied" in user_content
    assert "actual shared substrates, catalysts, reagents, and conditions" in user_content
    assert "Do not choose an empty or title-only template" in user_content


def test_mixed_user_content_does_not_add_no_reference_note_for_explicit_selection():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    user_content = extractor._stage2_user_content_for_job(
        "--- Page 9 ---\nPrepared according to General Procedure A.",
        "chunk",
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["GeneralProcedureA"]},
        gp_block="GENERAL PROCEDURE CONTEXT\n=== GeneralProcedureA ===\n{}",
        gp_selection_debug={
            "mode": "explicit_match",
            "selected_gp_keys": ["GeneralProcedureA"],
        },
    )

    assert "GP selection determined that this chunk belongs to the supplied GP" not in user_content


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


def test_empty_substrate_gp_continuation_retry_for_selected_gp_product_result():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    message = extractor._detect_empty_substrate_gp_continuation_for_retry(
        [
            {
                "id": "NonGP-Entry1",
                "substrates": [],
                "products": [{"name": "product", "symbol": "3ma"}],
                "targets": {"yield": "88%"},
            }
        ],
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["GeneralProcedureUnlabeled1"]},
        {"mode": "llm_resolved_no_reference"},
    )

    assert message
    assert "GP selector selected a supplied GP" in message
    assert "llm_resolved_no_reference" in message
    assert "empty-substrate NonGP reactions" in message
    assert "set _gp_source and copy the GP template shared fields" in message
    assert "NonGP-Entry1" in message


def test_empty_substrate_gp_continuation_retry_ignores_gp_source_and_complete_standalone():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    message = extractor._detect_empty_substrate_gp_continuation_for_retry(
        [
            {
                "id": "gp_empty_substrates",
                "_gp_source": "GeneralProcedureUnlabeled1",
                "substrates": [],
                "products": [{"name": "product", "symbol": "3ma"}],
                "targets": {"yield": "88%"},
            },
            {
                "id": "standalone_ok",
                "substrates": [{"name": "reported substrate"}],
                "products": [{"name": "standalone product"}],
                "targets": {"yield": "70%"},
            },
        ],
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["GeneralProcedureUnlabeled1"]},
        {"mode": "llm_resolved_no_reference"},
    )

    assert message
    assert "gp_empty_substrates" in message
    assert "standalone_ok" not in message


def test_empty_substrate_gp_continuation_retry_ignores_complete_gp_source_and_complete_standalone():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    message = extractor._detect_empty_substrate_gp_continuation_for_retry(
        [
            {
                "id": "gp_ok",
                "_gp_source": "GeneralProcedureUnlabeled1",
                "substrates": [{"name": "Aldehyde"}],
                "products": [{"name": "product", "symbol": "3ma"}],
                "targets": {"yield": "88%"},
            },
            {
                "id": "standalone_ok",
                "substrates": [{"name": "reported substrate"}],
                "products": [{"name": "standalone product"}],
                "targets": {"yield": "70%"},
            },
        ],
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["GeneralProcedureUnlabeled1"]},
        {"mode": "llm_resolved_no_reference"},
    )

    assert message is None


def test_empty_substrate_retry_mentions_switching_from_empty_gp_source_template():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    message = extractor._detect_empty_substrate_gp_continuation_for_retry(
        [
            {
                "id": "r1",
                "_gp_source": "EmptyTitleGP",
                "substrates": [],
                "products": [{"name": "product", "symbol": "3ma"}],
                "targets": {"yield": "88%"},
            }
        ],
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["EmptyTitleGP", "UsefulGP"]},
        {"mode": "llm_resolved_no_reference"},
    )

    assert message
    assert "chosen _gp_source template has empty shared fields" in message
    assert "switch to another supplied GP template" in message
    assert "actual shared substrates/conditions" in message


def test_uninjected_gp_source_retry_rejects_skipped_template():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    message = extractor._detect_uninjected_gp_source_for_retry(
        [
            {
                "id": "r1",
                "_gp_source": "EmptyTitleGP",
                "substrates": [{"name": "Aldehyde"}],
                "products": [{"name": "product", "symbol": "3ma"}],
                "targets": {"yield": "88%"},
            }
        ],
        {"job_id": "mixed_1", "mode": "mixed", "gp_keys": ["UsefulGP"]},
        {"skipped_non_inheritable_gp_templates": ["EmptyTitleGP"]},
    )

    assert message
    assert "not injected for this job" in message
    assert "non-inheritable or title-only templates" in message
    assert "EmptyTitleGP" in message
    assert "UsefulGP" not in message


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


def test_no_gp_template_chunks_use_mixed_prompt_without_gp_block():
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
                "prompt_mode": "mixed_gp_non_gp",
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
    assert calls[0]["job"]["mode"] == "mixed"
    assert calls[0]["job"]["job_id"] == "mixed_1"
    assert calls[0]["gp_block"] == ""
    assert result["dispatch"]["dispatch_mode"] == "mixed_without_gp_templates"
    assert result["dispatch"]["router_disabled"] is True
    assert result["dispatch"]["non_gp_prompt_disabled"] is True
    assert result["jobs"][0]["prompt_mode"] == "mixed_gp_non_gp"
