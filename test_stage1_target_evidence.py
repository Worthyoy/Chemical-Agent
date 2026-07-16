from batch_si_extractor import SIExtractor


def _chunk():
    pages = [
        {"page_num": page_num, "text": f"content for page {page_num}"}
        for page_num in range(42, 47)
    ]
    return {
        "chunk_id": 16,
        "page_range": "42-46",
        "page_nums": [42, 43, 44, 45, 46],
        "text": SIExtractor._format_chunk_text(None, pages),
    }


def test_stage1_relevant_pages_are_advisory_by_default():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    chunk = _chunk()

    selected = extractor._stage1_chunk_for_stage2(
        chunk,
        {"has_reactions": True, "relevant_pages": [42, 43, 45, 46]},
    )

    assert selected is chunk
    assert selected["page_nums"] == [42, 43, 44, 45, 46]
    assert "--- Page 44 ---" in selected["text"]


def test_legacy_stage1_page_trimming_requires_explicit_opt_in():
    extractor = SIExtractor(
        api_key="test",
        enable_stage2_audit=False,
        enable_stage1_page_trimming=True,
    )

    selected = extractor._stage1_chunk_for_stage2(
        _chunk(),
        {"has_reactions": True, "relevant_pages": [42, 43, 45, 46]},
    )

    assert selected["page_nums"] == [42, 43, 45, 46]
    assert "--- Page 44 ---" not in selected["text"]


def test_empty_stage1_relevant_pages_keep_the_full_chunk():
    for trimming_enabled in (False, True):
        extractor = SIExtractor(
            api_key="test",
            enable_stage2_audit=False,
            enable_stage1_page_trimming=trimming_enabled,
        )
        chunk = _chunk()

        selected = extractor._stage1_chunk_for_stage2(
            chunk,
            {"has_reactions": True, "relevant_pages": []},
        )

        assert selected["page_nums"] == [42, 43, 44, 45, 46]


def test_stage1_screen_response_preserves_reason():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)

    result = extractor._parse_screen_response(
        '{"has_reactions":true,"relevant_pages":[43,44],'
        '"reason":"page 44 contains cross-page ee evidence"}'
    )

    assert result == {
        "has_reactions": True,
        "relevant_pages": [43, 44],
        "reason": "page 44 contains cross-page ee evidence",
    }


def test_stage1_and_mixed_prompts_preserve_cross_page_target_evidence():
    screen_prompt = SIExtractor.SCREEN_PROMPT
    mixed_prompt = SIExtractor.MIXED_REACTION_PROMPT

    assert "continuation pages that report the yield, ee, er, or dr" in screen_prompt
    assert "reaction evidence, not analytical-only text" in screen_prompt
    assert "product entry may continue onto the following page" in mixed_prompt
    assert "source_pages must include both the product/preparation page" in mixed_prompt
    assert "Do not create a separate reaction from an HPLC table" in mixed_prompt
