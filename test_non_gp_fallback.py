from types import SimpleNamespace

from batch_si_extractor import SIExtractor


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        message = SimpleNamespace(content="not valid json")
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_non_gp_split_fallback_is_disabled_by_default():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    extractor.client = _FakeClient()
    fallback_calls = {"count": 0}

    def _fail_if_called(*_args, **_kwargs):
        fallback_calls["count"] += 1
        return {"reactions": [{"id": "fallback"}], "audit_recovered": 0, "job_logs": []}

    extractor._retry_non_gp_job_by_page = _fail_if_called

    result = extractor._run_stage2_job(
        chunk_text="--- Page 18 ---\nA standalone reaction paragraph, 81% yield.",
        chunk_label="test chunk",
        job={"job_id": "non_gp_1", "mode": "non_gp", "gp_keys": []},
        gp_block="",
        allowed_page_nums=[18],
    )

    assert result["error"] == "stage2_failed_after_retries"
    assert result["reactions"] == []
    assert fallback_calls["count"] == 0
    assert len(result["job_log"]["attempts"]) == 3


def test_non_gp_split_fallback_can_still_be_enabled_explicitly():
    extractor = SIExtractor(api_key="test", enable_stage2_audit=False)
    extractor.client = _FakeClient()
    fallback_calls = {"count": 0}

    def _fallback(*_args, **_kwargs):
        fallback_calls["count"] += 1
        return {"reactions": [{"id": "fallback"}], "audit_recovered": 0, "job_logs": []}

    extractor._retry_non_gp_job_by_page = _fallback

    result = extractor._run_stage2_job(
        chunk_text="--- Page 18 ---\nA standalone reaction paragraph, 81% yield.",
        chunk_label="test chunk",
        job={"job_id": "non_gp_1", "mode": "non_gp", "gp_keys": []},
        gp_block="",
        allowed_page_nums=[18],
        allow_non_gp_split_fallback=True,
    )

    assert result["error"] is None
    assert result["reactions"] == [{"id": "fallback"}]
    assert fallback_calls["count"] == 1
