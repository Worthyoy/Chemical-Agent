from merge_filtered import convert_er_to_ee, parse_er_to_ee


def test_safe_er_formats_convert_to_ee():
    assert parse_er_to_ee("95:5") == ("90%", "converted_exact")
    assert parse_er_to_ee("95/5") == ("90%", "converted_exact")
    assert parse_er_to_ee("95:5 e.r.") == ("90%", "converted_exact")
    assert parse_er_to_ee("95:5 (1S, 5R, 6S)") == ("90%", "converted_exact")
    assert parse_er_to_ee(">99:1") == (">98%", "converted_threshold")
    assert parse_er_to_ee(">99:1 (3aS, 5R, 7aR)") == (
        ">98%",
        "converted_threshold",
    )


def test_ambiguous_or_missing_er_is_not_converted():
    assert parse_er_to_ee("96:4/88:12")[0] is None
    assert parse_er_to_ee("syn 96:4; anti 89:11")[0] is None
    assert parse_er_to_ee("N/A") == (None, "explicitly_unreported")


def test_existing_ee_is_never_overwritten_and_raw_er_is_preserved():
    targets = {"er": "95:5 (1S, 5R, 6S)", "ee": "88%"}
    assert convert_er_to_ee(targets) == "existing_ee"
    assert targets == {"er": "95:5 (1S, 5R, 6S)", "ee": "88%"}

    targets = {"er": ">99:1 (3aS, 5R, 7aR)"}
    assert convert_er_to_ee(targets) == "converted_threshold"
    assert targets == {"er": ">99:1 (3aS, 5R, 7aR)", "ee": ">98%"}
