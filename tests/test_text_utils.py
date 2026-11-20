"""Unit tests for the classical-NLP components:  python -m pytest -q"""
from src.eliza import eliza_reply, small_talk_reply
from src.routing import DOMAINS, route
from src.text_utils import Normalizer, min_edit_distance


def test_edit_distance():
    assert min_edit_distance("intention", "execution") == 5
    assert min_edit_distance("", "abc") == 3
    assert min_edit_distance("kitten", "sitting") == 3


def test_model_code_and_correction():
    n = Normalizer({"the", "water", "bridge"})
    out = n("bulasik makinem su boşaltmıyor, model DW-6534")
    assert out["model_codes"] == ["DW-6534"]
    assert ("bulasik", "bulaşık") in out["corrections"]
    assert "MODELCODE" in out["clean"]
    assert n("my dishwaher leaks")["corrections"] == [("dishwaher", "dishwasher")]
    # known words are never "corrected"
    assert n("traffic on the bridge in 2024")["corrections"] == []
    assert n("traffic on the bridge in 2024")["model_codes"] == []


def test_error_codes_and_repeats():
    out = Normalizer()("Washer shows E-07, pleaseeee help")
    assert out["error_codes"] == ["E-07"]
    assert "pleasee" in out["clean"]


def test_small_talk():
    assert small_talk_reply("merhaba") is not None
    assert small_talk_reply("hello, how are you?") is not None
    assert small_talk_reply("hello my dishwasher is broken") is None
    assert "angry" in eliza_reply("I feel angry about my oven")


def test_routing():
    assert sum(len(v) for v in DOMAINS.values()) == 156
    assert route("appliance_leaking")[1] == "Appliance Technical Service"
    assert route("oos")[1] == "Human agent"
