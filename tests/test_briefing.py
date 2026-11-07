import sys
from types import SimpleNamespace

from demandcast import briefing


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            usage=SimpleNamespace(input_tokens=500, output_tokens=90),
        )


def test_offline_briefing_uses_only_computed_numbers(art):
    for loc in art["locations"]:
        f = briefing.facts(art, loc)
        out = briefing.OfflineProvider().write(f)
        assert briefing.unsupported_numbers(out["text"], f) == []


def test_invented_number_is_caught(art):
    f = briefing.facts(art, "lodo")
    assert briefing.unsupported_numbers("Expect 987654 guests.", f) == ["987654"]


def test_claude_provider_accepts_grounded_text(art):
    f = briefing.facts(art, "lodo")
    text = f"Tomorrow looks like {f['forecast_units']} items at {f['location']}."
    fake = FakeClient(text)
    out = briefing.ClaudeProvider(client=fake).write(f)
    assert out["provider"] == "claude" and not out["fallback"] and out["text"] == text
    assert fake.calls[0]["model"] == "claude-opus-5-5"
    assert out["usage"]["output_tokens"] == 90


def test_claude_provider_falls_back_on_invented_number(art):
    f = briefing.facts(art, "lodo")
    out = briefing.ClaudeProvider(client=FakeClient("Prep 123456 bowls.")).write(f)
    assert out["fallback"] and out["provider"] == "offline" and out["unsupported_numbers"] == ["123456"]


def test_default_provider_is_offline_and_never_imports_anthropic(monkeypatch):
    monkeypatch.delenv("DEMANDCAST_BRIEFING_PROVIDER", raising=False)
    sys.modules.pop("anthropic", None)
    p = briefing.make_provider()
    assert p.name == "offline"
    assert "anthropic" not in sys.modules
