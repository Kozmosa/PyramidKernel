import unittest
from unittest.mock import patch

from stark.providers import ClaudeCompatibleConfig, ClaudeCompatibleProvider


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = "ok"

    def json(self):
        return self._payload



class ClaudeProviderTests(unittest.TestCase):
    def test_messages_endpoint_uses_v1_messages(self):
        provider = ClaudeCompatibleProvider(
            ClaudeCompatibleConfig(
                api_key="test-key",
                base_url="https://api.anthropic.com",
                model="claude-sonnet-4-6",
            )
        )
        self.assertEqual(provider._messages_endpoint(), "https://api.anthropic.com/v1/messages")

    def test_post_json_adds_anthropic_version_header(self):
        provider = ClaudeCompatibleProvider(
            ClaudeCompatibleConfig(
                api_key="test-key",
                base_url="https://api.anthropic.com",
                model="claude-sonnet-4-6",
                api_version="2023-06-01",
            )
        )
        captured = {}

        def _fake_post(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return _DummyResponse({"content": [{"text": "ok"}]})

        with patch("stark.providers.http_utils.requests.post", side_effect=_fake_post):
            payload = provider._post_json(
                provider._messages_endpoint(),
                {"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hello"}]},
            )

        headers = {key.lower(): value for key, value in captured["kwargs"]["headers"].items()}
        self.assertEqual(payload["content"][0]["text"], "ok")
        self.assertEqual(captured["url"], provider._messages_endpoint())
        self.assertEqual(captured["kwargs"]["timeout"][1], provider.config.timeout_seconds)
        self.assertEqual(headers["authorization"], "Bearer test-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")


if __name__ == "__main__":
    unittest.main()
