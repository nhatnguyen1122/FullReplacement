import os
import unittest
from unittest.mock import patch

from openevolve.config import Config, load_config


class TestProviderDefaults(unittest.TestCase):
    def test_no_config_defaults_to_codestral(self):
        with patch.dict(os.environ, {"CODESTRAL_API_KEY": "test-codestral-key"}, clear=False):
            config = load_config(None)

        self.assertEqual(config.llm.provider, "codestral")
        self.assertEqual(config.llm.api_base, "https://api.mistral.ai/v1")
        self.assertEqual(config.llm.api_key, "test-codestral-key")
        self.assertEqual(config.llm.models[0].provider, "codestral")
        self.assertEqual(config.llm.models[0].name, "codestral-latest")
        self.assertEqual(config.llm.models[0].api_key, "test-codestral-key")

    def test_provider_config_adds_default_model(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}, clear=False):
            config = Config.from_dict({"llm": {"provider": "gemini"}})

        self.assertEqual(config.llm.api_base, "https://generativelanguage.googleapis.com/v1beta/openai/")
        self.assertEqual(config.llm.api_key, "test-gemini-key")
        self.assertEqual(len(config.llm.models), 1)
        self.assertEqual(config.llm.models[0].name, "gemini-2.5-flash")
        self.assertEqual(config.llm.models[0].api_key, "test-gemini-key")

    def test_model_level_provider_supports_mixed_models(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-openai-key", "MISTRAL_API_KEY": "test-mistral-key"},
            clear=False,
        ):
            config = Config.from_dict(
                {
                    "llm": {
                        "models": [
                            {"provider": "gpt", "name": "gpt-4.1", "weight": 0.5},
                            {"provider": "codestral", "name": "codestral-latest", "weight": 0.5},
                        ]
                    }
                }
            )

        self.assertEqual(config.llm.models[0].api_base, "https://api.openai.com/v1")
        self.assertEqual(config.llm.models[0].api_key, "test-openai-key")
        self.assertEqual(config.llm.models[1].api_base, "https://api.mistral.ai/v1")
        self.assertEqual(config.llm.models[1].api_key, "test-mistral-key")


if __name__ == "__main__":
    unittest.main()

