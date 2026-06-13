import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from openevolve.llm.openai import OpenAILLM
from openevolve.llm.token_budget import (
    TokenBudgetExceeded,
    configure_token_budget,
    get_token_usage,
    record_token_usage,
)


class TestTokenBudget(unittest.TestCase):
    def tearDown(self):
        configure_token_budget(max_total_tokens=None)

    def test_records_usage(self):
        configure_token_budget(max_total_tokens=None)

        usage = record_token_usage(prompt_tokens=10, completion_tokens=5)

        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 5)
        self.assertEqual(usage.total_tokens, 15)
        self.assertEqual(usage.requests, 1)

    def test_raises_when_budget_exceeded(self):
        configure_token_budget(max_total_tokens=10)

        with self.assertRaises(TokenBudgetExceeded):
            record_token_usage(prompt_tokens=8, completion_tokens=3)

        usage = get_token_usage()
        self.assertEqual(usage.total_tokens, 11)

    def test_openai_llm_records_response_usage_object(self):
        configure_token_budget(max_total_tokens=None)
        model_cfg = Mock()
        model_cfg.name = "test-model"
        model_cfg.system_message = "system"
        model_cfg.temperature = 0.7
        model_cfg.top_p = 0.95
        model_cfg.max_tokens = 100
        model_cfg.timeout = 60
        model_cfg.retries = 0
        model_cfg.retry_delay = 0
        model_cfg.api_base = "https://api.openai.com/v1"
        model_cfg.api_key = "test-key"
        model_cfg.random_seed = None
        model_cfg.reasoning_effort = None
        model_cfg.manual_mode = False
        model_cfg.track_token_usage = True

        llm = OpenAILLM(model_cfg)
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19)
        )

        llm._record_response_usage(response)

        usage = get_token_usage()
        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.completion_tokens, 7)
        self.assertEqual(usage.total_tokens, 19)
        self.assertEqual(usage.requests, 1)


if __name__ == "__main__":
    unittest.main()

