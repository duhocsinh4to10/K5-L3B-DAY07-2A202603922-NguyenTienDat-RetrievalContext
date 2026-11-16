"""Offline tests for Gemini action parsing and the local evidence fallback."""
import json
import unittest
from unittest.mock import Mock, patch

import requests

from src.agent import LLMController


def history(context: str, chunk_id: str = "c1"):
    return [{
        "tool": "search",
        "queries": ["query"],
        "context": f"[{chunk_id} | report.pdf | page 1]\n{context}",
        "context_ids": [chunk_id],
    }]


class GeminiControllerTests(unittest.TestCase):
    def test_first_decision_searches_without_calling_api(self):
        controller = LLMController(api_key="")
        with patch("src.agent.requests.post") as post:
            action = controller.decide("  What is risk management?  ", [])
        self.assertEqual(action, {"action": "search", "query": "What is risk management"})
        post.assert_not_called()

    def test_valid_gemini_json_action_and_usage_are_parsed(self):
        controller = LLMController(api_key="test-key")
        response = Mock()
        response.json.return_value = {
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8},
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "action": "answer",
                "answer": "Risks are prioritized by impact [c1].",
                "citations": ["c1"],
            })}]}}],
        }
        with patch("src.agent.requests.post", return_value=response) as post:
            action = controller.decide("How are risks prioritized?", history("Risks are prioritized."))

        self.assertEqual(action["action"], "answer")
        self.assertEqual(action["citations"], ["c1"])
        self.assertEqual(controller.call_count, 1)
        self.assertEqual(controller.total_prompt_tokens, 20)
        self.assertEqual(controller.total_candidates_tokens, 8)
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "test-key")

    def test_missing_api_key_uses_only_a_matching_source_sentence(self):
        controller = LLMController(api_key="")
        action = controller.decide(
            "How should risks be prioritized?",
            history("Risks are prioritized based on impact, likelihood, and resources."),
        )
        self.assertEqual(action["action"], "answer")
        self.assertEqual(action["citations"], ["c1"])
        self.assertIn("Risks are prioritized", action["answer"])
        self.assertEqual(controller.call_count, 0)

    def test_missing_evidence_abstains_instead_of_inventing_an_answer(self):
        controller = LLMController(api_key="")
        action = controller.decide(
            "What is the secret launch code?",
            history("The report describes sustainable development goals."),
        )
        self.assertEqual(action["action"], "answer")
        self.assertEqual(action["citations"], [])
        self.assertIn("Không đủ bằng chứng", action["answer"])

    def test_network_failure_is_recorded_and_falls_back_safely(self):
        controller = LLMController(api_key="test-key")
        with patch("src.agent.requests.post", side_effect=requests.ConnectionError("offline")):
            action = controller.decide(
                "How should risks be prioritized?",
                history("Risks are prioritized based on impact and likelihood."),
            )
        self.assertEqual(action["citations"], ["c1"])
        self.assertIn("ConnectionError", controller.last_error)


if __name__ == "__main__":
    unittest.main()
