"""Input validation tests for the local RAG web API."""
import json
import unittest

from src.web_server import find_available_port, parse_chat_request, validate_chat_request


class WebRequestValidationTests(unittest.TestCase):
    def test_accepts_valid_request_and_normalizes_question(self):
        body = json.dumps({
            "question": "  What are the four functions?  ",
            "corpus": "nist",
            "top_k": 5,
            "budget": 1200,
        }).encode("utf-8")
        self.assertEqual(parse_chat_request(body), {
            "question": "What are the four functions?",
            "corpus": "nist",
            "top_k": 5,
            "budget": 1200,
        })

    def test_rejects_invalid_json_and_non_object_payloads(self):
        with self.assertRaises(ValueError):
            parse_chat_request(b"not-json")
        with self.assertRaises(ValueError):
            parse_chat_request(b"[]")

    def test_rejects_bad_corpus_types_and_limits(self):
        base = {"question": "Question?"}
        for payload in (
            {**base, "corpus": "other"},
            {**base, "question": ["not", "text"]},
            {**base, "top_k": True},
            {**base, "top_k": 21},
            {**base, "budget": 0},
            {**base, "budget": False},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_chat_request(payload)

    def test_port_finder_validates_arguments(self):
        with self.assertRaises(ValueError):
            find_available_port(start_port=0)
        with self.assertRaises(ValueError):
            find_available_port(max_attempts=0)


if __name__ == "__main__":
    unittest.main()
