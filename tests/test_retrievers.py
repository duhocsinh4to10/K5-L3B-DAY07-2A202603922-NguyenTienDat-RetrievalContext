"""Unit tests for Day 7 retrievers, context packing, and agent components."""
import unittest
from src.retrievers import BM25Retriever, pack_context, tokenize
from src.evaluation import hit


class TestRetrievers(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                'id': 'chunk-1',
                'source_id': 'doc-1',
                'source': 'test.pdf',
                'page': 1,
                'text': 'The quick brown fox jumps over the lazy dog.',
                'metadata': {'section': 'intro'}
            },
            {
                'id': 'chunk-2',
                'source_id': 'doc-1',
                'source': 'test.pdf',
                'page': 2,
                'text': 'Artificial intelligence risk management framework includes GOVERN, MAP, MEASURE, MANAGE.',
                'metadata': {'section': 'core'}
            },
            {
                'id': 'chunk-3',
                'source_id': 'doc-1',
                'source': 'test.pdf',
                'page': 3,
                'text': 'Sustainable development goals address climate action, biodiversity, and clean energy.',
                'metadata': {'section': 'sdg'}
            }
        ]
        self.bm25 = BM25Retriever(self.chunks)

    def test_tokenize(self):
        tokens = tokenize("Hello, World! 123")
        self.assertEqual(tokens, ['hello', 'world', '123'])

    def test_bm25_search(self):
        results = self.bm25.search("risk management framework", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['id'], 'chunk-2')

    def test_bm25_filter(self):
        results = self.bm25.search("goals", top_k=2, filters={'section': 'sdg'})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['id'], 'chunk-3')

    def test_pack_context(self):
        results = [self.chunks[0], self.chunks[1]]
        ctx, ids = pack_context(results, budget=200)
        self.assertTrue(len(ids) >= 1)
        self.assertTrue(len(ctx) <= 200)
        self.assertIn('chunk-1', ids)

    def test_hit_evaluation(self):
        gold = [{'source_id': 'doc-1', 'page': 2, 'evidence': 'GOVERN, MAP, MEASURE, MANAGE'}]
        self.assertTrue(hit(self.chunks[1], gold))
        self.assertFalse(hit(self.chunks[0], gold))


if __name__ == '__main__':
    unittest.main()
