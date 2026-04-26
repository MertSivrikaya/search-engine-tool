import unittest
from src.search import SearchEngine

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        # A tiny, predictable mock registry
        self.mock_registry = {
            1: "https://example.com/quote1",
            2: "https://example.com/quote2",
            3: "https://example.com/quote3"
        }
        
        # A mock inverted index using our optimized [frequency, [positions]] structure
        self.mock_index = {
            "hello": {
                1: [1, [0]],       # Doc 1: "hello" at pos 0
                2: [1, [0]],       # Doc 2: "hello" at pos 0
            },
            "world": {
                1: [1, [1]],       # Doc 1: "world" at pos 1  (Phrase: "hello world")
                2: [1, [2]],       # Doc 2: "world" at pos 2  (Gap: "hello beautiful world")
                3: [1, [0]]        # Doc 3: "world" at pos 0
            },
            "_EXTENT_h1": {
                2: [[0, 0]]        # Doc 2 has an <h1> tag around "hello" at pos 0
            }
        }
        
        self.searcher = SearchEngine(self.mock_index, self.mock_registry)

    def test_tokenization(self):
        """Tests that the searcher applies the same stemming/cleaning as the indexer."""
        tokens = self.searcher._tokenize_query("Hello, Worlds!!!")
        # 'Hello' -> 'hello', 'Worlds' -> 'world' (stemmed)
        self.assertEqual(tokens, ["hello", "world"])

    def test_conjunctive_processing(self):
        """Tests that documents must contain ALL query terms (Boolean AND)."""
        # "hello world" -> Doc 1 and 2 have both. Doc 3 only has "world".
        results = self.searcher.find("hello world")
        
        # Should only return 2 documents
        self.assertEqual(len(results), 2)
        
        # Extract just the URLs from the result tuples: [(score, url), ...]
        urls = [res[1] for res in results]
        self.assertIn("https://example.com/quote1", urls)
        self.assertIn("https://example.com/quote2", urls)
        self.assertNotIn("https://example.com/quote3", urls)

    def test_zone_boosting(self):
        """Tests that words inside HTML extents receive a score multiplier."""
        # Search just for "hello". 
        # Both Doc 1 and Doc 2 have tf=1.
        # But Doc 2 has "hello" inside an _EXTENT_h1 (multiplier 2.0).
        results = self.searcher.find("hello")
        
        doc2_score = next(res[0] for res in results if res[1] == "https://example.com/quote2")
        doc1_score = next(res[0] for res in results if res[1] == "https://example.com/quote1")
        
        # Doc 2's score should be exactly double Doc 1's score due to the 2.0 multiplier
        self.assertEqual(doc2_score, doc1_score * 2.0)
        
    def test_exact_phrase_proximity(self):
        """Tests that sequential words receive the 2.0x proximity boost."""
        # Search "hello world". 
        # Doc 1 has them at positions [0] and [1] -> Sequential!
        # Doc 2 has them at positions [0] and [2] -> Gap!
        
        # Even though Doc 2 has an h1 boost on "hello", the exact phrase boost (2.0x applied
        # to the ENTIRE document score) on Doc 1 should make it highly competitive.
        # Let's directly verify the phrase logic itself:
        is_phrase_doc1 = self.searcher._check_exact_phrase(["hello", "world"], 1)
        is_phrase_doc2 = self.searcher._check_exact_phrase(["hello", "world"], 2)
        
        self.assertTrue(is_phrase_doc1)
        self.assertFalse(is_phrase_doc2)

if __name__ == '__main__':
    unittest.main()