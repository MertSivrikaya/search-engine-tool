import unittest
import tempfile
import json
import os

from src.search import SearchEngine
from src.indexer import Indexer


class TestSearchEngine(unittest.TestCase):
    """Test suite for the SearchEngine, verifying ranking algorithms and query processing."""
    
    def setUp(self) -> None:
        """
        Initializes a predictable mock registry and inverted index for testing math logic.
        
        Bypasses the HTML crawler/indexer entirely to strictly test the Vector Space 
        math, TF-IDF calculation, and proximity Boolean logic.
        """

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

    def test_tokenization(self) -> None:
        """Tests that the searcher applies the same stemming/cleaning as the indexer."""
        tokens = self.searcher._tokenize_query("Hello, Worlds!!!")
        # 'Hello' -> 'hello', 'Worlds' -> 'world' (stemmed)
        self.assertEqual(tokens, ["hello", "world"])

    def test_conjunctive_processing(self) -> None:
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

    def test_zone_boosting(self) -> None:
        """Tests that words inside HTML extents receive a score multiplier."""

        # Search just for "hello". 
        # Both Doc 1 and Doc 2 have tf=1.
        # But Doc 2 has "hello" inside an _EXTENT_h1 (multiplier 2.0).
        results = self.searcher.find("hello")
        
        doc2_score = next(res[0] for res in results if res[1] == "https://example.com/quote2")
        doc1_score = next(res[0] for res in results if res[1] == "https://example.com/quote1")
        
        # Doc 2's score should be exactly double Doc 1's score due to the 2.0 multiplier
        self.assertEqual(doc2_score, doc1_score * 2.0)
        
    def test_exact_phrase_proximity(self) -> None:
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


class TestSearchIntegration(unittest.TestCase):
    """Integration test suite verifying the pipeline from Raw HTML to Search Engine ranking."""
    
    def setUp(self) -> None:
        """Sets up a temporary filesystem and runs the Indexer to feed the SearchEngine."""

        # 1. Create raw mock crawled data (HTML)
        self.mock_crawled_data = {
            "https://example.com/hamlet": "<h1>Hamlet</h1><p>to be or not to be</p>",
            "https://example.com/descartes": "<h1>Descartes</h1><p>I think therefore I am</p>"
        }
        
        # Save to a temp file so the Indexer can load it
        self.temp_crawl_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        json.dump(self.mock_crawled_data, self.temp_crawl_file)
        self.temp_crawl_file.close()

        # 2. Build the Index using the Indexer class
        self.indexer = Indexer()
        self.indexer.build_index(self.temp_crawl_file.name)
        
        # 3. Hook up the SearchEngine to the Indexer's output
        self.searcher = SearchEngine(self.indexer.inverted_index, self.indexer.document_registry)

    def tearDown(self) -> None:
        """Cleans up the temporary files from the filesystem."""
        os.unlink(self.temp_crawl_file.name)

    def test_end_to_end_search(self) -> None:
        """Tests the entire pipeline: Raw HTML -> Tokenizer -> Indexer -> SearchEngine."""
        
        # Search for a quote from document 1
        results = self.searcher.find("to be or not to be")
        
        # It should find exactly 1 document
        self.assertEqual(len(results), 1)
        
        # The URL should match our first mock document
        self.assertEqual(results[0][1], "https://example.com/hamlet")
        
        # Search for a word in document 2, inside an h1 tag to implicitly test 
        # that the HTML structure survived the journey from raw text to search multiplier
        results_h1 = self.searcher.find("descartes")
        self.assertEqual(len(results_h1), 1)
        self.assertEqual(results_h1[0][1], "https://example.com/descartes")


if __name__ == '__main__':
    unittest.main()