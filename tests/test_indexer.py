import unittest
import json
import os
import tempfile

from src.indexer import HTMLProcessor, Indexer


class TestHTMLProcessor(unittest.TestCase):
    """Test suite for the HTMLProcessor, verifying NLP tokenization and extent parsing."""
    
    def setUp(self) -> None:
        """Initializes the HTMLProcessor instance before each test."""
        self.processor = HTMLProcessor()

    def test_tokenization_and_positions(self) -> None:
        """Tests that punctuation is stripped, words are stemmed, and positions are absolute."""
        html = "<p>Hello world, hello!</p>"
        result = self.processor.tokenize(html)
        
        # 'hello' should appear twice, at index 0 and 2.
        self.assertIn("hello", result)
        self.assertEqual(result["hello"][0], 2)          # Frequency
        self.assertEqual(result["hello"][1], [0, 2])     # Positions
        
        # 'world' should appear once, at index 1.
        self.assertIn("world", result)
        self.assertEqual(result["world"][0], 1)
        self.assertEqual(result["world"][1], [1])

    def test_extent_extraction_and_nested_merging(self) -> None:
        """
        Tests that nested HTML tags create perfect, merged interval extents.
        
        This explicitly verifies the logic designed to prevent extent fragmentation 
        when dealing with nested tags like <h1><b>Text</b></h1>.
        """

        # This is the exact edge case we engineered the merging logic for
        html = "<h1>The <b>quick brown</b> fox</h1>"
        result = self.processor.tokenize(html)
        
        # Words should be positioned 0, 1, 2, 3
        self.assertEqual(result["the"][1], [0])
        self.assertEqual(result["fox"][1], [3])
        
        # The <b> tag only wraps 'quick brown' (positions 1 to 2)
        self.assertIn("_EXTENT_b", result)
        self.assertEqual(result["_EXTENT_b"], [[1, 2]])
        
        # The <h1> tag wraps everything. Thanks to our merge logic, 
        # it should be one clean interval [0, 3], NOT fragmented!
        self.assertIn("_EXTENT_h1", result)
        self.assertEqual(result["_EXTENT_h1"], [[0, 3]])

    def test_quote_integrity_no_stopwords(self) -> None:
        """
        Tests that function words are not thrown out, preserving famous quotes.
        
        Since the domain is quotes.toscrape.com, traditional NLP stopwords (to, be, or) 
        must remain in the index to allow exact phrase proximity matching.
        """
        html = "<p>to be or not to be</p>"
        result = self.processor.tokenize(html)
        
        # All these traditional stop words should exist in the index
        for word in ["to", "be", "or", "not"]:
            self.assertIn(word, result)
            
        self.assertEqual(result["to"][1], [0, 4])


class TestIndexerIntegration(unittest.TestCase):
    """Integration test suite for the Indexer, verifying file I/O and global index construction."""

    def setUp(self) -> None:
        """
        Creates temporary mock crawled data files for safe I/O testing.
        
        Utilizes the tempfile library to ensure tests do not overwrite actual 
        project data during execution.
        """
        self.mock_crawled_data = {
            "https://example.com/1": "<h1>Data structures</h1><p>are fun</p>",
            "https://example.com/2": "<title>Data</title>"
        }
        
        # Create temporary files for our tests to read/write to safely
        self.temp_crawl_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
        json.dump(self.mock_crawled_data, self.temp_crawl_file)
        self.temp_crawl_file.close()
        
        self.temp_index_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_index_file.close()
        
        self.temp_registry_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_registry_file.close()

    def tearDown(self) -> None:
        """Cleans up the temporary files from the filesystem after tests complete."""
        os.unlink(self.temp_crawl_file.name)
        os.unlink(self.temp_index_file.name)
        os.unlink(self.temp_registry_file.name)

    def test_build_index(self) -> None:
        """Tests that the indexer correctly maps URLs to DocIDs and merges terms globally."""
        indexer = Indexer()
        indexer.build_index(self.temp_crawl_file.name)
        
        # Verify Registry (DocIDs 1 and 2 assigned)
        self.assertEqual(len(indexer.document_registry), 2)
        self.assertEqual(indexer.document_registry[1], "https://example.com/1")
        
        # Verify Global Inverted Index
        # 'data' appears in both Doc 1 (pos 0) and Doc 2 (pos 0)
        self.assertIn("data", indexer.inverted_index)
        self.assertIn(1, indexer.inverted_index["data"])
        self.assertIn(2, indexer.inverted_index["data"])
        
        # Check that doc 1 recorded 'data' at position 0
        self.assertEqual(indexer.inverted_index["data"][1][1], [0])

    def test_save_and_load_state(self) -> None:
        """Tests that the minified JSON index can be serialized to disk and perfectly restored."""
        
        # 1. Build and Save
        indexer_a = Indexer()
        indexer_a.build_index(self.temp_crawl_file.name)
        indexer_a.save_index(self.temp_index_file.name, self.temp_registry_file.name)
        
        # 2. Create a brand new instance and Load
        indexer_b = Indexer()
        indexer_b.load_index(self.temp_index_file.name, self.temp_registry_file.name)
        
        # 3. Verify the state transferred perfectly
        self.assertEqual(indexer_a.inverted_index, indexer_b.inverted_index)
        self.assertEqual(indexer_a.document_registry, indexer_b.document_registry)
        
        # Verify the next_doc_id updated correctly so we don't overwrite
        self.assertEqual(indexer_b.next_doc_id, 3)


if __name__ == '__main__':
    unittest.main()