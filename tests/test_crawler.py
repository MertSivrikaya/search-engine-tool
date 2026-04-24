import unittest
import requests
from unittest.mock import patch, MagicMock
from src.crawler import Crawler

class TestCrawler(unittest.TestCase):

    def setUp(self):
        """Initialize a crawler instance with the network initialization mocked out."""
        
        # By patching _load_robots_txt, we prevent the crawler from hitting the 
        # real internet during initialization. The parser starts completely clean.
        with patch('src.crawler.Crawler._load_robots_txt'):
            self.crawler = Crawler(base_url="https://example.com", min_delay=0)

    def test_extract_text(self):
        """Verify that HTML tags, scripts, and styles are correctly stripped."""
        
        html_content = """
        <html>
            <style>.header { color: red; }</style>
            <script>console.log('hello');</script>
            <body>
                <h1>Title</h1>
                <p>This is a <b>test</b> quote.</p>
            </body>
        </html>
        """
        expected_text = "Title This is a test quote."
        
        result = self.crawler.extract_text(html_content)
        self.assertEqual(result, expected_text)

    @patch('src.crawler.requests.Session.get')
    def test_fetch_page_success(self, mock_get):
        """Test fetch_page returns HTML on a 200 OK response."""
        
        # Create a mock (simulated) response object to simulate a successful HTTP request
        # We only need to set the attributes that our crawler's fetch_page method uses: status_code and text.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html>Success</html>"

        # Make the mocked (simulated) get() method return our mock_response when called
        mock_get.return_value = mock_response

        # Whenever get() is called within fetch_page, it will call our mock_get, which will return mock_response,
        # instead of making a real HTTP request. This allows us to test the logic of fetch_page without relying on network access.
        result = self.crawler.fetch_page("https://example.com/page1")
        self.assertEqual(result, "<html>Success</html>")

    @patch('src.crawler.requests.Session.get')
    def test_fetch_page_failure(self, mock_get):
        """Test fetch_page returns None on a 404 error."""
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("404 Client Error")
        
        mock_get.return_value = mock_response

        result = self.crawler.fetch_page("https://example.com/404")
        self.assertIsNone(result)

    def test_robots_txt_logic(self):
        """Verify the robotparser correctly handles Disallow rules."""
        
        # Since setUp mocked out the real internet, the parser is empty.
        # We can safely inject our own rules to test the logic.
        self.crawler.rp.parse([
            "User-agent: *",
            "Disallow: /private/",
            "Allow: /public/"
        ])
        
        self.assertTrue(self.crawler.rp.can_fetch(self.crawler.user_agent_name, "https://example.com/public/page"))
        self.assertFalse(self.crawler.rp.can_fetch(self.crawler.user_agent_name, "https://example.com/private/secret"))

if __name__ == '__main__':
    unittest.main()