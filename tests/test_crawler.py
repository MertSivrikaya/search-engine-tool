import unittest
from unittest.mock import patch, MagicMock

import requests

from src.crawler import Crawler


class TestCrawler(unittest.TestCase):
    """Test suite for the web crawler, covering unit, integration, and performance testing."""

    def setUp(self) -> None:
        """Initialize a crawler instance with the network initialization mocked out."""
        
        # By patching _load_robots_txt, we prevent the crawler from hitting the 
        # real internet during initialization. The parser starts completely clean.
        with patch('src.crawler.Crawler._load_robots_txt'):
            self.crawler = Crawler(base_url="https://example.com/", min_delay=0)

    @patch('src.crawler.requests.Session.get')
    def test_fetch_page_success(self, mock_get) -> None:
        """
        Tests that fetch_page successfully returns HTML on a 200 OK HTTP response.
        
        Args:
            mock_get (MagicMock): The mocked requests.Session.get method.
        """
        
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
    def test_fetch_page_failure(self, mock_get) -> None:
        """
        Tests that fetch_page catches RequestExceptions and returns None on a 404 error.
        
        Args:
            mock_get (MagicMock): The mocked requests.Session.get method.
        """
        
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("404 Client Error")
        
        mock_get.return_value = mock_response

        result = self.crawler.fetch_page("https://example.com/404")
        self.assertIsNone(result)

    def test_robots_txt_logic(self) -> None:
        """Verifies the robotparser correctly enforces Disallow/Allow rules."""
        
        # Since setUp mocked out the real internet, the parser is empty.
        # We can safely inject our own rules to test the logic.
        self.crawler.rp.parse([
            "User-agent: *",
            "Disallow: /private/",
            "Allow: /public/"
        ])
        
        self.assertTrue(self.crawler.rp.can_fetch(self.crawler.user_agent_name, "https://example.com/public/page"))
        self.assertFalse(self.crawler.rp.can_fetch(self.crawler.user_agent_name, "https://example.com/private/secret"))

    @patch('src.crawler.Crawler.fetch_page')
    def test_crawl_loop_prevention(self, mock_fetch) -> None:
        """
        Integration test verifying graph navigation and infinite loop avoidance.
        
        Simulates a cyclic graph (A -> B -> A) to ensure the 'visited' set prevents 
        redundant processing.
        
        Args:
            mock_fetch (MagicMock): The mocked fetch_page method.
        """

        # 1. Build a mock web dictionary
        mock_web = {
            "https://example.com/": """
                <html>
                    <body>
                        <h1>Home</h1>
                        <a href='/page2'>Go to Page 2</a>
                        <a href='/'>Loop back to Home</a>
                    </body>
                </html>
            """,
            "https://example.com/page2": """
                <html>
                    <body>
                        <h1>Page 2</h1>
                        <p>End of the line.</p>
                    </body>
                </html>
            """
        }

        # 2. Configure the mock to return the correct HTML for each URL
        # If the URL isn't in our dictionary, return None (simulating a 404)
        mock_fetch.side_effect = lambda url: mock_web.get(url, None)

        # 3. Explicitly allow everything in robots.txt for this test
        self.crawler.rp.can_fetch = MagicMock(return_value=True)

        # 4. Run the crawl
        result = self.crawler.crawl()

        # 5. Assertions
        # It should have found exactly 2 unique pages (ignoring the infinite loop link)
        self.assertEqual(len(result), 2)
        
        # It should have extracted the HTML correctly
        self.assertIn("<h1>Home</h1>", result["https://example.com/"])
        self.assertIn("<h1>Page 2</h1>", result["https://example.com/page2"])

    @patch('src.crawler.Crawler.fetch_page')
    def test_crawl_dead_end(self, mock_fetch) -> None:
        """
        Integration test verifying crawler termination on orphan pages (no out-links).
        
        Args:
            mock_fetch (MagicMock): The mocked fetch_page method.
        """
        mock_web = {
            "https://example.com/": """
                <html>
                    <body>
                        <h1>Dead End</h1>
                        <p>There are no links on this page.</p>
                    </body>
                </html>
            """
        }
        mock_fetch.side_effect = lambda url: mock_web.get(url, None)
        self.crawler.rp.can_fetch = MagicMock(return_value=True)

        result = self.crawler.crawl()

        # It should successfully crawl the base URL and then cleanly terminate
        self.assertEqual(len(result), 1)
        
        self.assertIn("<h1>Dead End</h1>", result["https://example.com/"])

    @patch('src.crawler.Crawler.fetch_page')
    def test_crawl_external_link_trap(self, mock_fetch) -> None:
        """
        Integration test verifying strict domain boundary enforcement.
        
        Ensures links pointing outside the base_url domain are not added to the frontier.
        
        Args:
            mock_fetch (MagicMock): The mocked fetch_page method.
        """
        mock_web = {
            "https://example.com/": """
                <html>
                    <body>
                        <h1>Home</h1>
                        <a href='https://youtube.com/video'>External Link</a>
                        <a href='/internal'>Internal Link</a>
                    </body>
                </html>
            """,
            "https://example.com/internal": """
                <html>
                    <body>
                        <h1>Safe</h1>
                    </body>
                </html>
            """
        }
        mock_fetch.side_effect = lambda url: mock_web.get(url, None)
        self.crawler.rp.can_fetch = MagicMock(return_value=True)

        result = self.crawler.crawl()

        # It should find the Home page and the Internal page (2 pages), completely ignoring YouTube
        self.assertEqual(len(result), 2)
        self.assertNotIn("https://youtube.com/video", result)

    @patch('src.crawler.Crawler.fetch_page')
    def test_crawl_blocked_path(self, mock_fetch) -> None:
        """
        Integration test verifying dynamic link discovery against robots.txt.
        
        Args:
            mock_fetch (MagicMock): The mocked fetch_page method.
        """
        mock_web = {
            "https://example.com/": """
                <html>
                    <body>
                        <h1>Home</h1>
                        <a href='/public'>Public Page</a>
                        <a href='/admin'>Secret Admin Panel</a>
                    </body>
                </html>
            """,
            "https://example.com/public": """
            <html>
                <body>
                    <h1>Public</h1>
                </body>
            </html>""",
            "https://example.com/admin": """
            <html>
                <body>
                    <h1>Admin Area</h1>
                </body>
            </html>""" # Should never be reached
        }
        mock_fetch.side_effect = lambda url: mock_web.get(url, None)

        # Inject a strict robots.txt rule blocking the /admin path
        self.crawler.rp.parse([
            "User-agent: *",
            "Disallow: /admin"
        ])

        result = self.crawler.crawl()

        # It should crawl Home and Public (2 pages), but NOT the Admin page
        self.assertEqual(len(result), 2)
        self.assertIn("https://example.com/public", result)
        self.assertNotIn("https://example.com/admin", result)


if __name__ == '__main__':
    unittest.main()