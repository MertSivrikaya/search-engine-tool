import time
import requests
from urllib.parse import urljoin
from urllib import robotparser
from bs4 import BeautifulSoup
from collections import deque
from typing import Dict, Optional
from random import uniform
import json

class Crawler:
    """
    A robust web crawler that navigates a website, extracts text, 
    respects robots.txt, and strictly enforces politeness policies.
    """
    
    def __init__(self, base_url: str = "https://quotes.toscrape.com/", min_delay: int = 6):
        """
        Initializes the crawler with connection pooling, User-Agent header, and robots.txt parsing.
        
        Args:
            base_url (str): The starting URL to crawl.
            min_delay (int): Minimum seconds to wait between requests.
        """
        print(f"[*] Initializing...")

        self.base_url = base_url
        self.min_delay = min_delay
        self.session = requests.Session()

        # Adding a User-Agent header is a good defensive programming practice
        # to ensure the target server doesn't reject our requests outright.
        # We'll use a clear, honest User-Agent string that identifies our crawler and provides contact info.
        self.user_agent_name = "COMP3011-Crawler/1.0 (Student Project; +https://github.com/MertSivrikaya/search-engine-tool)"
        self.session.headers.update({
            "User-Agent": self.user_agent_name
        })

        # Initialize and parse robots.txt
        self.rp = robotparser.RobotFileParser()
        self._load_robots_txt()

    def _load_robots_txt(self):
        """
        Fetches and parses the robots.txt file using our custom session.
        """
        robots_url = urljoin(self.base_url, "/robots.txt")
        print(f"[*] Fetching {robots_url}...")
        
        try:
            response = self.session.get(robots_url, timeout=10)

            response.raise_for_status()
            
            # Parse the text content into the robotparser
            self.rp.parse(response.text.splitlines())
            print("    -> robots.txt parsed successfully.")
        except requests.exceptions.HTTPError as e:
            # If the server actively denies us permission to even read the rules, 
            # ethical crawling dictates we should back off entirely.
            if e.response.status_code in (401, 403):
                print("    -> [!] Access denied to robots.txt (401/403). Assuming Disallow All.")
                self.rp.parse(["User-agent: *", "Disallow: /"])
            # If the file simply doesn't exist (404), explicitly tell the parser to allow everything.
            elif e.response.status_code == 404:
                print("    -> [!] Warning: robots.txt not found (404). Explicitly allowing all paths.")
                self.rp.parse(["User-agent: *", "Allow: /"])
            else:
                print(f"    -> [!] HTTP Error {e.response.status_code}. Explicitly allowing all paths.")
                self.rp.parse(["User-agent: *", "Allow: /"])
                
        except requests.exceptions.RequestException as e:
            print(f"    -> [!] Network error fetching robots.txt: {e}. Explicitly allowing all paths.")
            self.rp.parse(["User-agent: *", "Allow: /"])

    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetches the HTML content of a given URL with with randomized politeness delay.
        """
        try:
            print(f"[*] Fetching: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status() 
            
            # Enforce the strict politeness window
            # Randomized delay: minimum delay + (0 to 2) extra seconds
            # This makes the traffic pattern less predictable and more human
            # to avoid triggering anti-bot measures (like firewalls) while still respecting the server's resources.
            wait_time = self.min_delay + uniform(0, 2)  
            print(f"    -> Sleeping for {wait_time:.2f} seconds (Politeness Window)...")
            time.sleep(wait_time) 
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            print(f"[!] Error fetching {url}: {e}")
            return None

    def extract_text(self, html: str) -> str:
        """
        Parses HTML and extracts clean, readable text, stripping out code/styling.
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements as they don't contain indexable natural language
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
            
        # Extract the visible text, separated by spaces, and strip leading/trailing whitespace
        return soup.get_text(separator=' ', strip=True)

    def crawl(self) -> Dict[str, str]:
        """
        Crawls the website using a Queue (Frontier) and Visited Set to discover
        all internal links while preventing infinite loops.
        Checks robots.txt permissions before adding URLs to the frontier.
        
        Returns:
            Dict[str, str]: A dictionary mapping URLs to their extracted text.
        """
        crawled_data = {}

        # Ensure we are allowed to crawl the starting URL before proceeding
        if not self.rp.can_fetch(self.user_agent_name, self.base_url):
            print(f"[!] Crawl aborted: robots.txt forbids accessing the base URL ({self.base_url}).")
            return crawled_data

        frontier = deque([self.base_url])
        # URLs that are either already crawled or waiting in the frontier to be crawled. 
        seen_urls = set([self.base_url])
        
        while frontier:
            current_url = frontier.popleft()
                
            html = self.fetch_page(current_url)
            if not html:
                continue # Network error, skip to the next URL in the queue
                
            seen_urls.add(current_url)
            
            # Clean the HTML and store the text
            text = self.extract_text(html)
            crawled_data[current_url] = text
            
            # --- Link Discovery Logic ---
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find every link on the page
            for a_tag in soup.find_all('a', href=True):
                next_url = urljoin(current_url, a_tag['href'])
                next_url = next_url.split('#')[0] # Clean anchor fragments

                # Check if the next URL is within the same domain (we dont' want to crawl external links) and hasn't been visited
                if (self.base_url in next_url) and (next_url not in seen_urls):
                    
                    # Check if robots.txt permit us to crawl this URL
                    if self.rp.can_fetch(self.user_agent_name, next_url):
                        frontier.append(next_url)
                    else:
                        print(f"    -> [BLOCKED] robots.txt forbids crawling: {next_url}")
                        
                    seen_urls.add(next_url)
                    
        print(f"\n[+] Crawling complete. Successfully processed {len(crawled_data)} unique pages.")
        return crawled_data

# Quick testing block
if __name__ == "__main__":
    print("Starting crawler test...")
    crawler = Crawler()
    # Let's test it! It will take about a minute since it sleeps for 6 seconds per page.
    data = crawler.crawl()

    # Save to secondary storage ---
    if data:
        output_file = "crawled_data.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"[+] Data successfully saved to {output_file}")