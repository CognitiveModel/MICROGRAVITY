import asyncio
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

class WebCrawler:
    def __init__(self, max_depth=2, max_pages=10):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited = set()
        self.results = []

    def is_valid_url(self, url, base_domain):
        """Checks if the URL is valid and belongs to the same domain."""
        parsed = urlparse(url)
        return bool(parsed.netloc) and parsed.netloc == base_domain

    def crawl(self, start_url):
        """Starts crawling from a starting URL."""
        self.visited = set()
        self.results = []
        base_domain = urlparse(start_url).netloc
        self._crawl_recursive(start_url, base_domain, depth=0)
        return self.results

    def _crawl_recursive(self, url, base_domain, depth):
        if depth > self.max_depth or len(self.visited) >= self.max_pages or url in self.visited:
            return

        print(f"Crawling: {url} (Depth: {depth})")
        self.visited.add(url)
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text
            text = soup.get_text(separator=' ').strip()
            self.results.append({"url": url, "content": text[:2000]}) # Cap content
            
            # Find links
            if depth < self.max_depth:
                for a in soup.find_all('a', href=True):
                    next_url = urljoin(url, a['href'])
                    if self.is_valid_url(next_url, base_domain):
                        self._crawl_recursive(next_url, base_domain, depth + 1)
        except Exception as e:
            print(f"Error crawling {url}: {e}")

if __name__ == "__main__":
    crawler = WebCrawler(max_depth=1, max_pages=3)
    print("Testing crawler on 'https://pypi.org/project/duckduckgo-search/'...")
    data = crawler.crawl("https://pypi.org/project/duckduckgo-search/")
    print(f"Crawled {len(data)} pages.")
