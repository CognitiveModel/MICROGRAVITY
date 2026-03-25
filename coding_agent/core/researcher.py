import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebResearcher:
    def __init__(self):
        self.ddgs = DDGS()

    def search(self, query, max_results=5):
        """Searches the web using DuckDuckGo."""
        logger.info(f"Searching for: {query}")
        results = []
        try:
            for r in self.ddgs.text(query, max_results=max_results):
                results.append(r)
        except Exception as e:
            logger.error(f"Search error: {e}")
        return results

    def scrape_url(self, url):
        """Scrapes text content from a URL."""
        logger.info(f"Scraping URL: {url}")
        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove scripts and styles
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
                
            # Get text and clean up
            text = soup.get_text(separator=' ')
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:5000] # Cap at 5000 characters for now
        except Exception as e:
            logger.error(f"Scrape error for {url}: {e}")
            return f"Error scraping {url}: {str(e)}"

if __name__ == "__main__":
    # Test
    researcher = WebResearcher()
    print("Testing search for 'Python programming'...")
    results = researcher.search("Python programming", max_results=2)
    for i, r in enumerate(results):
        print(f"Result {i+1}: {r['title']} - {r['href']}")
        if i == 0:
            print(f"Scraping first result...")
            content = researcher.scrape_url(r['href'])
            print(f"Scraped content (first 200 chars): {content[:200]}...")
