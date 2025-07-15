import requests
from bs4 import BeautifulSoup
import os
import logging
from urllib.parse import urljoin, urlparse
import mimetypes
import time
# --- Selenium imports ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# ---

# Note: Original script used Selenium. This version uses requests for simplicity.
# If Selenium is strictly needed, this class would need significant rework.

logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, download_dir):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        logger.info(f"WebScraper initialized. Download directory: {self.download_dir}")

    def _download_image(self, image_url, image_filename):
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            filepath = os.path.join(self.download_dir, image_filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info(f"Downloaded image: {image_url} to {filepath}")
            return filepath, image_url # Return saved path and original URL
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image {image_url}: {e}")
            return None, image_url
        except Exception as e:
            logger.error(f"An unexpected error occurred while downloading image {image_url}: {e}")
            return None, image_url

    def fetch_text_and_images(self, url):
        """
        Fetches HTML using Selenium, extracts text, and downloads images. 
        Returns:
            tuple: (text_content_path, image_local_paths, image_original_urls)
                   Returns (None, [], []) on failure.
        """
        # headers = { ... } # Not needed for Selenium
        try:
            logger.info(f"Fetching URL with Selenium: {url}")
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            time.sleep(3)  # Wait for dynamic content to load
            html_content = driver.page_source
            driver.quit()
            soup = BeautifulSoup(html_content, 'html.parser')

            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            page_text = soup.get_text(separator='\n', strip=True)
            
            text_filename = "product_page_text.txt"
            text_content_path = os.path.join(self.download_dir, text_filename)
            with open(text_content_path, "w", encoding="utf-8") as f:
                f.write(page_text)
            logger.info(f"Saved page text for {url} to {text_content_path}")

            found_image_sources = []
            for img_tag in soup.find_all('img'):
                src = img_tag.get('src')
                if src:
                    full_url = urljoin(url, src)
                    if full_url.startswith('http'):
                        found_image_sources.append(full_url)
            unique_image_urls = sorted(list(set(found_image_sources)), key=found_image_sources.index)
            logger.info(f"Found {len(unique_image_urls)} unique image URLs on {url}")

            downloaded_image_paths = []
            saved_image_original_urls = []

            for i, img_url_to_download in enumerate(unique_image_urls):
                try:
                    parsed_image_url = urlparse(img_url_to_download)
                    original_filename, ext = os.path.splitext(os.path.basename(parsed_image_url.path))
                    if not ext:
                        try:
                            head_resp = requests.head(img_url_to_download, timeout=5)
                            content_type = head_resp.headers.get('content-type')
                            if content_type:
                                guessed_ext = mimetypes.guess_extension(content_type)
                                if guessed_ext: ext = guessed_ext
                        except: pass
                    if not ext or len(ext) > 5: ext = ".jpg"
                    image_filename = f"image_{i:03d}{ext}"
                    time.sleep(0.1)
                    saved_path, original_dl_url = self._download_image(img_url_to_download, image_filename)
                    if saved_path:
                        downloaded_image_paths.append(saved_path)
                        saved_image_original_urls.append(original_dl_url)
                except Exception as e:
                    logger.error(f"Error processing image URL {img_url_to_download}: {e}")
                    continue
            logger.info(f"Scraping {url} complete. Text saved, {len(downloaded_image_paths)} images downloaded.")
            return text_content_path, downloaded_image_paths, saved_image_original_urls

        except Exception as e:
            logger.error(f"An unexpected error occurred during scraping {url}: {e}", exc_info=True)
            return None, [], []

        # --- Previous requests-based code for reference ---
        # try:
        #     logger.info(f"Fetching URL with requests: {url}")
        #     response = requests.get(url, headers=headers, timeout=30)
        #     response.raise_for_status()
        #     html_content = response.content
        #     soup = BeautifulSoup(html_content, 'html.parser')
        #     ...
        # except ...
        #     ...

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        scraper = WebScraper(download_dir=tmpdir)
        test_url = "https://www.ikea.com/kr/ko/p/malm-ottoman-bed-white-20404807/"
        
        text_path, img_paths, img_urls = scraper.fetch_text_and_images(test_url)
        if text_path:
            logger.info(f"Text saved to: {text_path}")
            logger.info(f"Downloaded {len(img_paths)} images. URLs: {img_urls}")
            # Check content of text_path
            with open(text_path, 'r', encoding='utf-8') as f:
                logger.info(f"First 200 chars of text: {f.read(200)}...")
        else:
            logger.error(f"Failed to scrape {test_url}") 