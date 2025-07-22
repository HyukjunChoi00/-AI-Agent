import asyncio
import re
from typing import List, Dict
from urllib.parse import urljoin
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, Comment
import nest_asyncio
# Jupyter notebook에서 asyncio 사용을 위한 설정
nest_asyncio.apply()
class tariff_search():
    def __init__(self, max_articles=8):
        self.max_articles = max_articles
        self.base_url = "https://www.econotimes.com"

    async def setup_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()

    async def close_browser(self):
        await self.browser.close()
        await self.playwright.stop()

    def clean_title(self, title: str) -> str:
        title = re.sub(r'\s+', ' ', title.strip())
        title = re.sub(r'^\W+|\W+$', '', title)
        return title if len(title) > 10 else ""

    # 제목과 URL 추출

    async def extract_articles(self, page) -> List[Dict[str, str]]:
        articles = []
        for i in range(1, self.max_articles + 1):
            try:
                xpath = f'//*[@id="archivePage"]/div/div[2]/div[{i}]/p[1]/a'
                link = await page.query_selector(f'xpath={xpath}')
                if not link:
                    continue
                title = await link.inner_text()
                url = await link.get_attribute('href')
                url = urljoin(self.base_url, url) if url and not url.startswith("http") else url
                title = self.clean_title(title)
                if title and url:
                    articles.append({"title": title, "url": url})
            except:
                continue
        return articles

    async def fetch_html(self, url: str) -> str:
        page = await self.context.new_page()
        await page.goto(url, wait_until='domcontentloaded', timeout=20000)
        await asyncio.sleep(3)
        html = await page.inner_html("body")
        await page.close()
        return html

    def clean_html(self, raw_html: str) -> str:
        soup = BeautifulSoup(raw_html, 'html.parser')
        for tag in soup.find_all(True):
            tag.attrs = {}
        for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
            tag.decompose()
        for comment in soup.find_all(text=lambda t: isinstance(t, Comment)):
            comment.extract()
        return str(soup)

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        article = soup.find('article')
        if not article:
            return "본문을 찾을 수 없습니다."
        for tag in article(['ins', 'iframe', 'script']):
            tag.decompose()
        paragraphs = [p.get_text(strip=True) for p in article.find_all('p')]
        return '\n\n'.join(paragraphs)

    async def run_pipeline(self) -> List[Dict[str, str]]:
        await self.setup_browser()
        query = "tariff"
        search_url = f"https://www.econotimes.com/search?v={query}"
        page = await self.context.new_page()
        await page.goto(search_url, wait_until='domcontentloaded', timeout=20000)
        await asyncio.sleep(2)
        articles = await self.extract_articles(page)
        await page.close()

        results = []
        for article in articles:
            html = await self.fetch_html(article["url"])
            cleaned_html = self.clean_html(html)
            text = self.extract_text(cleaned_html)
            results.append({
                "title": article["title"],
                "url": article["url"],
                "content": text
            })

        await self.close_browser()
        return results

class interestrate_search():
    def __init__(self, max_articles=8):
        self.max_articles = max_articles
        self.base_url = "https://www.econotimes.com"

    async def setup_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()

    async def close_browser(self):
        await self.browser.close()
        await self.playwright.stop()

    def clean_title(self, title: str) -> str:
        title = re.sub(r'\s+', ' ', title.strip())
        title = re.sub(r'^\W+|\W+$', '', title)
        return title if len(title) > 10 else ""

    # 제목과 URL 추출

    async def extract_articles(self, page) -> List[Dict[str, str]]:
        articles = []
        for i in range(1, self.max_articles + 1):
            try:
                xpath = f'//*[@id="archivePage"]/div/div[2]/div[{i}]/p[1]/a'
                link = await page.query_selector(f'xpath={xpath}')
                if not link:
                    continue
                title = await link.inner_text()
                url = await link.get_attribute('href')
                url = urljoin(self.base_url, url) if url and not url.startswith("http") else url
                title = self.clean_title(title)
                if title and url:
                    articles.append({"title": title, "url": url})
            except:
                continue
        return articles

    async def fetch_html(self, url: str) -> str:
        page = await self.context.new_page()
        await page.goto(url, wait_until='domcontentloaded', timeout=20000)
        await asyncio.sleep(3)
        html = await page.inner_html("body")
        await page.close()
        return html

    def clean_html(self, raw_html: str) -> str:
        soup = BeautifulSoup(raw_html, 'html.parser')
        for tag in soup.find_all(True):
            tag.attrs = {}
        for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
            tag.decompose()
        for comment in soup.find_all(text=lambda t: isinstance(t, Comment)):
            comment.extract()
        return str(soup)

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        article = soup.find('article')
        if not article:
            return "본문을 찾을 수 없습니다."
        for tag in article(['ins', 'iframe', 'script']):
            tag.decompose()
        paragraphs = [p.get_text(strip=True) for p in article.find_all('p')]
        return '\n\n'.join(paragraphs)

    async def run_pipeline(self) -> List[Dict[str, str]]:
        await self.setup_browser()
        query = "interest rate"
        search_url = f"https://www.econotimes.com/search?v={query}"
        page = await self.context.new_page()
        await page.goto(search_url, wait_until='domcontentloaded', timeout=20000)
        await asyncio.sleep(2)
        articles = await self.extract_articles(page)
        await page.close()

        results = []
        for article in articles:
            html = await self.fetch_html(article["url"])
            cleaned_html = self.clean_html(html)
            text = self.extract_text(cleaned_html)
            results.append({
                "title": article["title"],
                "url": article["url"],
                "content": text
            })

        await self.close_browser()
        return results