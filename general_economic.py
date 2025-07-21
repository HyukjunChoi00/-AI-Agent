import asyncio
import re
from typing import List, Dict
from urllib.parse import urljoin
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup, Comment
import nest_asyncio
# Jupyter notebook에서 asyncio 사용을 위한 설정
nest_asyncio.apply()
class overall_economic():
    def __init__(self, max_articles=5):
        self.max_articles = max_articles
        self.base_url = "https://www.naver.com"

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
    # class 이름은 실제 HTML 구조에 따라 조정 가능
        elements = await page.query_selector_all("a.BAeT2rSB_v3C8l_Lu2U6")  # 또는 "a[class*='BAeT2rSB']"
        for i, element in enumerate(elements[:self.max_articles]):
            try:
                title_span = await element.query_selector("span")
                title = await title_span.inner_text() if title_span else await element.inner_text()
                url = await element.get_attribute('href')
                title = self.clean_title(title)
                if title and url:
                    articles.append({"title": title, "url": url})
            except Exception as e:
                print(f"[ERROR] 기사 {i} 파싱 실패: {e}")
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
        query = "금리"
        search_url = f"https://search.naver.com/search.naver?ssc=tab.news.all&where=news&sm=tab_jum&query={query}"
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
