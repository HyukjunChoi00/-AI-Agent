import asyncio
import json
import re
from playwright.async_api import async_playwright
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Optional
import nest_asyncio
import requests
from bs4 import BeautifulSoup

# Jupyter notebook에서 asyncio 사용을 위한 설정
nest_asyncio.apply()

class EconoTimesNewsScraper:
    def __init__(self, max_articles=20):
        self.max_articles = max_articles
        self.base_url = "https://www.econotimes.com"
        self.articles = []

    async def setup_browser(self):
        """브라우저 설정"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

    async def close_browser(self):
        """브라우저 종료"""
        await self.browser.close()
        await self.playwright.stop()

    def is_valid_article_url(self, url: str) -> bool:
        """유효한 기사 URL인지 확인 (더 관대한 검증)"""
        if not url or url.startswith('javascript:') or url.startswith('mailto:'):
            return False
        
        # 상대 경로 처리
        if url.startswith('/'):
            url = urljoin(self.base_url, url)
        
        parsed = urlparse(url)
        if not parsed.netloc:
            return False
            
        # EconoTimes 관련 URL 확인
        if 'econotimes.com' in url:
            # 제외할 패턴들
            exclude_patterns = [
                '/search', '/category', '/tag', '/author', '/about', '/contact',
                '/privacy', '/terms', '/sitemap', '/rss', '/feed', '/api',
                'javascript:', 'mailto:', '#', '?', '/page/', '/archive/'
            ]
            
            # 제외 패턴 확인
            for pattern in exclude_patterns:
                if pattern in url.lower():
                    return False
            
            # 기사 URL 패턴 확인 (더 관대하게)
            # 상대 경로로 시작하는 기사들 (예: /Asian-Stocks-Edge-Higher...)
            if re.match(r'^/[A-Za-z0-9\-]+', url) or '/news/' in url or '/article/' in url:
                return True
            
            # 숫자가 포함된 URL (보통 기사 ID)
            if re.search(r'\d+', url):
                return True
        
        return False

    def clean_title(self, title: str) -> str:
        """제목 정리"""
        if not title:
            return ""
        
        # 불필요한 공백 제거
        title = re.sub(r'\s+', ' ', title.strip())
        
        # 특수 문자나 불필요한 텍스트 제거
        title = re.sub(r'^\W+|\W+$', '', title)
        
        # 너무 짧거나 의미없는 제목 필터링
        if len(title) < 10 or title.lower() in ['more', 'read more', 'click here', 'link']:
            return ""
        
        return title

    async def extract_articles_from_page(self, page, query: str) -> List[Dict[str, str]]:
        """페이지에서 기사 정보 추출 (XPath 반복 방식)"""
        articles = []
        try:
            await page.wait_for_load_state('domcontentloaded', timeout=20000)
            await asyncio.sleep(3)

            for i in range(1, self.max_articles + 1):
                xpath = f'//*[@id="archivePage"]/div/div[2]/div[{i}]/p[1]/a'
                try:
                    link = await page.query_selector(f'xpath={xpath}')
                    if not link:
                        continue

                    title = await link.inner_text()
                    url = await link.get_attribute('href')
                    if url and not url.startswith('http'):
                        url = urljoin(self.base_url, url)
                    title = self.clean_title(title)
                    if title and url:
                        articles.append({
                            'title': title,
                            'url': url,
                            'query': query
                        })
                        print(f"추가된 기사: {title}")
                except Exception as e:
                    print(f"{i}번째 기사 처리 중 오류: {e}")
                    continue

        except Exception as e:
            print(f"기사 추출 중 오류: {e}")

        return articles


    async def search_news(self, query: str) -> List[Dict[str, str]]:
        """뉴스 검색 및 기사 정보 추출"""
        # 여러 검색 URL 시도
        search_urls = [f"https://www.econotimes.com/search?v={query}"]
        
        try:
            await self.setup_browser()
            
            for search_url in search_urls:
                try:
                    print(f"\n검색 URL 시도: {search_url}")
                    page = await self.context.new_page()
                    
                    # 페이지 로딩 시도
                    await page.goto(search_url, wait_until='domcontentloaded', timeout=20000)
                    
                    # 메인 페이지인 경우 검색 실행
                    if search_url == "https://www.econotimes.com":
                        await self.perform_search_on_page(page, query)
                    
                    # 기사 추출
                    articles = await self.extract_articles_from_page(page, query)
                    
                    await page.close()
                    
                    if articles:
                        print(f"총 {len(articles)}개의 기사를 찾았습니다.")
                        return articles
                    else:
                        print("기사를 찾지 못했습니다.")
                        
                except Exception as e:
                    print(f"URL {search_url} 처리 중 오류: {e}")
                    continue
            
            print("모든 검색 방법이 실패했습니다.")
            return []
            
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return []
            
        finally:
            await self.close_browser()

    async def perform_search_on_page(self, page, query: str):
        """페이지에서 검색 실행"""
        try:
            await asyncio.sleep(2)
            
            # 검색창 찾기
            search_selectors = [
                'input[type="search"]',
                'input[name="q"]',
                'input[name="search"]',
                'input[name="s"]',
                'input[placeholder*="search"]',
                'input[placeholder*="Search"]',
                '#search',
                '.search-input',
                '.search-box input',
                'form input[type="text"]'
            ]
            
            search_input = None
            for selector in search_selectors:
                try:
                    search_input = await page.query_selector(selector)
                    if search_input:
                        print(f"검색창 발견: {selector}")
                        break
                except:
                    continue
            
            if search_input:
                await search_input.fill(query)
                await search_input.press('Enter')
                await asyncio.sleep(5)  # 검색 결과 로딩 대기
                print(f"검색 실행 완료: {query}")
            else:
                print("검색창을 찾지 못했습니다.")
                
        except Exception as e:
            print(f"검색 실행 중 오류: {e}")

    async def get_news_by_query(self, query: str, max_articles: int = 20) -> List[Dict[str, str]]:
        """LangChain tool용 메인 함수"""
        self.max_articles = max_articles
        return await self.search_news(query)



# 동기 래퍼 함수들
def search_econotimes_news(query: str, max_articles: int = 20, use_requests: bool = False) -> List[Dict[str, str]]:
    """
    EconoTimes에서 뉴스 기사를 검색하는 함수
    
    Args:
        query (str): 검색할 키워드
        max_articles (int): 최대 기사 수 (기본값: 20)
        use_requests (bool): requests 방식 사용 여부 (기본값: False)
    
    Returns:
        List[Dict[str, str]]: 기사 정보 리스트 (title, url, query 포함)
    """
    scraper = EconoTimesNewsScraper(max_articles=max_articles)
    
    if use_requests:
        return scraper.scrape_with_requests(query, max_articles)
    
    # 비동기 방식
    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(scraper.get_news_by_query(query, max_articles))
    except RuntimeError:
        return asyncio.run(scraper.get_news_by_query(query, max_articles))

def quick_search(query: str, max_articles: int = 10, use_requests: bool = False):
    """빠른 검색 함수"""
    print(f"'{query}' 검색 중...")
    results = search_econotimes_news(query, max_articles, use_requests)
    
    print(f"\n=== 검색 결과 ({len(results)}개) ===")
    for i, article in enumerate(results, 1):
        print(f"{i}. {article['title']}")
        print(f"   URL: {article['url']}")
        print("---")
    
    return results

def create_econotimes_tool():
    """LangChain tool 생성"""
    from langchain.tools import tool
    
    @tool
    def econotimes_search_tool(query: str, max_articles: int = 10) -> str:
        """
        EconoTimes에서 뉴스 기사를 검색합니다.
        
        Args:
            query: 검색할 키워드
            max_articles: 최대 기사 수 (기본값: 10)
        
        Returns:
            str: 검색 결과를 문자열로 반환
        """
        try:
            # 먼저 requests 방식으로 시도
            articles = search_econotimes_news(query, max_articles, use_requests=True)
            
            # 실패하면 Playwright 방식으로 시도
            if not articles:
                articles = search_econotimes_news(query, max_articles, use_requests=False)
            
            if not articles:
                return f"'{query}'에 대한 검색 결과가 없습니다."
            
            result = f"'{query}' 검색 결과 ({len(articles)}개):\n\n"
            
            for i, article in enumerate(articles, 1):
                result += f"{i}. {article['title']}\n"
                result += f"   URL: {article['url']}\n\n"
            
            return result
            
        except Exception as e:
            return f"검색 중 오류가 발생했습니다: {str(e)}"
    
    return econotimes_search_tool



# 테스트 실행
if __name__ == "__main__":

    
    # Playwright 방식으로 테스트
    print("\n2. Playwright 방식 테스트:")
    try:
        nvidia_news = quick_search("nvidia", 5, use_requests=False)
        if nvidia_news:
            print("✓ Playwright 방식 성공")
        else:
            print("✗ Playwright 방식 실패")
    except Exception as e:
        print(f"✗ Playwright 방식 오류: {e}")
