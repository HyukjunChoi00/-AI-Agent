#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install playwright


# In[ ]:


import subprocess

# Playwright 브라우저 설치 (chromium)
subprocess.run(["python3", "-m", "playwright", "install", "chromium"])


# In[1]:


import nest_asyncio
import asyncio
from playwright.async_api import async_playwright

# Jupyter 환경에서 event loop 충돌 방지
nest_asyncio.apply()

async def search_ft_world_page(keyword: str):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # headless=True 필수 (GUI 없음)
        page = await browser.new_page()
        
        await page.goto("https://www.ft.com/world", timeout=60000)

        # 기사 헤드라인 링크 기다리기
        await page.wait_for_selector("a[data-trackable='heading-link']")

        # 기사 목록 가져오기
        articles = await page.query_selector_all("a[data-trackable='heading-link']")

        results = []
        for article in articles:
            title = await article.inner_text()
            href = await article.get_attribute("href")
            if title and keyword.lower() in title.lower():
                full_url = "https://www.ft.com" + href
                results.append((title.strip(), full_url))
        
        await browser.close()
        return results

# 🔍 실제 검색 실행 (Jupyter에서)
keyword = "China"  # 사용자 입력 질의
results = await search_ft_world_page(keyword)

# ✅ 결과 출력
for title, link in results:
    print(f" {title}\n🔗 {link}\n")


# In[ ]:




