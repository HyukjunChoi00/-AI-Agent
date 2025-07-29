import json
from typing import TypedDict, Annotated, List, Dict, Optional
import operator
import os
import asyncio
import re
from urllib.parse import urlparse, urljoin
import nest_asyncio
nest_asyncio.apply()
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

# Playwright 관련 임포트

try:
    from playwright.async_api import async_playwright
    from bs4 import BeautifulSoup, Comment
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    st.error("Playwright 또는 BeautifulSoup이 설치되지 않았습니다. 다음 명령어로 설치해주세요:")
    st.code("pip install playwright beautifulsoup4")
    st.code("playwright install chromium")

# 환경 변수 설정
os.environ["CLOVASTUDIO_API_KEY"] = ''

from langchain_naver import ChatClovaX
  
chat = ChatClovaX(
    model="HCX-DASH-002" # 모델명 입력 (기본값: HCX-005) 
)

### Econotimes로부터 뉴스 검색을 수행하기 위한 클래스 정의  ##################

class EconoTimesFullPipeline:
    def __init__(self, max_articles=5):
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

    async def extract_articles(self, page, query: str) -> List[Dict[str, str]]:
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

    async def run_pipeline(self, query: str) -> List[Dict[str, str]]:
        await self.setup_browser()
        search_url = f"https://www.econotimes.com/search?v={query}"
        page = await self.context.new_page()
        await page.goto(search_url, wait_until='domcontentloaded', timeout=20000)
        await asyncio.sleep(2)
        articles = await self.extract_articles(page, query)
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

####################################################################################################################
## 쿼리 확장 파트

class ExpandedQuery(BaseModel):
    expanded_search_query_list: List[str] = Field(description="질의 확장된 검색 키워드형 질의 리스트. 키워드형 검색 엔진을 위해")

# 핵심 키워드 추출 Chain
def build_keyword_extraction_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 사용자의 질문에서 핵심 검색 키워드를 추출하는 Assistant이다."),
         ("user", "다음 사용자의 질문에서 검색에 사용할 핵심 키워드만 추출하라. 반드시 영문 키워드로 제공하라.\n"
                  "예시:\n"
                  "- 'NVIDIA 주식 동향 조사좀 해줘' → 'NVIDIA'\n"
                  "- '테슬라 주가 전망이 어때?' → 'Tesla'\n"
                  "- '애플 아이폰 최신 소식 알려줘' → 'Apple'\n"
                  "- 'AI 기술 발전 현황' → 'AI'\n"
                  "질문: {query}\n"
                  "핵심 키워드:")]
    )
    return prompt_template | chat
    )


# Chain 정의
def build_query_expansion_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 키워드형 Query Expansion Assistant이다."),
         ("user", "다음 유저의 질의를 확장하라. 유저가 입력한 주식 종목과 관련된 산업에 대한, 주요 키워드로 확장하라. 반드시 영문 키워드만 이용하라.예를 들어, NVIDIA에 대한 키워드를 받으면 AI, GPU 등으로 확장할 수 있다.\n"
                  "질의: {query}\n"
                  "확장 질의 갯수: {n}\n"
                  "질의 '{query}'에 대해 {n}개의 확장된 키워드형 질의를 만들어라.")]
    )
    query_expansion_chain = prompt_template | chat.with_structured_output(ExpandedQuery)
    
    return query_expansion_chain

#######################################################################################################################

class NewsAnalysis(BaseModel):
    """뉴스 분석 결과"""
    has_relevant_news: bool = Field(description="관련 뉴스가 있는지 여부")
    analysis_summary: str = Field(description="뉴스 분석 요약. bullet point를 활용한 개조식 답변. 만약 관련 뉴스가 없으면 공백으로 반환")

def build_news_analysis_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 경제 뉴스 분석 전문가이다."),
         ("user", "다음 유저의 질문에 대해 검색된 뉴스내에 관련 정보가 있는지 확인하고 분석을 만들어라\n"
                  "질문: {query}\n"
                  "검색된 뉴스: {news}\n"
                  "검색된 뉴스에서 '{query}'에 대해 관련 정보가 있는지 확인하고 분석을 만들어라\n"
                  "사용자가 특정 주식 종목에 대한 분석을 요청했다면, 다음 사항을 포함하세요:\n"
                  "1. 주요 트렌드와 동향 요약\n"
                  "2. 시장 영향 분석\n"
                  "3. 향후 전망이나 주목할 점\n"
                  "4. 전반적인 경제 상황")]
    )
    
    news_analysis_chain = prompt_template | chat.with_structured_output(NewsAnalysis)
    
    return news_analysis_chain



# 그래프 상태 정의
class State(TypedDict):
    query: str
    extracted_keyword: str
    expanded_query_list: List[str]
    analysis_list: List[str]
    answer: str
    messages: Annotated[List[BaseMessage], operator.add]
    search_results: List[dict]
    search_results2 : List[dict]


# 노드 함수 정의
def extract_keyword(state):
    """핵심 키워드 추출 노드"""
    keyword_extraction_chain = build_keyword_extraction_chain()
    original_query = state["query"]
    
    # 핵심 키워드 추출
    extracted_keyword = keyword_extraction_chain.invoke({"query": original_query})
    
    # 추출된 키워드 정리 (공백 제거, 첫 글자 대문자 등)
    keyword = extracted_keyword.content.strip()
    
    return {"extracted_keyword": keyword}


def query_expansion(state):
    query_expansion_chain = build_query_expansion_chain()
    extracted_keyword = state["extracted_keyword"]
    
    expanded_query_obj = query_expansion_chain.invoke({"query": extracted_keyword,
                                                       "n": 2}) 
    expanded_query_list = [extracted_keyword, *expanded_query_obj.expanded_search_query_list]

    return {"expanded_query_list": expanded_query_list}

############################################################################### 아래 비동기 함수 처리가 제일 문제임.

async def search(state):
    expanded_query_list = state["expanded_query_list"]
    pipeline = EconoTimesFullPipeline(max_articles=3)
    all_results = []
    await pipeline.setup_browser()
    try:
        for query in expanded_query_list:
            try:
                results = await pipeline.run_pipeline(query)
                if not results:
                    # 결과가 없으면 그냥 다음 쿼리로
                    continue
                
                # 각 결과에 검색 쿼리 정보 추가
                all_results.extend([{
                    "title": article["title"],
                    "url": article["url"],
                    "content": article["content"],
                    "search_query": query
                } for article in results])
                
            except Exception as e:
                # 예외 발생 시 로깅하거나 무시하고 다음 쿼리 계속 진행 가능
                # 필요하면 아래 주석 제거 후 로깅 추가
                # print(f"Error processing query '{query}': {e}")
                continue
    finally:
        await pipeline.close_browser()

    
    return {"search_results": all_results}


#############################################################################

async def general_economics(state):
    pipeline = EconoTimesFullPipeline(max_articles=3)
    expanded_query_list = ['tariff', 'interest rate']
    all_results = []
    await pipeline.setup_browser()
    try:
        for query in expanded_query_list:
            try:
                results = await pipeline.run_pipeline(query)
                if not results:
                    # 결과가 없으면 그냥 다음 쿼리로
                    continue
                
                # 각 결과에 검색 쿼리 정보 추가
                all_results.extend([{
                    "title": article["title"],
                    "url": article["url"],
                    "content": article["content"],
                    "search_query": query
                } for article in results])
                
            except Exception as e:
                # 예외 발생 시 로깅하거나 무시하고 다음 쿼리 계속 진행 가능
                # 필요하면 아래 주석 제거 후 로깅 추가
                # print(f"Error processing query '{query}': {e}")
                continue
    finally:
        await pipeline.close_browser()

    
    return {"search_results2": all_results}

#############################################################################

def make_analysis_for_each_query(state):
    news_analysis_chain = build_news_analysis_chain()

    search_results = state["search_results"]
    search_results += state["search_results2"]
    
    analysis_obj_list = news_analysis_chain.batch([{"query": doc['search_query'],
                                                   "news": doc['content']}
                                                   for doc in search_results])

    analysis_list = []
    for analysis_obj, search_item in zip(analysis_obj_list, search_results):
        analysis_list.append({**analysis_obj.dict(),
                             "search_query": search_item['search_query'],
                             "title": search_item['title'],
                             "url": search_item['url'],
                             "content": search_item['content']
                             })
    return {"analysis_list": analysis_list}

#############################################################################

def generate_response(state):
    context = "\n\n".join([analysis['analysis_summary']
                           for analysis in state["analysis_list"]
                           if analysis['has_relevant_news']])
    
    curr_human_turn = HumanMessage(content=f"질문: {state['query']}\n"
                            f"```\n{context}```"
                             "\n---\n"
                             "응답은 markdown을 이용해 리포트 스타일로 한국어로 응답해라. "
                             "사용자의 질문의 의도에 맞는 정답 부분을 강조해라.")
    messages = state["messages"] + [curr_human_turn]
    response = llm.invoke(messages)

    return {"messages": [*messages, response],
            "answer": response.content}

# Streamlit 앱 초기화
if 'graph' not in st.session_state:
    llm = chat

    # 그래프 구성
    workflow = StateGraph(State)
    workflow.add_node("extract_keyword", extract_keyword)
    workflow.add_node("query_expansion", query_expansion)
    workflow.add_node("search", search)
    workflow.add_node("general economic", general_economics)
    workflow.add_node("make_analysis_for_each_query", make_analysis_for_each_query)
    workflow.add_node("generate", generate_response)
    workflow.add_edge("extract_keyword", "query_expansion")
    workflow.add_edge("query_expansion", "search")
    workflow.add_edge("search", "general economic")
    workflow.add_edge("general economic", "make_analysis_for_each_query")
    workflow.add_edge("make_analysis_for_each_query", "generate")

    workflow.set_entry_point("extract_keyword")
    workflow.set_finish_point("generate")

    graph = workflow.compile()

    st.session_state.graph = graph
    st.session_state.llm = llm

graph = st.session_state.graph 
llm = st.session_state.llm 

###############################################################################
# Streamlit View
st.title("📰 금융 투자 지원 챗봇 (EconoTimes)")

# 사이드바 설정
with st.sidebar:
    st.title("🤖 챗봇 선택")
    
    chatbot_mode = st.selectbox(
        "챗봇 모드를 선택하세요:",
        ["투자 의사결정 지원 챗봇", "금융 사전 챗봇"],
        index=0
    )
    
    st.markdown("---")
    
    if chatbot_mode == "투자 의사결정 지원 챗봇":
        st.markdown("### 📈 투자 의사결정 지원")
        st.markdown("**기능:**")
        st.markdown("- 실시간 금융 뉴스 분석")
        st.markdown("- 종목별 시장 동향 분석")
        st.markdown("- 투자 관련 리포트 생성")
        
        st.markdown("### 💡 사용 예시")
        st.markdown("- NVIDIA 주식 동향")
        st.markdown("- Tesla 전망")
        st.markdown("- Apple 실적 분석")
        st.markdown("- AI 관련 투자")
        
    else:
        st.markdown("### 📚 금융 사전")
        st.markdown("**기능:**")
        st.markdown("- 금융 용어 설명")
        st.markdown("- 투자 개념 해설")
        st.markdown("- 금융 상품 정보")
        
        st.markdown("### 💡 사용 예시")
        st.markdown("- P/E 비율이란?")
        st.markdown("- ETF 설명해줘")
        st.markdown("- 배당수익률")
        st.markdown("- 옵션 거래")
    
    st.markdown("---")
    st.markdown("### 📋 필요 설치")
    st.code("pip install playwright beautifulsoup4")
    st.code("playwright install chromium")
    
    if PLAYWRIGHT_AVAILABLE:
        st.success("✅ Playwright 설치됨")
    else:
        st.error("❌ Playwright 설치 필요")

import streamlit as st
import asyncio

# graph는 이미 준비되어 있다고 가정

async def async_stream(query):
    # 비동기 스트리밍 함수
    async for event in graph.astream({"query": query}, debug=True):
        yield event

def run_async_stream(query):
    # Streamlit 동기 함수에서 비동기 제너레이터를 동기적으로 실행하는 헬퍼
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    gen = async_stream(query)
    
    async def gather_events():
        events = []
        async for event in gen:
            events.append(event)
        return events

    return loop.run_until_complete(gather_events())

if query := st.chat_input("검색할 키워드를 입력하세요"):
    if not PLAYWRIGHT_AVAILABLE:
        st.error("Playwright가 설치되지 않았습니다. 사이드바의 설치 명령어를 참고해주세요.")
    else:
        st.subheader(f"🔍 검색: {query}")
        st.subheader("🤖 답변")
        with st.spinner("답변 생성중..."):
            try:
                # 비동기 제너레이터에서 모든 이벤트를 받아옴 (동기 함수 안에서 실행)
                events = run_async_stream(query)

                # 받은 이벤트들을 화면에 출력
                for event in events:
                    for k, v in event.items():
                        if k == 'extract_keyword':
                            with st.container():
                                st.write("### 🔑 추출된 핵심 키워드")
                                st.markdown(f"**원본 질문:** {query}")
                                st.markdown(f"**추출된 키워드:** {v['extracted_keyword']}")
                        
                        elif k == 'query_expansion':
                            with st.container():
                                st.write("### 🔍 확장된 쿼리 리스트")
                                expanded_query_md = '\n'.join([f"- {q}" for q in v['expanded_query_list']])
                                st.markdown(expanded_query_md)
                        
                        elif k == 'search':
                            with st.expander("📰 검색된 뉴스 (EconoTimes)"):
                                for search_item in v['search_results']:
                                    with st.container():
                                        st.markdown(f"**제목:** {search_item['title']}")
                                        st.markdown(f"**검색 쿼리:** {search_item['search_query']}")
                                        st.markdown(f"**URL:** [{search_item['url']}]({search_item['url']})")
                                        st.markdown(f"**내용:** {search_item['content'][:500]}...")
                                        
                        elif k == 'general economic':
                            with st.expander("📰 검색된 뉴스 (일반 경제 시장)"):
                                for search_item in v['search_results2']:
                                    with st.container():
                                        st.markdown(f"**제목:** {search_item['title']}")
                                        st.markdown(f"**검색 쿼리:** {search_item['search_query']}")
                                        st.markdown(f"**URL:** [{search_item['url']}]({search_item['url']})")
                                        st.markdown(f"**내용:** {search_item['content'][:500]}...")
                        
                        elif k == 'make_analysis_for_each_query':
                            with st.expander("📊 뉴스 분석"):
                                for analysis in v['analysis_list']:
                                    st.markdown(f"**검색 쿼리:** {analysis['search_query']}")
                                    st.markdown(f"**관련 뉴스 존재:** {'✅' if analysis['has_relevant_news'] else '❌'}")
                                    if analysis['has_relevant_news']:
                                        st.markdown(f"**분석 요약:** {analysis['analysis_summary']}")
                                    st.markdown("---")
                                    
                       
                        elif k == 'generate':
                            st.markdown("## 📋 최종 분석 리포트")
                            st.markdown(v['answer'])

            except Exception as e:
                st.error(f"처리 중 오류 발생: {e}")
                st.info("잠시 후 다시 시도해주세요.")


# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Powered by Streamlit + LangGraph + Google Gemini + EconoTimes</p>
    <p><small>⚠️ 투자 결정은 신중히 하시고, 이 정보는 참고용입니다.</small></p>
</div>
""", unsafe_allow_html=True)
