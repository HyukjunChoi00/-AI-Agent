### Powershell에서 streamlit run ctest.py   실행하면 됩니다.

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
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# 전문용어 주석 기능을 위한 추가 import
from datasets import load_dataset
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever

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
    model="HCX-007" # 모델명 입력 (기본값: HCX-005) 
)

###########################################
# RAG 데이터 불러오기
pdf_docs = []
pdf_loader = PyPDFLoader("경제금융용어 700선.pdf")
pdf_docs = pdf_loader.load()
qa_dataset = load_dataset("coorung/Kor-financial-qa-7K", split="train")
qa_docs = []
for example in qa_dataset:
    if "query" in example and "pos" in example and example["pos"]:
    text = f"질문: {example['query']}\n답변: {example['pos'][0]}"
    qa_docs.append(Document(page_content=text))
all_docs = pdf_docs + qa_docs
splitter = RecursiveCharacterTextSplitter(chunk_size=642, chunk_overlap=50)
split_docs = splitter.split_documents(all_docs)
retriever = BM25Retriever.from_documents(split_docs)
self.qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)
###########################################

# 전문용어 질의 그래프





##############################################

 

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

# 핵심 키워드 추출 Chain - JSON 파싱 방식
def build_keyword_extraction_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 사용자의 질문에서 핵심 검색 키워드를 추출하는 Assistant이다."),
         ("user", "다음 사용자의 질문에서 검색에 사용할 핵심 키워드만 추출하라. 반드시 영문 키워드로 제공하라.\n"
                  "예시:\n"
                  "- 'NVIDIA 주식 동향 조사좀 해줘' → 'NVIDIA'\n"
                  "- '테슬라 주가 전망이 어때?' → 'Tesla'\n"
                  "- '애플 아이폰 최신 소식 알려줘' → 'Apple'\n"
                  "- 'AI 기술 발전 현황' → 'AI'\n"
                  "질문: {query}\n\n"
                  "다음 JSON 형식으로만 응답해줘:\n"
                  '{{"expanded_search_query_list": ["추출된_키워드"]}}\n'
                  "핵심 키워드:")]
    )
    
    return prompt_template | chat

# Chain 정의 - 개선된 프롬프트
def build_query_expansion_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 키워드형 Query Expansion Assistant이다. 주어진 키워드와 관련된 산업, 기술, 시장 키워드로 확장해야 한다."),
         ("user", "다음 키워드를 확장하라:\n"
                  "원본 키워드: {query}\n"
                  "확장할 개수: {n}개\n\n"
                  "확장 규칙:\n"
                  "- Tesla → electric vehicle, EV, battery\n"
                  "- NVIDIA → AI, GPU, semiconductor\n"
                  "- Apple → iPhone, technology, consumer electronics\n"
                  "- Microsoft → cloud computing, software, Azure\n\n"
                  "'{query}' 키워드에 대해 관련 산업/기술 키워드 {n}개를 영문으로 생성하라.\n\n"
                  "반드시 다음 JSON 형식으로만 응답하라 (다른 텍스트 포함 금지):\n"
                  '{{"expanded_search_query_list": ["키워드1", "키워드2"]}}\n')]
    )
    
    return prompt_template | chat

#######################################################################################################################

class NewsAnalysis(BaseModel):
    """뉴스 분석 결과"""
    has_relevant_news: bool = Field(description="관련 뉴스가 있는지 여부")
    analysis_summary: str = Field(description="뉴스 분석 요약. bullet point를 활용한 개조식 답변. 만약 관련 뉴스가 없으면 공백으로 반환")

def build_news_analysis_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "너는 경제 뉴스 분석 전문가이다."),
         ("user", "다음 유저의 질문에 대해 검색된 뉴스내에 관련 정보가 있는지 확인하고 분석을 만들어라. 또한, 금리와 관세와 같은 일반적인 경제 상황을 다루는 정보도 전달하라. 금리의 경우 명확한 수치가 있다면 반드시 표기하라.\n"
                  "질문: {query}\n"
                  "검색된 뉴스: {news}\n"
                  "검색된 뉴스에서 '{query}'에 대해 관련 정보가 있는지 확인하고 분석을 만들어라\n"
                  "사용자가 특정 주식 종목에 대한 분석을 요청했다면, 다음 사항을 포함하세요:\n"
                  "1. 주요 트렌드와 동향 요약\n"
                  "2. 시장 영향 분석\n"
                  "3. 향후 전망이나 주목할 점\n"
                  "4. 전반적인 경제 상황\n\n"
                  "다음 JSON 형식으로만 응답해줘:\n"
                  '{{"has_relevant_news": true/false, "analysis_summary": "분석내용_또는_빈문자열"}}\n')]
    )
    
    return prompt_template | chat

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
    annotated_answer : str

# JSON 파싱 헬퍼 함수 - 더 안전한 버전
def parse_json_response(response) -> dict:
    """JSON 응답을 파싱하는 헬퍼 함수"""
    try:
        import json
        
        # AIMessage 객체에서 content 추출
        if hasattr(response, 'content'):
            content = str(response.content)
        else:
            content = str(response)
        
        print(f"[DEBUG] Raw response content: {content[:200]}...")  # 처음 200자만 출력
        
        # JSON 블록 찾기
        if '```json' in content:
            start = content.find('```json') + 7
            end = content.find('```', start)
            json_str = content[start:end].strip()
        elif '{' in content and '}' in content:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
        else:
            print("[DEBUG] No JSON found in response")
            return {"expanded_search_query_list": []}
        
        print(f"[DEBUG] Extracted JSON: {json_str}")
        
        result = json.loads(json_str)
        print(f"[DEBUG] Parsed result: {result}")
        return result
        
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON decode error: {e}")
        return {"expanded_search_query_list": []}
    except Exception as e:
        print(f"[DEBUG] General parsing error: {e}")
        return {"expanded_search_query_list": []}

# 노드 함수 정의 - JSON 파싱 방식
def extract_keyword(state):
    """핵심 키워드 추출 노드"""
    keyword_extraction_chain = build_keyword_extraction_chain()
    original_query = state["query"]
    
    try:
        print(f"[DEBUG] Starting keyword extraction for: {original_query}")
        # 핵심 키워드 추출
        response = keyword_extraction_chain.invoke({"query": original_query})
        print(f"[DEBUG] Keyword extraction response type: {type(response)}")
        
        parsed_response = parse_json_response(response)
        
        if parsed_response.get('expanded_search_query_list'):
            keyword = parsed_response['expanded_search_query_list'][0].strip()
            print(f"[DEBUG] Extracted keyword: {keyword}")
        else:
            # 백업: 원본 쿼리에서 영문 키워드 추출
            keyword = original_query.strip()
            print(f"[DEBUG] Using original query as keyword: {keyword}")
            
    except Exception as e:
        # 오류 발생 시 원본 쿼리 사용
        print(f"[DEBUG] Error in keyword extraction: {e}")
        st.warning(f"키워드 추출 중 오류: {e}")
        keyword = original_query.strip()
    
    return {"extracted_keyword": keyword}

def query_expansion(state):
    query_expansion_chain = build_query_expansion_chain()
    extracted_keyword = state["extracted_keyword"]
    
    try:
        print(f"[DEBUG] Starting query expansion for: {extracted_keyword}")
        response = query_expansion_chain.invoke({"query": extracted_keyword, "n": 2})
        print(f"[DEBUG] Query expansion response type: {type(response)}")
        
        parsed_response = parse_json_response(response)
        
        if parsed_response.get('expanded_search_query_list'):
            # 원본 키워드 + 확장된 키워드들
            expanded_list = parsed_response['expanded_search_query_list']
            expanded_query_list = [extracted_keyword] + expanded_list
            print(f"[DEBUG] Final expanded list: {expanded_query_list}")
        else:
            print("[DEBUG] No expanded queries found, using original keyword only")
            expanded_query_list = [extracted_keyword]
            
    except Exception as e:
        # 오류 발생 시 기본 키워드만 사용
        print(f"[DEBUG] Error in query expansion: {e}")
        st.warning(f"쿼리 확장 중 오류: {e}")
        expanded_query_list = [extracted_keyword]

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
    
    analysis_list = []
    
    for doc in search_results:
        try:
            response = news_analysis_chain.invoke({"query": doc['search_query'], "news": doc['content']})
            parsed_response = parse_json_response(response)
            
            analysis_list.append({
                "has_relevant_news": parsed_response.get('has_relevant_news', False),
                "analysis_summary": parsed_response.get('analysis_summary', ''),
                "search_query": doc['search_query'],
                "title": doc['title'],
                "url": doc['url'],
                "content": doc['content']
            })
        except Exception as e:
            # 개별 분석 실패 시 기본값으로 처리
            st.warning(f"뉴스 분석 중 오류: {e}")
            analysis_list.append({
                "has_relevant_news": False,
                "analysis_summary": "",
                "search_query": doc['search_query'],
                "title": doc['title'],
                "url": doc['url'],
                "content": doc['content']
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
    
    # 전문용어 주석 추가
    annotated_answer = term_annotator.explain_terms(response.content)

    return {"messages": [*messages, response],
            "answer": response.content,
            "annotated_answer" : annotated_answer}

################################################################그래프 연결 및 streamlit 적용

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

# 전문용어 주석 시스템 초기화
if 'term_annotator_initialized' not in st.session_state:
    st.session_state.term_annotator_initialized = False

graph = st.session_state.graph 
llm = st.session_state.llm 

###############################################################################
# Streamlit View

st.title("📰 금융 투자 지원 챗봇")

# 전문용어 주석 시스템 초기화 섹션
st.markdown("### 🔧 전문용어 주석 시스템")
col1, col2 = st.columns([3, 1])

with col1:
    if not st.session_state.term_annotator_initialized:
        st.warning("⚠️ 전문용어 주석 시스템이 초기화되지 않았습니다.")
        st.info("📌 전문용어 주석 기능을 사용하려면 초기화 버튼을 클릭하세요.")
    else:
        st.success("✅ 전문용어 주석 시스템이 초기화되었습니다.")

with col2:
    if st.button("🔧 전문용어 주석 시스템 초기화", type="primary"):
        with st.spinner("전문용어 주석 시스템 초기화 중..."):
            term_annotator.initialize()
            if term_annotator.is_initialized:
                st.session_state.term_annotator_initialized = True
                st.rerun()

st.markdown("---")

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
        st.markdown("- 전문용어 자동 주석")
        
        st.markdown("### 💡 사용 예시")
        st.markdown("- NVIDIA 주식 동향")
        st.markdown("- Tesla 전망")
        st.markdown("- Apple 실적 분석")
        st.markdown("- AI 관련 투자")
        
    else:
        st.markdown("### 📚 금융 사전")
        st.markdown("**기능:**")
        st.markdown("- 금융 전문용어 해설")
        st.markdown("- 투자 개념 설명")
        st.markdown("- 금융 상품 정보")
        st.markdown("- 전문용어 자동 주석")
        
        st.markdown("### 💡 사용 예시")
        st.markdown("- P/E 비율이란?")
        st.markdown("- ETF 설명해줘")
        st.markdown("- 배당수익률")
        st.markdown("- 옵션 거래")
    
    st.markdown("---")
    st.markdown("### 📋 필요 설치")
    st.code("pip install playwright beautifulsoup4")
    st.code("playwright install chromium")
    st.code("pip install datasets langchain")
    
    if PLAYWRIGHT_AVAILABLE:
        st.success("✅ Playwright 설치됨")
    else:
        st.error("❌ Playwright 설치 필요")

import streamlit as st
import asyncio

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
    <p>Powered by MIRAE ASSET + Naver ClovaX </p>
    <p><small>⚠️ 투자 결정은 신중히 하시고, 이 정보는 참고용입니다.</small></p>
</div>
""", unsafe_allow_html=True)