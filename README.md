# 뉴스 데이터 기반 투자 의사결정 지원 시스템 : HYPERCLOVA을 활용한 AI Agent 시스템

#### 사용법
```
streamlit run dasan509.py
```

#### 패키지 설치
```
pip install \
    streamlit \
    nest_asyncio \
    langgraph \
    langchain \
    langchain-core \
    langchain-community \
    langchain-navar \
    pydantic \
    datasets \
    playwright \
    beautifulsoup4 \
    lxml
```

#### Playwright 설치
```
playwright install chromium
```

#### CLOVASTUDIO_API_KEY
코드 49번 줄에 API KEY를 입력
```
os.environ["CLOVASTUDIO_API_KEY"] = ''
```

해당 프로젝트는, 사용자의 투자 결정 의사지원을 돕기 위한 AI Agent 시스템입니다.  
사용자의 질의에 맞는 뉴스 기사를 검색하여, 종합적인 분석 내용을 응답해줍니다.  
특히, 경제 전문 뉴스를 다루는 Econotimes의 뉴스를 추출하는 방식을 선택했습니다.  
Workflow는 아래와 같습니다.  

----------------------------------------
  



<img width="1827" height="752" alt="image" src="https://github.com/user-attachments/assets/a35942ff-f549-4555-acbe-b9768d544a5e" />

----------------------------------------

- 사용자의 질의에 따른 뉴스 기사 검색  ex) NVIDA 주식 동향이 어때?
- 쿼리 확장을 통해 사용자 질의와 관련된 파생 키워드 생성 ex) NVIDA --> AI, GPU
- 확장된 쿼리 키워드를 기반으로 뉴스 검색 수행
- '관세'와 '금리'를 키워드로 일반적인 세계 경제 상황 분석
- streamlit run ctest.py
- 위 코드 실행 시, 금융 뉴스 검색 기반으로 사용자가 관심있는 주식 종목 관련 동향 및 일반적인 경제 상황 분석 가능

-------------------------

<img width="781" height="750" alt="image" src="https://github.com/user-attachments/assets/e95874d4-f80f-40fc-814e-56e6ddd22a89" />

