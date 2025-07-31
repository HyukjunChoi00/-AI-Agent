from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing_extensions import TypedDict
from typing import List

#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field


voyageai_key = ''
hf_token = ""
openai_key = ''

# 데이터 불러와서 벡터 DB에 저장
loader = CSVLoader(file_path="updated_data_1.csv")
data = loader.load()

embedding_model = VoyageAIEmbeddings(
    voyage_api_key = voyageai_key, model = "voyage-multilingual-2"
)

#db = Chroma.from_documents(documents = data, embedding = embedding_model, persist_directory="./vector_db/chroma_langgraph")
db = Chroma(persist_directory = "./chroma_langgraph", embedding_function = embedding_model)





class GraphState(TypedDict):
    question: str
    answer: str
    documents: List[Document]
    rewrite_count: int




def retrieve(state):
    print("---VectorDB에서 검색---")
    question = state["question"]
    docs = db.similarity_search(question, k=20)
    return {"documents": docs, "question": state["question"], "rewrite_count": state["rewrite_count"]}




class GradeDocuments(BaseModel):
    """
    문서와 질문의 관련성을 평가하기 위한 데이터 모델.
    """
    binary_score: str = Field(
        description="문서가 질문과 관련이 있는지, 'yes' 또는 'no'"
    )


def documents_grader(state):
    """
    검색된 문서와 사용자 질문의 관련성을 평가하여 관련 문서만 반환.
    """
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key = openai_key)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    SYS_PROMPT = """
    당신은 검색된 문서와 사용자 질문의 관련성을 평가해야 합니다.
    문서 내용과 질문을 바탕으로 문서가 질문에 관련이 있는 경우 'yes', 그렇지 않은 경우 'no'를 반환하세요.
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            ("human", """질문: {question}\n문서 내용: {document}"""),
        ]
    )

    relevant_docs = []
    grader = grade_prompt | structured_llm_grader
    for doc in state["documents"]:
        grading_input = {"document": doc.page_content, "question": state["question"]}
        result = grader.invoke(grading_input)
        if result.binary_score == "yes":
            relevant_docs.append(doc)
    
    return {"documents": relevant_docs, "question": state["question"], "rewrite_count": state["rewrite_count"]} 



def decide_to_generate(state):
    if state["documents"] or state["rewrite_count"] >= 3:
        return "useful"
    else:
        return "not useful"
    

from langchain_core.output_parsers import StrOutputParser



def query_rewriter(state):
    rewrite_count = state["rewrite_count"] + 1

    SYS_PROMPT = """
    질문을 다시 작성하는 역할을 수행하고 다음 작업을 수행:
    - 입력 질문을 더 명확하고 최적화된 질문으로 변환하세요.
    - 다시 작성할 때 입력 질문을 살펴보고 기본 의미론적 의도/의미에 대해 추론하세요.
    """
    print("Input question:", state["question"])  # 입력 데이터 확인
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            ("human", """초기 질문: {question}\n개선된 질문 작성."""),
        ]
    )

    question = state["question"]
    print("Input question:", question)

    formatted_prompt = rewrite_prompt.format_messages(question=question)
    print("Formatted Prompt:", formatted_prompt)
    
    revised_question = llm.invoke(formatted_prompt) 
    print("Rewritten question:", revised_question)

    return {"question": revised_question.content, "rewrite_count": rewrite_count, "documents": state["documents"]}    


llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, openai_api_key = openai_key)


prompt_template = ChatPromptTemplate.from_template("""
당신은 대학생에게 강의(과목) 검색 가이드를 제공하는 교직원입니다. 
검색된 Document 조각을 사용하여 질문에 답을 하세요. 즉, 모든 내용은 제공한 벡터 데이터베이스에 기반해야 하며, 질문에 관한 내용을 제공된 데이터베이스에서 찾지 못한다면 없다고 솔직하게 답변해야 합니다.

제공된 Document에 해당하지 않는 답변을 구성하지 마십시오.
답변 구성 시 Document값이 Null이면 "RAG에 관련정보가 없습니다"라고 답변을 하세요.
질문에 대한 자세하게 요약한 답변을 해주세요.

참고로 학생들의 주 연령대는 20대 초반입니다. 따라서 학생들이 입력하는 질문이 다소 모호할 수도 있다는 점을 생각해야 합니다.

- 특정 학과의 강의에 대한 질문을 할 경우 '개설학부' 칼럼을 참고하세요.
- 선수과목은 '선수과목명' 칼럼을 참조하고, 기초과목 관련 내용은 '기초과목명' 칼럼을 참조하세요. 
- 동시 수강 추천이나 같이 들으면 좋은 과목과 같은 질문 내용이 있다면 '동시수강추천과목명' 칼럼을 참조하세요.
- '선수과목명'이나 '기초과목명' 칼럼에 대한 필드가 채워져 있다면 난이도가 쉽지 않은 과목으로 판단하세요. 반대로 두 칼럼에 대한 필드가 모두 비워져 있다면 난이도가 낮은 과목으로 판단하세요.
- 원어 강의(영어 강의)의 경우 '영어등급구분' 칼럼을 보고 판단하세요. A등급(100%영어)인 경우 원어 강의입니다.
- 수업 자료에 관련된 질문은 '수업체계명' 칼럼을 참조하세요. 설명에 'lecture note', '강의노트', 'pdf파일', '부교재' 등의 내용이 있다면 답변에 포함시키세요.
- '개설강좌번호' 칼럼의 정보를 반드시 알려주세요.


다음은 "수업시간및강의실" 칼럼 데이터 해석 방법입니다. 
예를 들어서 "화D 목C"는 화요일 D교시와 목요일 C교시에 강의가 진행된다는 것을 의미합니다. 또한 "월10 월11 월12"는 월요일 10교시부터 12교시까지 수업이 진행된다는 것을 의미합니다.
그리고 아주대학교의 시간표 체계는 아래와 같습니다. 오전 / 오후 등 시간대와 관련된 질문을 받을 때는 아래의 시간표 기준을 참조하여 답변하세요.

표기 방법 1.  
A교시 : 09:00 ~ 10:15  
B교시 : 10:30 ~ 11:45  
C교시 : 12:00 ~ 13:15  
D교시 : 13:30 ~ 14:45  
E교시 : 15:00 ~ 16:15  
F교시 : 16:30 ~ 17:45  
G교시 : 18:00 ~ 19:15  

표기 방법 2.  
1교시 : 09:00 ~ 09:50      
2교시 : 10:00 ~ 10:50    
3교시 : 11:00 ~ 11:50     
4교시 : 12:00 ~ 12:50     
5교시 : 13:00 ~ 13:50     
6교시 : 14:00 ~ 14:50     
7교시 : 15:00 ~ 15:50       
8교시 : 16:00 ~ 16:50  
9교시 : 17:00 ~ 17:50  
10교시 : 18:00 ~ 18:50  
11교시 : 19:00 ~ 19:50  
12교시 : 20:00 ~ 20:50


Question: {question}
Document: {documents}
Answer:
""")


import json
def generater(state):
    if state["rewrite_count"] >= 3:
        answer = "관련 내용이 없어 답변을 생성하지 못했습니다."
    else:
        context_docs = "\n\n".join(doc.page_content for doc in state["documents"])

        formatted_prompt = prompt_template.format_messages(
            question=state["question"],
            documents=context_docs # or "NULL"
        )

        
        temp_ = llm.invoke(formatted_prompt)
        answer = temp_.content

    state["rewrite_count"] = 0

    

    print(answer)

    return {
        "answer": answer,
        "documents": state["documents"],
        "question": state["question"],
        "rewrite_count": state["rewrite_count"]
    }    




workflow = StateGraph(GraphState)

workflow.set_entry_point("retrieve")

workflow.add_node("retrieve", retrieve)
workflow.add_node("query_rewriter", query_rewriter)
workflow.add_node("documents_grader", documents_grader)
workflow.add_node("generater", generater)

workflow.add_edge("retrieve", "documents_grader")
workflow.add_conditional_edges(
    "documents_grader",
    decide_to_generate,
    {
        "useful": "generater",
        "not useful": "query_rewriter",
    }
)
workflow.add_edge("query_rewriter", "retrieve")
workflow.add_edge("generater", END)

chatbot = workflow.compile()




import streamlit as st


st.title("LangGraph 기반 RAG 챗봇")
st.write("질문을 입력하고 Enter 키를 눌러 답변을 확인하세요.")


user_question = st.text_input("질문:", placeholder="예: 경영인텔리전스학과 수업 중에 선수과목 없어도 들을 수 있는 수업이 있나?")


if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []


if st.button("질문하기"):
    if user_question.strip():
        initial_state = {"question": user_question.strip(), "answer": "", "documents": [], "rewrite_count": 0}

        try:
            result = chatbot.invoke(initial_state)
            answer = result.get("answer", "답변을 생성하지 못했습니다.")
            context_used = result.get("documents", "사용된 문맥이 없습니다.")

            st.session_state["conversation_history"].append(
                {
                    "question": user_question.strip(),
                    "answer": answer,
                    "documents": context_used,
                }
            )
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("질문을 입력하세요.")


st.subheader("대화 기록")
if st.session_state["conversation_history"]:
    for i, entry in enumerate(st.session_state["conversation_history"]):
        st.write(f"**질문 {i+1}:** {entry['question']}")
        st.write(f"**답변:** {entry['answer']}")
        with st.expander("사용된 문맥 보기"):
            st.write(entry["documents"])
else:
    st.write("아직 질문이 입력되지 않았습니다.")


# streamlit run chatbot.py
