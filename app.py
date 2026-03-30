import streamlit as st
import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 페이지 및 스타일 설정
st.set_page_config(page_title="노인 비만 영양 관리 AI", page_icon="🥗", layout="wide")
st.title("🥗 노인 비만 맞춤형 영양 관리 AI 비서")
st.markdown("---")

# 지침 DB 연결 설정
GUIDELINE_URI = st.secrets["GUIDELINE_URI"]
GUIDELINE_USERNAME = st.secrets["GUIDELINE_USERNAME"]
GUIDELINE_PASSWORD = st.secrets["GUIDELINE_PASSWORD"]
GUIDELINE_DATABASE = st.secrets["GUIDELINE_DATABASE"]

RECIPE_URI = st.secrets["RECIPE_URI"]
RECIPE_USERNAME = st.secrets["RECIPE_USERNAME"]
RECIPE_PASSWORD = st.secrets["RECIPE_PASSWORD"]
RECIPE_DATABASE = st.secrets["RECIPE_DATABASE"]

# 2. 데이터베이스 및 API 연결 (캐싱 처리로 속도 향상)
@st.cache_resource
def init_connections():
    # load_dotenv()는 로컬 개발용으로만 유효, Streamlit Cloud에서는 st.secrets 사용
    load_dotenv()
    
    # 지침 DB
    graph_guideline = Neo4jGraph(
        url=GUIDELINE_URI,
        username=GUIDELINE_USERNAME,
        password=GUIDELINE_PASSWORD,
        database=GUIDELINE_DATABASE
    )
    
    # ✅ 레시피 DB도 st.secrets 사용 (os.getenv → st.secrets로 변경)
    graph_recipe = Neo4jGraph(
        url=RECIPE_URI,
        username=RECIPE_USERNAME,
        password=RECIPE_PASSWORD,
        database=RECIPE_DATABASE
    )
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return graph_guideline, graph_recipe, llm

graph_guideline, graph_recipe, llm = init_connections()

# 사이드바 상태 표시
with st.sidebar:
    st.header("⚙️ 시스템 정보")
    st.success(f"✅ 지침 DB: {graph_guideline._database}")
    st.success(f"✅ 레시피 DB: {graph_recipe._database}")
    st.info(f"👨‍🔬 연구원: 류혜리 (세계푸드테크창의센터)")

# 3. 핵심 RAG 로직 (혜리 님의 코드 이식)
def kg_enhanced_obesity_rag(question):
    try:
        # 지침 DB 검색 (Vector Store 객체는 미리 정의되어 있어야 함)
        # 예시에서는 vector_store를 전역 변수로 가정하거나 함수 내부에서 초기화
        docs = vector_store.similarity_search(question, k=3)
        doc_ids = [doc.metadata["id"] for doc in docs if "id" in doc.metadata]
        
        guideline_query = """
        MATCH (ch:Chapter)-[:HAS_SECTION]->(sec:Section)-[:HAS_RECOMMENDATION]->(reco:Recommendation)
        WHERE reco.id IN $doc_ids
        RETURN ch.title AS chapter_title, reco.content AS content
        """
        guideline_results = graph_guideline.query(guideline_query, {"doc_ids": doc_ids})
        full_guideline_text = " ".join([r['content'] for r in guideline_results])

        # 영양소 기준 설정
        min_protein = 20 if '단백질' in full_guideline_text or '근감소' in full_guideline_text else 0
        max_sodium = 500 if '나트륨' in full_guideline_text or '어르신' in full_guideline_text else 2000
        max_kcal = 600 if '저열량' in full_guideline_text or '비만' in full_guideline_text else 2000

        # 레시피 DB 탐색 (Bottom-Up 탐색)
        recipe_query = """
        MATCH (n:Nutrition)
        WHERE n.protein_g >= $min_protein 
          AND n.Sodium_mg <= $max_sodium
          AND n.energy_kcal <= $max_kcal
        MATCH (r:Recipe)-[:CONTAINS]->(n)
        OPTIONAL MATCH (f:Food)-[:HAS_INGREDIENT]->(r)
        RETURN 
            f.title AS food_title,
            r.title AS recipe_title,
            COLLECT(DISTINCT f.title) AS ingredients,
            n.energy_kcal AS kcal, 
            n.protein_g AS protein, 
            n.Sodium_mg AS sodium
        ORDER BY n.protein_g DESC
        LIMIT 3
        """
        recipe_results = graph_recipe.query(recipe_query, {
            "min_protein": min_protein, "max_sodium": max_sodium, "max_kcal": max_kcal
        })

        recipe_context = ""
        for rec in recipe_results:
            recipe_context += f"- 추천 메뉴: {rec['food_title']}\n"
            recipe_context += f"- 주요 레시피: {rec['recipe_title']}\n"
            recipe_context += f"  * 주요 식재료: {', '.join(rec['ingredients'])}\n"
            recipe_context += f"  * 영양분석: {rec['kcal']}kcal, 단백질 {rec['protein']}g, 나트륨 {rec['sodium']}mg\n\n"

        kg_context = f"[지침 근거]\n" + "\n".join([f"- {r['chapter_title']}: {r['content']}" for r in guideline_results])
        kg_context += f"\n\n[영양 분석 기반 추천 식단]\n" + (recipe_context if recipe_context else "조건에 맞는 레시피를 찾지 못했습니다.")

        prompt = PromptTemplate(
            template="""당신은 비만 지침의 영양 기준을 엄격히 준수하는 AI 영양사입니다. 
            주어진 지침 근거를 먼저 설명하고, 영양 수치가 검증된 레시피를 추천하세요. 
            {context} 
            질문: {question}""", 
            input_variables=["context", "question"]
        )
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": question, "context": kg_context})
    except Exception as e:
        return f"⚠️ 오류 발생: {str(e)}"

# 4. 사용자 UI 구성
st.subheader("❓ 맞춤형 식단 상담")
user_input = st.text_input("상담하고 싶은 내용을 입력하세요.", placeholder="비만인 어르신을 위한 단백질 식단을 추천해 주세요.")

if st.button("결과 확인"):
    if user_input:
        with st.spinner("지침과 식단 데이터를 분석 중입니다..."):
            answer = kg_enhanced_obesity_rag(user_input)
            
            # 레이아웃 분할 출력
            st.success("✅ 분석 완료")
            st.markdown("### 🤖 인공지능 영양사 답변")
            st.write(answer)
            
            st.divider()
            st.caption("※ 본 정보는 대한비만학회 진료지침 2022를 근거로 작성되었습니다.")
    else:
        st.warning("내용을 입력해 주세요.")

