import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 페이지 기본 설정 및 데이터 불러오기
# ==========================================
st.set_page_config(page_title="MMCA 전시 큐레이션", page_icon="🎨", layout="wide")

@st.cache_data # 데이터를 한 번만 불러와서 앱 속도를 높이는 캐싱 기능
def load_data():
    past_df = pd.read_excel('past.xlsx')
    current_df = pd.read_excel('current.xlsx')
    
    # 결측치 처리
    for col in ['top_tags', 'genres']:
        past_df[col] = past_df[col].fillna('')
        current_df[col] = current_df[col].fillna('')
        
    return past_df, current_df

past_df, current_df = load_data()

# ==========================================
# 2. UI: 온보딩 (과거 전시 선택 화면)
# ==========================================
st.title("🎨 당신을 위한 맞춤형 전시 큐레이션")
st.markdown("#### 과거에 관람했거나 흥미로워 보이는 전시를 모두 골라주세요!")

# 사용자가 멀티셀렉트로 여러 전시를 고를 수 있게 함
selected_past_titles = st.multiselect(
    "전시 검색 및 선택", 
    options=past_df['title'].tolist(),
    placeholder="여기를 클릭하여 전시를 선택하세요 (여러 개 선택 가능)"
)

# ==========================================
# 3. 추천 알고리즘 연산 및 결과 출력 (버튼 클릭 시)
# ==========================================
if st.button("내 취향 분석하고 추천 받기"):
    if not selected_past_titles:
        st.warning("최소 1개 이상의 전시를 선택해 주세요!")
    else:
        with st.spinner("사용자 취향을 분석 중입니다..."):
            
            # --- [데이터 처리] 사용자가 선택한 전시들의 메타데이터 모으기 ---
            user_selected_df = past_df[past_df['title'].isin(selected_past_titles)]
            
            # 사용자의 선호 태그와 장르를 하나의 텍스트로 결합 (취향 뭉치 생성)
            user_tags = " ".join(user_selected_df['top_tags'].tolist())
            user_genres = " ".join(user_selected_df['genres'].tolist())
            
            # 뼈대(장르) 30% + 살코기(태그) 70% 하이브리드 가중치를 위해 태그를 3번 반복
            user_profile_text = user_genres + " " + (user_tags + " ") * 3
            
            # 현재 전시들의 분석용 텍스트 생성
            current_df['features'] = current_df['genres'] + " " + (current_df['top_tags'] + " ") * 3
            
            # --- [알고리즘] TF-IDF 및 코사인 유사도 계산 ---
            tfidf = TfidfVectorizer()
            # 유저 프로필과 현재 전시 데이터를 합쳐서 기준 사전을 만듦
            all_text = [user_profile_text] + current_df['features'].tolist()
            tfidf_matrix = tfidf.fit_transform(all_text)
            
            # 0번 인덱스는 유저 취향 벡터, 1번부터는 현재 전시 벡터
            user_vector = tfidf_matrix[0:1]
            current_vectors = tfidf_matrix[1:]
            
            # 유사도 계산
            sim_scores = cosine_similarity(user_vector, current_vectors).flatten()
            current_df['similarity'] = sim_scores
            
            # 상위 3개 전시 추출
            top_3_recommendations = current_df.sort_values(by='similarity', ascending=False).head(3)
            
            # --- [UI] 추천 결과 화면 출력 ---
            st.divider()
            st.markdown("### ✨ 분석 완료! 취향 저격 전시 TOP 3")
            
            # 화면을 3개의 열(Column)로 나누어 카드 형태로 예쁘게 출력
            cols = st.columns(3)
            for idx, (i, row) in enumerate(top_3_recommendations.iterrows()):
                with cols[idx]:
                    st.info(f"**{idx+1}위. {row['title']}**")
                    st.write(f"🏢 **장소:** {row['cntc_instt_nm']}")
                    st.write(f"📅 **기간:** {row['period']}")
                    st.write(f"🏷️ **핵심 태그:** {row['top_tags']}")
                    st.write(f"🎨 **매칭 장르:** {row['genres']}")
                    # 유사도를 퍼센트로 변환하여 직관적으로 보여줌
                    st.metric(label="취향 일치도", value=f"{row['similarity'] * 100:.1f}%")
