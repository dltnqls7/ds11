import streamlit as st
import pandas as pd

# 페이지 기본 설정 (아이콘 + 브라우저 탭 제목)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 데이터 불러오기
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# 화면 제목
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("뇌졸중 발생 여부와 관련된 데이터를 살펴보고 분석 연습을 해보는 공간입니다.")

st.divider()

# 큰 숫자 카드 4개
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = round(stroke_count / total_people * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중(stroke=1) 사람 수", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio} %")

st.divider()

# 열 설명 표 만들기
st.subheader("📋 데이터 열 설명")

col_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [""] * len(df.columns),   # 직접 채워 넣을 칸 (비워둠)
    "값의 종류": [
        str(df[col].unique()[:5]) + (" ..." if df[col].nunique() > 5 else "")
        for col in df.columns
    ],
    "빈 값 개수": [df[col].isna().sum() for col in df.columns]
})

st.dataframe(col_info, use_container_width=True)

st.divider()

# 데이터 미리보기 (처음 5줄)
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# 데이터 출처 (직접 작성할 자리)
st.subheader("📚 데이터 출처")
st.info("여기에 교재에 있는 데이터 출처를 적어 주세요.")
