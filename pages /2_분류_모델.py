import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# 페이지 기본 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
    layout="wide"
)

# 데이터 불러오기
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("🤖 분류 모델 만들기")
st.markdown("나이, 평균 혈당, 체질량지수, 고혈압, 심장병 정보를 이용해 뇌졸중을 예측하는 모델을 만들어 봅니다.")

st.divider()

# 열 이름 <-> 우리말 이름 매핑
col_name_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
reverse_col_name_map = {v: k for k, v in col_name_map.items()}

all_features_kor = list(col_name_map.values())
default_features_kor = [col_name_map[c] for c in ["age", "avg_glucose_level", "hypertension", "heart_disease"]]

# 속성 선택
st.subheader("① 입력으로 사용할 속성 고르기")
selected_features_kor = st.multiselect(
    "입력으로 사용할 속성을 고르세요.",
    options=all_features_kor,
    default=default_features_kor
)

if len(selected_features_kor) < 2:
    st.warning("속성을 두 개 이상 골라 주세요.")
    st.stop()

selected_features = [reverse_col_name_map[f] for f in selected_features_kor]

st.divider()

# ---------------------------
# 데이터 분리: id 순서대로 정렬 후 10명씩 묶어 앞 3명 테스트용
# ---------------------------
df_sorted = df.sort_values("id").reset_index(drop=True)
df_sorted["group_index"] = df_sorted.index // 10
df_sorted["pos_in_group"] = df_sorted.index % 10

test_mask = df_sorted["pos_in_group"] < 3
train_mask = ~test_mask

train_df = df_sorted[train_mask].copy()
test_df = df_sorted[test_mask].copy()

# bmi 결측치 처리 (선택된 경우에만, 훈련용 중앙값으로 채움)
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

X_train = train_df[selected_features]
y_train = train_df["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

st.markdown(f"- 훈련용 사람 수: **{len(train_df):,} 명**")
st.markdown(f"- 테스트용 사람 수: **{len(test_df):,} 명**")

st.divider()

# ---------------------------
# 모델 학습
# ---------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree_model.fit(X_train, y_train)

dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# 정확도 계산
def get_accuracies(model):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model)
tree_train_acc, tree_test_acc = get_accuracies(tree_model)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model)

st.subheader("② 모델 정확도 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="로지스틱 회귀(확률로 답하는 모델)", value=f"{log_test_acc:.3f}")
    st.caption(f"훈련 정확도: {log_train_acc:.3f}  ·  테스트 정확도: {log_test_acc:.3f}")

with col2:
    st.metric(label="의사결정트리(질문으로 답하는 모델)", value=f"{tree_test_acc:.3f}")
    st.caption(f"훈련 정확도: {tree_train_acc:.3f}  ·  테스트 정확도: {tree_test_acc:.3f}")

with col3:
    st.metric(label="많은 쪽으로만 답하는 모델", value=f"{dummy_test_acc:.3f}")
    st.caption(f"훈련 정확도: {dummy_train_acc:.3f}  ·  테스트 정확도: {dummy_test_acc:.3f}")

st.divider()

# ---------------------------
# 산점도 + 결정 경계
# ---------------------------
st.subheader("③ 두 속성으로 살펴보는 산점도와 경계선")

col_x_kor, col_y_kor = st.columns(2)

with col_x_kor:
    x_feature_kor = st.selectbox("가로축으로 쓸 속성", options=selected_features_kor, index=0)

with col_y_kor:
    remaining = [f for f in selected_features_kor if f != x_feature_kor]
    y_feature_kor = st.selectbox("세로축으로 쓸 속성", options=remaining, index=0)

x_feature = reverse_col_name_map[x_feature_kor]
y_feature = reverse_col_name_map[y_feature_kor]

other_features = [f for f in selected_features if f not in [x_feature, y_feature]]

# 다른 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {}
for f in other_features:
    fixed_values[f] = X_test[f].median()

if fixed_values:
    fixed_text = ", ".join([f"{col_name_map[k]} = {v:.2f}" for k, v in fixed_values.items()])
    st.markdown(f"**다른 속성은 다음 값으로 고정하여 계산했습니다:** {fixed_text}")
else:
    st.markdown("**고정할 다른 속성이 없습니다. (선택한 속성이 두 개뿐입니다)**")

# 산점도 그리기
test_df_plot = test_df.copy()
test_df_plot["실제 뇌졸중"] = test_df_plot["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

fig = go.Figure()

for label, color in [("뇌졸중 없음", "blue"), ("뇌졸중 있음", "red")]:
    subset = test_df_plot[test_df_plot["실제 뇌졸중"] == label]
    fig.add_trace(go.Scatter(
        x=subset[x_feature],
        y=subset[y_feature],
        mode="markers",
        name=label,
        marker=dict(color=color, size=7, opacity=0.6)
    ))

# 로지스틱 회귀 0.5 경계선 계산
# 모델: w0*x0 + w1*x1 + ... + b = 0 (로짓 = 0 이 확률 0.5)
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]

feature_index = {f: i for i, f in enumerate(selected_features)}
x_idx = feature_index[x_feature]
y_idx = feature_index[y_feature]

# 다른 속성들의 기여도(고정값 * 계수) 합
other_contrib = sum(
    coef[feature_index[f]] * fixed_values[f] for f in other_features
) if other_features else 0

x_min, x_max = X_test[x_feature].min(), X_test[x_feature].max()

w_x = coef[x_idx]
w_y = coef[y_idx]

line_out_of_range = False

if abs(w_y) > 1e-10:
    # y = -(intercept + other_contrib + w_x * x) / w_y
    x_range = np.linspace(x_min, x_max, 100)
    y_range = -(intercept + other_contrib + w_x * x_range) / w_y

    y_min, y_max = X_test[y_feature].min(), X_test[y_feature].max()

    # 경계선이 그림 범위 안에 있는지 확인
    in_range_mask = (y_range >= y_min) & (y_range <= y_max)

    if in_range_mask.sum() == 0:
        line_out_of_range = True
    else:
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode="lines",
            name="경계선(확률 0.5)",
            line=dict(color="black", dash="dash")
        ))
else:
    line_out_of_range = True

if line_out_of_range:
    st.markdown("**경계선(확률 0.5)은 이 그림의 범위 밖에 있어 표시되지 않았습니다.**")

# 의사결정트리 결정 영역을 옅은 색으로 표시 (다른 속성은 고정값 사용)
x_grid = np.linspace(X_test[x_feature].min(), X_test[x_feature].max(), 80)
y_grid = np.linspace(X_test[y_feature].min(), X_test[y_feature].max(), 80)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_df = pd.DataFrame({x_feature: xx.ravel(), y_feature: yy.ravel()})
for f in other_features:
    grid_df[f] = fixed_values[f]

grid_df = grid_df[selected_features]  # 학습 시 순서와 동일하게

tree_pred_grid = tree_model.predict(grid_df).reshape(xx.shape)

fig.add_trace(go.Contour(
    x=x_grid,
    y=y_grid,
    z=tree_pred_grid,
    showscale=False,
    opacity=0.2,
    colorscale=[[0, "blue"], [1, "red"]],
    contours=dict(coloring="heatmap"),
    name="트리 결정영역",
    hoverinfo="skip"
))

fig.update_layout(
    title=f"{
