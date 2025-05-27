#!/usr/bin/env -S python -m streamlit run

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(layout='wide')

@st.cache_data
def get_tables():
    df = pd.read_csv("data/temperature.csv", encoding="shift_jis", header=2, skiprows=[3, 4], sep=None, engine='python')
    df = df.iloc[:, [0] + list(range(1, df.shape[1], 3))]
    df = df.rename(columns={"年月": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["年月"] = df["date"].dt.strftime("%Y/%m")
    df = df.set_index("年月")
    df = df.drop(columns=["date"])
    return df

pop = get_tables()
pref_names = list(pop.columns)
TEMPERATURE = 'https://www.data.jma.go.jp/risk/obsdl/index.php'

cols3 = st.columns(2)
cols3[0].markdown('<h2 style="margin-bottom: 0;">平均気温（1900～2024年）</h2>', unsafe_allow_html=True)
cols3[1].link_button(':material/link:気温データ', TEMPERATURE, help=f'気象庁: {TEMPERATURE}')

years = sorted({int(index[:4]) for index in pop.index})
months = sorted({int(index[-2:]) for index in pop.index})

mode = st.radio("モード選択", options=["月別", "年平均"], horizontal=True)

selected_pref = st.selectbox("地点を選択", options=pref_names)

if mode == "月別":
    selected_month = st.selectbox("月を選択", options=months, format_func=lambda x: f"{x}月")
else:
    selected_month = None

st.subheader("学習期間と表示期間の選択")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 学習期間（予測モデルの学習用）")
    learn_start_year, learn_end_year = st.select_slider(
        "学習する期間（年）を選択",
        options=years,
        value=(1970, 2020),
        key="learn"
    )

with col2:
    st.markdown("### 表示期間（予測も含めた表示）")
    display_start_year, display_end_year = st.select_slider(
        "表示する期間（年）を選択",
        options=years + [years[-1] + i for i in range(1, 21)],
        value=(1970, 2040),
        key="display"
    )

# データ準備
if mode == "月別":
    filtered_data = pop[pop.index.str.endswith(f"/{selected_month:02d}")][selected_pref]
else:
    filtered_data = pop[selected_pref].groupby(pop.index.str[:4]).mean()
    filtered_data.index = pd.to_datetime(filtered_data.index + "-01")

filtered_data = filtered_data.dropna()
if mode == "月別":
    filtered_data.index = pd.to_datetime(filtered_data.index + "-01")

# 学習データ
learn_mask = (filtered_data.index.year >= learn_start_year) & (filtered_data.index.year <= learn_end_year)
learn_data = filtered_data[learn_mask]

# 予測データ期間
forecast_years = np.arange(display_start_year, display_end_year + 1)
forecast_dates = pd.to_datetime([f"{y}-{selected_month:02d}-01" if mode == "月別" else f"{y}-01-01" for y in forecast_years])

fig = go.Figure()

# Prophet
if learn_data.shape[0] >= 2:
    df_prophet = pd.DataFrame({
        'ds': learn_data.index,
        'y': learn_data.values
    })

    prophet_model = Prophet()
    prophet_model.fit(df_prophet)

    future = pd.DataFrame({'ds': forecast_dates})
    forecast = prophet_model.predict(future)

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name=f"{selected_pref}（Prophet予測）",
        line=dict(dash='dash', color='blue')
    ))

# 線形回帰
lr_model = LinearRegression()
lr_model.fit(learn_data.index.year.values.reshape(-1, 1), learn_data.values)
y_pred_lr = lr_model.predict(forecast_years.reshape(-1, 1))

fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=y_pred_lr,
    mode='lines',
    name=f"{selected_pref}（線形回帰）",
    line=dict(dash='dot', color='green')
))

# 2次多項式回帰
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(learn_data.index.year.values.reshape(-1, 1))
lr_poly = LinearRegression()
lr_poly.fit(X_poly, learn_data.values)
X_forecast_poly = poly.transform(forecast_years.reshape(-1, 1))
y_pred_poly = lr_poly.predict(X_forecast_poly)

fig.add_trace(go.Scatter(
    x=forecast_dates,
    y=y_pred_poly,
    mode='lines',
    name=f"{selected_pref}（2次回帰）",
    line=dict(dash='dashdot', color='orange')
))

# 実測データ
display_mask = (filtered_data.index.year >= display_start_year) & (filtered_data.index.year <= display_end_year)
display_df = filtered_data[display_mask]

fig.add_trace(go.Scatter(
    x=display_df.index,
    y=display_df.values,
    mode='markers',
    name=f"{selected_pref}（実測）",
    marker=dict(color='black')
))

fig.update_layout(
    title=f"{selected_pref} {f'{selected_month}月' if mode == '月別' else '年平均'}の気温予測（学習期間: {learn_start_year}-{learn_end_year}、表示期間: {display_start_year}-{display_end_year}）",
    xaxis_title="年月",
    yaxis_title="気温（℃）"
)

st.plotly_chart(fig, use_container_width=True)
