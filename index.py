#!/usr/bin/env -S python -m streamlit run

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout='wide')

@st.cache_data
def get_tables():
    df = pd.read_csv("data/temperature.csv", encoding="shift_jis", header=2, skiprows=[3,4], sep=None, engine='python')
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
cols3[0].markdown('<h2 style="margin-bottom: 0;">月別平均気温（1900～2024年）</h2>', unsafe_allow_html=True)
cols3[1].link_button(':material/link:気温データ', TEMPERATURE, help=f'気象庁: {TEMPERATURE}')

years = sorted({int(index[:4]) for index in pop.index})

# 学習期間（予測モデルの学習に使う期間）
learn_start_year, learn_end_year = st.select_slider(
    "予測に使う学習期間（年）を選択",
    options=years,
    value=(years[0], years[-1])
)

# 表示期間（予測値を表示する期間）
display_start_year, display_end_year = st.select_slider(
    "表示する期間（年）を選択",
    options=years,
    value=(years[0], years[-1])
)

year_series = pd.Series(pop.index.str[:4].astype(int), index=pop.index)

# 学習期間データ（月別平均気温）
df_month = pop.copy()
df_month['year'] = df_month.index.str[:4].astype(int)
learn_data = df_month[(df_month['year'] >= learn_start_year) & (df_month['year'] <= learn_end_year)]
display_data = df_month[(df_month['year'] >= display_start_year) & (df_month['year'] <= display_end_year)]

table_tab, graph_tab = st.tabs(['表', 'グラフ'])

with table_tab:
    st.subheader("気温データ（月別）")
    st.dataframe(display_data.drop(columns=['year']), height=500)

with graph_tab:
    st.subheader("平均気温推移（月別）")
    prefs = st.multiselect("表示する地点を選択してください", options=pref_names, help="未選択時はすべての地点が選択されます")
    selected_columns = prefs if prefs else pref_names

    years_forecast = st.number_input("未来予測する年数", min_value=1, max_value=20, value=20, step=1)

    fig = go.Figure()
    color_palette = px.colors.qualitative.Set1

    for idx, col in enumerate(selected_columns):
        color = color_palette[idx % len(color_palette)]

        # 学習期間のデータを準備（Prophet用）
        prophet_df = learn_data[[col]].dropna().reset_index()
        if prophet_df.empty:
            continue
        prophet_df = prophet_df.rename(columns={"年月": "ds", col: "y"})
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'] + "-01")

        # Prophetモデル
        prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        prophet_model.fit(prophet_df)

        # 予測期間（学習期間＋未来）
        last_date = prophet_df['ds'].max()
        future_periods = years_forecast * 12  # 月単位
        future = prophet_model.make_future_dataframe(periods=future_periods, freq='MS')
        forecast = prophet_model.predict(future)

        # Prophetの予測線を描画（未来含む）
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name=f"{col}のProphet予測",
            line=dict(dash='dash', color=color)
        ))

        # 実測値（学習期間）
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            mode='markers',
            name=f"{col} 実測値",
            marker=dict(color=color)
        ))

        # --- 線形回帰 ---
        # 年と月を小数年に変換 (例: 2020年3月 → 2020 + 2/12)
        learn_year_float = prophet_df['ds'].dt.year + (prophet_df['ds'].dt.month - 1)/12
        lr_model = LinearRegression()
        lr_model.fit(learn_year_float.values.reshape(-1, 1), prophet_df['y'].values)

        # 未来予測用の年月（学習期間＋未来）
        future_years_float = pd.date_range(start=learn_year_float.iloc[0], periods=len(learn_year_float) + future_periods, freq='MS').to_series().apply(lambda d: d.year + (d.month-1)/12)
        # 予測値
        lr_pred = lr_model.predict(future_years_float.values.reshape(-1,1))

        # プロット（未来含む）
        future_dates = pd.date_range(start=prophet_df['ds'].iloc[0], periods=len(learn_year_float)+future_periods, freq='MS')
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lr_pred,
            mode='lines',
            name=f"{col}の線形回帰予測",
            line=dict(dash='dot', color=color)
        ))

        # --- 2次多項式回帰 ---
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(learn_year_float.values.reshape(-1, 1))
        lr_poly = LinearRegression()
        lr_poly.fit(X_poly, prophet_df['y'].values)

        X_future_poly = poly.transform(future_years_float.values.reshape(-1,1))
        poly_pred = lr_poly.predict(X_future_poly)

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=poly_pred,
            mode='lines',
            name=f"{col}の2次多項式回帰予測",
            line=dict(dash='dashdot', color=color)
        ))

        # --- 評価 ---
        # 学習期間の実測値と予測値（学習期間分のみ）
        y_true = prophet_df['y'].values
        y_pred_prophet = forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'yhat'].values
        y_pred_lr = lr_model.predict(learn_year_float.values.reshape(-1, 1))
        y_pred_poly = lr_poly.predict(X_poly)

        def evaluate(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # squared=Falseの代わりにnp.sqrtを使う
            r2 = r2_score(y_true, y_pred)
            return mae, rmse, r2

        prophet_scores = evaluate(y_true, y_pred_prophet)
        lr_scores = evaluate(y_true, y_pred_lr)
        poly_scores = evaluate(y_true, y_pred_poly)

        # 評価表示を1地点ずつ出す
        st.markdown(f"### {col} の予測精度（学習期間）")
        st.markdown(f"""
        | モデル         | MAE    | RMSE   | R²    |
        |----------------|--------|--------|-------|
        | Prophet        | {prophet_scores[0]:.3f} | {prophet_scores[1]:.3f} | {prophet_scores[2]:.3f} |
        | 線形回帰       | {lr_scores[0]:.3f} | {lr_scores[1]:.3f} | {lr_scores[2]:.3f} |
        | 2次多項式回帰  | {poly_scores[0]:.3f} | {poly_scores[1]:.3f} | {poly_scores[2]:.3f} |
        """)

    fig.update_xaxes(title="年月")
    fig.update_yaxes(title="気温（℃）")
    st.plotly_chart(fig, use_container_width=True)
