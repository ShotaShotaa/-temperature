#!/usr/bin/env -S python -m streamlit run

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet

st.set_page_config(layout='wide')

# データ読み込み関数
@st.cache_data
def get_tables():
    df = pd.read_csv("data/temperature.csv", encoding="shift_jis", header=2, skiprows=[3, 4], sep=None, engine='python')
    df = df.iloc[:, [0] + list(range(1, df.shape[1], 3))]
    df = df.rename(columns={"年月": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["年"] = df["date"].dt.year
    df["月"] = df["date"].dt.month
    df = df.drop(columns=["date"])
    columns = ["年", "月"] + [col for col in df.columns if col not in ["年", "月"]]
    return df[columns]

pop = get_tables()
pref_names = [col for col in pop.columns if col not in ["年", "月"]]
TEMPERATURE = 'https://www.data.jma.go.jp/risk/obsdl/index.php'

cols3 = st.columns(2)
cols3[0].markdown('<h2 style="margin-bottom: 0;">月別平均気温（1900～2024年）</h2>', unsafe_allow_html=True)
cols3[1].link_button(':material/link:気温データ', TEMPERATURE, help=f'気象庁: {TEMPERATURE}')

years = sorted(pop["年"].unique())

mode = st.radio("表示モードを選択", ["全データ", "特定の月・年平均", "特定の年"], horizontal=True)

if mode == "全データ":
    start_year, end_year = st.select_slider(
        "表示する期間（年）を選択",
        options=years,
        value=(years[0], years[-1])
    )
    year_filtered = pop[pop["年"].between(start_year, end_year)]

    table_tab, graph_tab = st.tabs(['表', 'グラフ'])
    with table_tab:
        st.subheader("気温データ")
        st.dataframe(year_filtered, height=500, hide_index=True)

    with graph_tab:
        st.subheader("気温推移")
        prefs = st.multiselect("表示する地点を選択してください", options=pref_names)
        selected_columns = prefs if prefs else pref_names

        fig = go.Figure()
        color_palette = px.colors.qualitative.Set1
        for idx, col in enumerate(selected_columns):
            color = color_palette[idx % len(color_palette)]
            date_strings = year_filtered['年'].astype(str) + '-' + year_filtered['月'].astype(str) + '-01'
            x_dates = pd.to_datetime(date_strings)
            y_values = year_filtered[col].values

            fig.add_trace(go.Scatter(
                x=x_dates,
                y=y_values,
                mode='markers',
                name=col,
                marker=dict(color=color)
            ))

        fig.update_xaxes(title="年月")
        fig.update_yaxes(title="気温（℃）")
        st.plotly_chart(fig, use_container_width=True)

elif mode == "特定の月・年平均":
    selected_month = st.selectbox("表示する月（1～12、年平均）", options=[*range(1, 13), 13], format_func=lambda x: f"{x}月" if x != 13 else "年平均")
    prefs = st.multiselect("表示する地点を選択してください", options=pref_names)

    st.markdown("学習期間（予測モデルの学習用）")
    learn_start_year, learn_end_year = st.select_slider(
        "学習する期間（年）を選択",
        options=years,
        value=(1900, 2004),
        key="learn"
    )

    start_year, end_year = st.select_slider(
        "表示する期間（年）を選択",
        options=years,
        value=(years[0], years[-1])
    )
    year_filtered = pop[pop["年"].between(start_year, end_year)]

    table_tab, graph_tab = st.tabs(['表', 'グラフ'])

    with table_tab:
        st.subheader("気温データ")
        for pref in (prefs if prefs else pref_names):
            if selected_month != 13:
                df_pref = year_filtered[year_filtered["月"] == selected_month][["年", pref]]
            else:
                df_pref = year_filtered.groupby("年")[pref].mean().reset_index()

            df_pref = df_pref.rename(columns={pref: "気温"})
            df_pref = df_pref[df_pref["年"].between(learn_start_year, end_year)]
            df_pref["学・検"] = np.where(df_pref["年"] <= learn_end_year, "学習", "検証")

            # 線形回帰
            learn_df = df_pref[df_pref["学・検"] == "学習"]
            df_pref["予測（線形）"] = np.nan
            df_pref["予測（２次）"] = np.nan
            df_pref["予測（Prophet）"] = np.nan
            if len(learn_df) >= 2:
                # 線形
                coef1 = np.polyfit(learn_df["年"], learn_df["気温"], 1)
                df_pref["予測（線形）"] = np.polyval(coef1, df_pref["年"])
                # 2次
                coef2 = np.polyfit(learn_df["年"], learn_df["気温"], 2)
                df_pref["予測（２次）"] = np.polyval(coef2, df_pref["年"])
                # Prophet
                prophet_df = learn_df[["年", "気温"]].rename(columns={"年": "ds", "気温": "y"})
                prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
                model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
                model.fit(prophet_df)
                future = pd.DataFrame({"ds": pd.to_datetime(df_pref["年"], format="%Y")})
                forecast = model.predict(future)
                df_pref["予測（Prophet）"] = forecast["yhat"].values

            # 誤差
            df_pref["誤差（線形）"] = (df_pref["予測（線形）"] - df_pref["気温"]).abs()
            df_pref["誤差（２次）"] = (df_pref["予測（２次）"] - df_pref["気温"]).abs()
            df_pref["誤差（Prophet）"] = (df_pref["予測（Prophet）"] - df_pref["気温"]).abs()

            df_disp = df_pref[["年", "気温", "学・検", "予測（線形）", "誤差（線形）",
                            "予測（２次）", "誤差（２次）", "予測（Prophet）", "誤差（Prophet）"]]
            st.markdown(f"#### {pref}")
            st.dataframe(df_disp, hide_index=True)

    with graph_tab:
        st.subheader("気温推移（実測・予測・誤差）")

        fig = go.Figure()

        for pref in (prefs if prefs else pref_names):
            # データの準備
            if selected_month != 13:
                df_pref = year_filtered[year_filtered["月"] == selected_month][["年", pref]]
            else:
                df_pref = year_filtered.groupby("年")[pref].mean().reset_index()

            df_pref = df_pref.rename(columns={pref: "気温"})
            df_pref = df_pref[df_pref["年"].between(learn_start_year, end_year)]
            df_pref["学・検"] = np.where(df_pref["年"] <= learn_end_year, "学習", "検証")

            learn_df = df_pref[df_pref["学・検"] == "学習"]

            df_pref["予測（線形）"] = np.nan
            df_pref["予測（２次）"] = np.nan
            df_pref["予測（Prophet）"] = np.nan

            if len(learn_df) >= 2:
                # 線形回帰
                coef1 = np.polyfit(learn_df["年"], learn_df["気温"], 1)
                df_pref["予測（線形）"] = np.polyval(coef1, df_pref["年"])

                # 2次回帰
                coef2 = np.polyfit(learn_df["年"], learn_df["気温"], 2)
                df_pref["予測（２次）"] = np.polyval(coef2, df_pref["年"])

                # Prophet
                prophet_df = learn_df[["年", "気温"]].rename(columns={"年": "ds", "気温": "y"})
                prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], format="%Y")
                model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
                model.fit(prophet_df)
                future = pd.DataFrame({"ds": pd.to_datetime(df_pref["年"], format="%Y")})
                forecast = model.predict(future)
                df_pref["予測（Prophet）"] = forecast["yhat"].values

            # 実測データ
            fig.add_trace(go.Scatter(x=df_pref["年"], y=df_pref["気温"], mode="lines+markers",
                                     name=f"{pref}：実測"))

            # 線形回帰予測
            fig.add_trace(go.Scatter(x=df_pref["年"], y=df_pref["予測（線形）"], mode="lines",
                                     name=f"{pref}：線形回帰予測"))

            # 2次回帰予測
            fig.add_trace(go.Scatter(x=df_pref["年"], y=df_pref["予測（２次）"], mode="lines",
                                     name=f"{pref}：2次回帰予測"))

            # Prophet予測
            fig.add_trace(go.Scatter(x=df_pref["年"], y=df_pref["予測（Prophet）"], mode="lines",
                                     name=f"{pref}：Prophet予測"))

        fig.update_layout(title="各地点の気温推移と予測比較",
                          xaxis_title="年", yaxis_title="気温（℃）",
                          height=700)
        st.plotly_chart(fig, use_container_width=True)

elif mode == "特定の年":
    selected_year = st.selectbox("表示する年", options=years, format_func=lambda x: f"{x}年")
    st.write("※このモードはまだ未実装です")
