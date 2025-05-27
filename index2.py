#!/usr/bin/env -S python -m streamlit run

import streamlit as st
import pandas as pd
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
start_year, end_year = st.select_slider(
    "表示する期間（年）を選択",
    options=years,
    value=(years[0], years[-1])
)

year_series = pd.Series(pop.index.str[:4].astype(int), index=pop.index)
year_filtered = pop[year_series.between(start_year, end_year)]

table_tab, graph_tab = st.tabs(['表', 'グラフ'])

with table_tab:
    st.subheader("気温データ")
    st.dataframe(year_filtered, height=500)

with graph_tab:
    st.subheader("平均気温推移")
    mode = st.radio("表示モードを選択", ["月別平均気温の推移", "特定年の比較"], horizontal=True)
    prefs = st.multiselect("表示する地点を選択してください", options=pref_names, help="未選択時はすべての地点が選択されます")
    selected_columns = prefs if prefs else pref_names

    fig = go.Figure()
    color_palette = px.colors.qualitative.Set1

    if mode == "月別平均気温の推移":
        months = sorted({int(index[-2:]) for index in pop.index})
        months.append(13)
        selected_month = st.selectbox(
            "表示する月（1～12、年平均）",
            options=months,
            format_func=lambda x: f"{x}月" if x != 13 else "年平均"
        )

        years_forecast = st.number_input("未来予測する年数", min_value=1, max_value=20, value=20, step=1)

        if selected_month == 13:
            df_month = year_filtered.copy()
            df_month["year"] = df_month.index.str[:4]
            df_annual_avg = df_month.groupby("year").mean()
            df_annual_avg = df_annual_avg[selected_columns].dropna(how='any')
            df_annual_avg.index = pd.to_datetime(df_annual_avg.index + "-01-01")

            for idx, col in enumerate(selected_columns):
                color = color_palette[idx % len(color_palette)]

                # ✅ 安定する安全な整形
                df_prophet = df_annual_avg[[col]].reset_index()
                df_prophet.columns = ['ds', 'y']
                df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

                prophet_model = Prophet()
                prophet_model.fit(df_prophet)

                future = prophet_model.make_future_dataframe(periods=years_forecast, freq='YS')
                forecast = prophet_model.predict(future)

                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name=f"{col}のProphet予測",
                    line=dict(dash='dash', color=color)
                ))

                fig.add_trace(go.Scatter(
                    x=df_prophet['ds'],
                    y=df_prophet['y'],
                    mode='markers',
                    name=f"{col}",
                    marker=dict(color=color)
                ))

            fig.update_xaxes(title="年")
            fig.update_yaxes(title="気温（℃）")

        else:
            filtered_data = year_filtered[year_filtered.index.str.endswith(f"/{selected_month:02d}")]

            for idx, col in enumerate(selected_columns):
                color = color_palette[idx % len(color_palette)]
                x_dates = pd.to_datetime(filtered_data.index + "-01")
                y_values = filtered_data[col].values

                fig.add_trace(go.Scatter(
                    x=x_dates,
                    y=y_values,
                    mode='markers',
                    name=col,
                    marker=dict(color=color)
                ))

                df_nonan = filtered_data[[col]].dropna()
                df_nonan.index = pd.to_datetime(df_nonan.index + "-01")

                if df_nonan.shape[0] > 1:
                    df_prophet = df_nonan.reset_index().rename(columns={col: 'y', '年月': 'ds'})
                    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

                    prophet_model = Prophet()
                    prophet_model.fit(df_prophet)

                    future = prophet_model.make_future_dataframe(periods=years_forecast * 12, freq='MS')
                    forecast = forecast = prophet_model.predict(future)
                    forecast = forecast[forecast['ds'].dt.month == selected_month]

                    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat'],
                        mode='lines',
                        name=f"{col}のProphet予測",
                        line=dict(dash='dash', color=color)
                    ))

            fig.update_xaxes(title="年月")
            fig.update_yaxes(title="気温（℃）")

    elif mode == "特定年の比較":
        selected_years = st.multiselect("比較する年を選択", options=years, default=[years[-1]])
        selected_data = pop[pop.index.str[:4].astype(int).isin(selected_years)][selected_columns]

        df_plot = selected_data.copy()
        df_plot["year"] = df_plot.index.str[:4]
        df_plot["month"] = df_plot.index.str[-2:].astype(int)

        for i, col in enumerate(selected_columns):
            color_base = px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            for j, year in enumerate(selected_years):
                year_df = df_plot[df_plot["year"] == str(year)]
                fig.add_trace(go.Scatter(
                    x=year_df["month"],
                    y=year_df[col],
                    mode='lines+markers',
                    name=f"{col}（{year}年）",
                    marker=dict(color=color_base),
                    line=dict(dash=["solid", "dot", "dash", "dashdot"][j % 4])
                ))

        fig.update_xaxes(title="月", dtick=1)
        fig.update_yaxes(title="気温（℃）")

    st.plotly_chart(fig, use_container_width=True)
