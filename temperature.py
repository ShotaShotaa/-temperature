#!/usr/bin/env -S python -m streamlit run

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import warnings

# ProphetのFutureWarningを抑制
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(layout='wide')

# --- 定数定義 ---
ANNUAL_AVERAGE_MONTH_PROXY = 13
DEFAULT_LEARN_START_YEAR = 1900
DEFAULT_LEARN_END_YEAR = 2024
MONTHS_IN_YEAR = 12
FUTURE_PREDICTION_YEARS = 20
LINE_STYLES_FOR_YEARS = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
MARKER_SYMBOLS_FOR_YEARS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star', 'hexagram', 'hourglass']


# --- データ読み込み関数 ---
@st.cache_data  # CSVの読み込み結果をキャッシュ
def get_temperature_data(file_path="data/temperature.csv"):
    try:
        df = pd.read_csv(file_path, encoding="shift_jis", header=2, skiprows=[3, 4], sep=None, engine='python')
        df = df.iloc[:, [0] + list(range(1, df.shape[1], 3))]
        df = df.rename(columns={"年月": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df["年"] = df["date"].dt.year
        df["月"] = df["date"].dt.month
        df = df.drop(columns=["date"])
        columns_ordered = ["年", "月"] + [col for col in df.columns if col not in ["年", "月"]]
        return df[columns_ordered]
    except FileNotFoundError:
        st.error(f"データファイルが見つかりません: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"データ読み込み中にエラーが発生しました: {e}")
        return pd.DataFrame()


# --- 「特定の月・年平均」モード用のデータ処理・予測関数 ---
@st.cache_data  # 関数全体の計算結果をキャッシュ
def prepare_and_predict_monthly_data(
        _base_df: pd.DataFrame,  # アンダーバーはキャッシュキーへの影響を避ける意図 (内容は変化しない想定)
        pref_name: str,
        selected_month_value: int,
        learn_start_year: int,
        learn_end_year: int,
        display_start_year: int,
        display_end_year: int
) -> pd.DataFrame:
    # base_dfはキャッシュされたpop_dataが渡されるので、ここではコピーして使う
    base_df = _base_df.copy()

    if pref_name not in base_df.columns:
        print(f"地点 '{pref_name}' のデータが元のデータフレームに存在しません。")
        return pd.DataFrame()

    all_years_in_display_period = pd.DataFrame({'年': range(display_start_year, display_end_year + 1)})

    if selected_month_value != ANNUAL_AVERAGE_MONTH_PROXY:
        monthly_data = base_df[
            (base_df["月"] == selected_month_value) &
            (base_df["年"] >= display_start_year) &  # 修正: display_start_yearからでOK
            (base_df["年"] <= display_end_year)  # display_end_yearまで (未来の年はデータがない)
            ][["年", pref_name]].copy()
        df_processed = pd.merge(all_years_in_display_period, monthly_data, on="年", how="left")
    else:
        yearly_data_counts = base_df.groupby('年')[pref_name].count().rename('data_count')
        yearly_means_data = base_df.groupby('年')[pref_name].mean().rename(pref_name)
        df_annual_avg_raw = pd.merge(yearly_means_data, yearly_data_counts, on='年', how='left').reset_index()
        df_annual_avg_raw[pref_name] = df_annual_avg_raw.apply(
            lambda row: row[pref_name] if row['data_count'] == MONTHS_IN_YEAR else np.nan, axis=1
        )
        df_processed = pd.merge(all_years_in_display_period, df_annual_avg_raw[['年', pref_name]], on="年", how="left")

    df_processed = df_processed.rename(columns={pref_name: "気温"})
    df_display_period = df_processed.copy()

    if df_display_period.empty:
        return pd.DataFrame()

    df_display_period["学・検"] = np.where(
        df_display_period["年"].between(learn_start_year, learn_end_year), "学習", "検証"
    )

    learn_df = df_display_period[
        (df_display_period["学・検"] == "学習") &
        (df_display_period["年"] >= learn_start_year) &
        (df_display_period["年"] <= learn_end_year) &
        (df_display_period["気温"].notna())
        ].copy()

    df_display_period["予測（線形）"] = np.nan
    df_display_period["予測（２次）"] = np.nan
    df_display_period["予測（Prophet）"] = np.nan

    if len(learn_df) >= 2:
        coef1 = np.polyfit(learn_df["年"], learn_df["気温"], 1)
        df_display_period["予測（線形）"] = np.polyval(coef1, df_display_period["年"])
        coef2 = np.polyfit(learn_df["年"], learn_df["気温"], 2)
        df_display_period["予測（２次）"] = np.polyval(coef2, df_display_period["年"])

        prophet_learn_df = learn_df[["年", "気温"]].rename(columns={"年": "ds", "気温": "y"})
        prophet_learn_df["ds"] = pd.to_datetime(prophet_learn_df["ds"], format="%Y")

        # Prophetモデルの学習と予測（この関数内では毎回実行、関数自体がキャッシュされる）
        model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
        try:
            model.fit(prophet_learn_df)
            future_dates = pd.DataFrame({"ds": pd.to_datetime(df_display_period["年"], format="%Y")})
            forecast = model.predict(future_dates)
            df_display_period["予測（Prophet）"] = forecast["yhat"].values
        except Exception as e:
            print(f"Prophetモデルエラー ({pref_name} - 月:{selected_month_value}): {e}")

    for model_type in ["線形", "２次", "Prophet"]:
        pred_col = f"予測（{model_type}）"
        error_col = f"誤差（{model_type}）"
        if pred_col in df_display_period.columns:
            df_display_period[error_col] = np.where(
                df_display_period["気温"].isnull(), np.nan,
                (df_display_period[pred_col] - df_display_period["気温"]).abs()
            )
        else:
            df_display_period[error_col] = np.nan
    return df_display_period


# --- Prophetモデル学習関数（月別気温予測用、学習済みモデルをキャッシュ）---
@st.cache_resource  # 学習済みモデルオブジェクトをキャッシュ
def get_trained_monthly_prophet_model(_historical_data, pref_name, _last_actual_data_year):
    """指定された地点の過去データでProphetモデルを学習し、学習済みモデルを返す。"""
    # _historical_dataはキャッシュされたpop_dataが渡される想定
    df_train = _historical_data[_historical_data['年'] <= _last_actual_data_year][['年', '月', pref_name]].copy()

    if df_train.empty or df_train[pref_name].isnull().all():
        print(f"学習データなし (get_trained_monthly_prophet_model): {pref_name}")
        return None

    df_train['ds'] = pd.to_datetime(df_train['年'].astype(str) + '-' + df_train['月'].astype(str) + '-15')
    df_train = df_train.rename(columns={pref_name: 'y'})
    df_train = df_train[['ds', 'y']].dropna()

    if len(df_train) < MONTHS_IN_YEAR * 2:
        print(f"学習データ不足 (get_trained_monthly_prophet_model): {pref_name} ({len(df_train)}点)")
        return None

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    try:
        model.fit(df_train)
        return model
    except Exception as e:
        print(f"Prophetモデル学習エラー (get_trained_monthly_prophet_model - {pref_name}): {e}")
        return None


# --- 特定年の月別気温取得または予測関数（予測結果をキャッシュ）---
@st.cache_data  # 予測結果のDataFrameをキャッシュ
def get_or_predict_specific_year_monthly_data(
        _historical_pop_data_for_pred,  # キャッシュキーとして使用される引数
        pref_name: str,
        target_year: int,
        _last_actual_data_year_for_pred  # キャッシュキーとして使用される引数
):
    months_df = pd.DataFrame({'月': range(1, MONTHS_IN_YEAR + 1)})

    if target_year <= _last_actual_data_year_for_pred:
        actual_df_year = _historical_pop_data_for_pred[
            _historical_pop_data_for_pred["年"] == target_year
            ][['月', pref_name]].copy()
        if actual_df_year.empty:
            return months_df.assign(気温=np.nan)
        actual_df_year = actual_df_year.rename(columns={pref_name: '気温'})
        return pd.merge(months_df, actual_df_year, on='月', how='left').sort_values(by='月')
    else:
        # 学習済みモデルを取得（この呼び出しがキャッシュされる）
        trained_model = get_trained_monthly_prophet_model(
            _historical_pop_data_for_pred, pref_name, _last_actual_data_year_for_pred
        )
        if trained_model is None:  # モデル学習に失敗した場合
            return months_df.assign(気温=np.nan)

        future_dates_ds = [pd.to_datetime(f"{target_year}-{month}-15") for month in range(1, MONTHS_IN_YEAR + 1)]
        future_df = pd.DataFrame({'ds': future_dates_ds})

        try:
            forecast = trained_model.predict(future_df)
            return pd.DataFrame({'月': range(1, MONTHS_IN_YEAR + 1), '気温': forecast['yhat'].values}).sort_values(
                by='月')
        except Exception as e:
            print(
                f"Prophetモデル予測エラー (get_or_predict_specific_year_monthly_data - {pref_name}, 年: {target_year}): {e}")
            return months_df.assign(気温=np.nan)


# --- メイン処理 ---
pop_data = get_temperature_data()  # 初回実行時またはファイル変更時に読み込み

if pop_data.empty:
    st.error("気温データの読み込みに失敗したため、アプリケーションを起動できません。")
    st.stop()

pref_names = [col for col in pop_data.columns if col not in ["年", "月"]]
if not pref_names:
    st.error("データから有効な地点情報が取得できませんでした。")
    st.stop()

TEMPERATURE_DATA_SOURCE_URL = 'https://www.data.jma.go.jp/risk/obsdl/index.php'

header_cols = st.columns(2)
header_cols[0].markdown('<h2 style="margin-bottom: 0;">月別平均気温（1900～現在）</h2>', unsafe_allow_html=True)
header_cols[1].link_button(':material/link: 気象庁 気温データ', TEMPERATURE_DATA_SOURCE_URL,
                           help=f'出典: {TEMPERATURE_DATA_SOURCE_URL}')

available_years = sorted(pop_data["年"].unique())
if not available_years:
    st.error("データから有効な年情報が取得できませんでした。")
    st.stop()
max_data_year = available_years[-1]

display_mode = st.radio(
    "表示モードを選択",
    ["全期間データ表示", "特定月・年平均の分析・予測", "特定年の月別データ"],
    horizontal=True,
    key="display_mode_radio"
)

if display_mode == "全期間データ表示":
    # (このブロックは変更なし)
    st.markdown("---")
    start_year_display, end_year_display = st.select_slider(
        "表示する期間（年）を選択",
        options=available_years,
        value=(available_years[0], available_years[-1]),
        key="all_data_year_slider"
    )
    year_filtered_all_data = pop_data[pop_data["年"].between(start_year_display, end_year_display)]

    table_tab_all, graph_tab_all = st.tabs(['📋 表データ', '📊 グラフ表示'])
    with table_tab_all:
        st.subheader(f"{start_year_display}年～{end_year_display}年の気温データ")
        st.dataframe(year_filtered_all_data, height=500, hide_index=True, use_container_width=True)

    with graph_tab_all:
        st.subheader(f"{start_year_display}年～{end_year_display}年の気温推移")
        selected_prefs_from_ui_all_data = st.multiselect(
            "表示する地点を選択してください（未選択時は全地点表示）",
            options=pref_names, key="all_data_prefs_multiselect"
        )
        target_prefs_for_all_data_graph = selected_prefs_from_ui_all_data if selected_prefs_from_ui_all_data else pref_names

        if not target_prefs_for_all_data_graph:
            st.info("表示できる地点データがありません。")
        else:
            fig_all_data = go.Figure()
            color_palette_all_data = px.colors.qualitative.Plotly
            for idx, col_name in enumerate(target_prefs_for_all_data_graph):
                plot_color = color_palette_all_data[idx % len(color_palette_all_data)]
                date_strings_all = year_filtered_all_data['年'].astype(str) + '-' + year_filtered_all_data['月'].astype(
                    str).str.zfill(2) + '-01'
                x_axis_dates = pd.to_datetime(date_strings_all)
                y_axis_values = year_filtered_all_data[col_name].values
                fig_all_data.add_trace(go.Scatter(
                    x=x_axis_dates, y=y_axis_values, mode='lines+markers', name=col_name,
                    marker=dict(color=plot_color, size=4), line=dict(color=plot_color, width=1.5)
                ))
            fig_all_data.update_layout(
                xaxis_title="年月", yaxis_title="平均気温（℃）", height=600, legend_title_text="地点"
            )
            st.plotly_chart(fig_all_data, use_container_width=True)


elif display_mode == "特定月・年平均の分析・予測":
    st.markdown("---")
    selected_month = st.selectbox(
        "分析・予測対象の月を選択（「年平均」も選択可）",
        options=[*range(1, 13), ANNUAL_AVERAGE_MONTH_PROXY],
        format_func=lambda x: f"{x}月" if x != ANNUAL_AVERAGE_MONTH_PROXY else "年平均",
        key="specific_month_selectbox"
    )
    selected_prefs_from_ui_specific_month = st.multiselect(
        "分析・予測する地点を選択してください（未選択時は全地点）",
        options=pref_names, key="specific_month_multiselect_prefs"
    )
    target_prefs_for_analysis = selected_prefs_from_ui_specific_month if selected_prefs_from_ui_specific_month else pref_names

    st.markdown("##### 予測モデルの学習と表示期間の設定")
    col_learn_period, col_display_period = st.columns(2)
    with col_learn_period:
        learn_start_year, learn_end_year = st.select_slider(
            "学習データとして使用する期間（年）", options=available_years,
            value=(DEFAULT_LEARN_START_YEAR if DEFAULT_LEARN_START_YEAR in available_years else available_years[0],
                   DEFAULT_LEARN_END_YEAR if DEFAULT_LEARN_END_YEAR in available_years else max_data_year),
            key="learn_period_slider_specific_month"
        )
    with col_display_period:
        max_prediction_display_year = max_data_year + FUTURE_PREDICTION_YEARS
        display_slider_options_years = list(range(available_years[0], max_prediction_display_year + 1))
        default_display_start_val = available_years[0]
        default_display_end_val = max_prediction_display_year
        display_start_year, display_end_year = st.select_slider(
            "グラフに表示/予測する期間（年）", options=display_slider_options_years,
            value=(default_display_start_val, default_display_end_val),
            key="display_period_slider_specific_month_future"
        )
    st.markdown("---")

    all_prefectures_processed_data = {}
    if not target_prefs_for_analysis:
        st.info("分析・予測する地点データがありません。")
    else:
        with st.spinner("予測モデルを計算中...⏳"):  # スピナーを追加
            for pref_name_analysis in target_prefs_for_analysis:
                try:
                    # キャッシュされたpop_dataを渡す
                    processed_df_for_pref = prepare_and_predict_monthly_data(
                        pop_data, pref_name_analysis, selected_month,
                        learn_start_year, learn_end_year,
                        display_start_year, display_end_year
                    )
                    all_prefectures_processed_data[pref_name_analysis] = processed_df_for_pref
                except Exception as e:
                    st.error(f"データ処理中に予期せぬエラー ({pref_name_analysis}): {e}")
                    all_prefectures_processed_data[pref_name_analysis] = pd.DataFrame()

        table_tab_specific, graph_tab_specific = st.tabs(['📈📉 予測結果（表）', '📊 予測グラフ'])
        with table_tab_specific:
            st.subheader("各地点の気温データと予測結果")
            if not all_prefectures_processed_data or all(df.empty for df in all_prefectures_processed_data.values()):
                st.info("表示できるデータがありません。地点や期間を確認してください。")
            else:
                for pref_name_table, df_result_table in all_prefectures_processed_data.items():
                    st.markdown(f"#### {pref_name_table}")
                    if df_result_table.empty:
                        st.warning("この地点のデータは処理できませんでした。")
                        continue
                    cols_to_show_in_table = ["年", "気温", "学・検", "予測（線形）", "誤差（線形）", "予測（２次）",
                                             "誤差（２次）", "予測（Prophet）", "誤差（Prophet）"]
                    existing_cols_table = [col for col in cols_to_show_in_table if col in df_result_table.columns]
                    st.dataframe(df_result_table[existing_cols_table].style.format(na_rep='-', precision=2),
                                 hide_index=True, use_container_width=True)

        with graph_tab_specific:
            st.subheader("気温推移と予測の比較グラフ")
            if not all_prefectures_processed_data or all(df.empty for df in all_prefectures_processed_data.values()):
                st.info("グラフを表示できるデータがありません。")
            else:
                fig_specific_month = go.Figure()
                color_palette_graph = px.colors.qualitative.Set1
                line_styles_graph_pred = {"線形回帰予測": "dash", "2次回帰予測": "dot", "Prophet予測": "dashdot"}
                plot_idx = 0
                for pref_name_graph, df_result_graph in all_prefectures_processed_data.items():
                    if df_result_graph.empty: continue
                    current_pref_color_graph = color_palette_graph[plot_idx % len(color_palette_graph)]
                    if "気温" in df_result_graph.columns and not df_result_graph["気温"].isnull().all():
                        fig_specific_month.add_trace(go.Scatter(
                            x=df_result_graph["年"], y=df_result_graph["気温"], mode="markers",
                            name=f"{pref_name_graph}：実測",
                            marker=dict(color=current_pref_color_graph, size=7, symbol='circle'),
                        ))


                    def add_pred_trace_to_fig(trace_suffix, col_name_pred, style_key):
                        if col_name_pred in df_result_graph.columns and not df_result_graph[
                            col_name_pred].isnull().all():
                            fig_specific_month.add_trace(go.Scatter(
                                x=df_result_graph["年"], y=df_result_graph[col_name_pred], mode="lines",
                                name=f"{pref_name_graph}：{trace_suffix}",
                                line=dict(color=current_pref_color_graph, dash=line_styles_graph_pred[style_key]),
                            ))


                    add_pred_trace_to_fig("線形回帰予測", "予測（線形）", "線形回帰予測")
                    add_pred_trace_to_fig("2次回帰予測", "予測（２次）", "2次回帰予測")
                    add_pred_trace_to_fig("Prophet予測", "予測（Prophet）", "Prophet予測")
                    plot_idx += 1
                month_name_display = f"{selected_month}月" if selected_month != ANNUAL_AVERAGE_MONTH_PROXY else "年平均"
                fig_specific_month.update_layout(title_text=f"各地点の気温推移と予測比較（{month_name_display}）",
                                                 xaxis_title="年", yaxis_title="平均気温（℃）", height=700,
                                                 legend_title_text='凡例')
                st.plotly_chart(fig_specific_month, use_container_width=True)


elif display_mode == "特定年の月別データ":
    st.markdown("---")
    max_displayable_year_monthly = max_data_year + FUTURE_PREDICTION_YEARS
    years_options_monthly = list(range(available_years[0], max_displayable_year_monthly + 1))

    selected_years_for_monthly_display = st.multiselect(
        "比較表示する年を選択（複数選択可）",
        options=years_options_monthly,
        default=[max_data_year] if max_data_year in years_options_monthly else \
            ([years_options_monthly[-1]] if years_options_monthly else []),
        key="specific_years_multiselect_future"
    )
    st.caption(f"💡 {max_data_year}年を超える年は、Prophetによる予測値で表示されます。")

    if not selected_years_for_monthly_display:
        st.info("比較表示する年を1つ以上選択してください。")
    else:
        selected_prefs_from_ui_specific_years = st.multiselect(
            f"表示する地点を選択（未選択時は全地点）",
            options=pref_names,
            key="specific_years_prefs_multiselect"
        )
        target_prefs_for_specific_years = selected_prefs_from_ui_specific_years if selected_prefs_from_ui_specific_years else pref_names

        if not target_prefs_for_specific_years:
            st.info("表示する地点を1つ以上選択してください。")
        else:
            st.subheader("選択年の月別気温比較（実測値と予測値）")

            fig_combined_years_monthly = go.Figure()
            color_palette_prefs_monthly = px.colors.qualitative.Plotly
            all_monthly_data_for_table = []

            # スピナーの表示タイミングを調整
            with st.spinner("実測値取得および未来年予測を実行中... お待ちください...  Predictions in progress... ⏳"):
                for year_idx, current_selected_year in enumerate(selected_years_for_monthly_display):
                    year_type_label = "予測" if current_selected_year > max_data_year else "実測"

                    for pref_idx, current_pref_name in enumerate(target_prefs_for_specific_years):
                        # キャッシュされたpop_dataとmax_data_yearを渡す
                        monthly_temps_df = get_or_predict_specific_year_monthly_data(
                            pop_data, current_pref_name, current_selected_year, max_data_year
                        )

                        if monthly_temps_df.empty or monthly_temps_df['気温'].isnull().all():
                            continue

                        current_pref_color = color_palette_prefs_monthly[pref_idx % len(color_palette_prefs_monthly)]
                        current_year_line_style = LINE_STYLES_FOR_YEARS[year_idx % len(LINE_STYLES_FOR_YEARS)]
                        current_year_marker_symbol = MARKER_SYMBOLS_FOR_YEARS[year_idx % len(MARKER_SYMBOLS_FOR_YEARS)]

                        fig_combined_years_monthly.add_trace(go.Scatter(
                            x=monthly_temps_df['月'],
                            y=monthly_temps_df['気温'],
                            mode='lines+markers',
                            name=f"{current_pref_name} ({current_selected_year}年 {year_type_label})",
                            line=dict(color=current_pref_color, dash=current_year_line_style),
                            marker=dict(color=current_pref_color, symbol=current_year_marker_symbol, size=6),
                        ))

                        table_df_temp = monthly_temps_df.copy()
                        table_df_temp['年'] = current_selected_year
                        table_df_temp['地点'] = current_pref_name
                        table_df_temp['種類'] = year_type_label
                        all_monthly_data_for_table.append(table_df_temp[['月', '年', '地点', '気温', '種類']])

            if not fig_combined_years_monthly.data:
                st.info("選択された条件で表示できるグラフデータがありません。")
            else:
                fig_combined_years_monthly.update_layout(
                    title_text=f"選択年の月別平均気温比較",
                    xaxis_title="月",
                    yaxis_title="平均気温（℃）",
                    xaxis=dict(tickmode='array', tickvals=list(range(1, 13)),
                               ticktext=[f"{m}月" for m in range(1, 13)]),
                    height=700,
                    legend_title_text="凡例（地点 - 年 - 種別）"
                )
                st.plotly_chart(fig_combined_years_monthly, use_container_width=True)

            if not all_monthly_data_for_table:
                st.info("選択された条件で表示できる表データがありません。")
            else:
                st.markdown("##### データ表（全選択年・地点：実測値と予測値）")
                combined_table_data_monthly = pd.concat(all_monthly_data_for_table, ignore_index=True)

                if combined_table_data_monthly.empty:
                    st.info("表データがありません。")
                else:
                    try:
                        pivot_table_monthly_combined = combined_table_data_monthly.pivot_table(
                            index='月',
                            columns=['地点', '年', '種類'],
                            values='気温'
                        ).sort_index(axis=0)

                        if isinstance(pivot_table_monthly_combined.columns, pd.MultiIndex):
                            pivot_table_monthly_combined = pivot_table_monthly_combined.sort_index(axis=1,
                                                                                                   level=['地点', '年',
                                                                                                          '種類'])

                        st.dataframe(pivot_table_monthly_combined.style.format("{:.1f}", na_rep='-'),
                                     use_container_width=True)
                    except Exception as e:
                        st.error(f"表の作成中にエラーが発生しました: {e}")
                        st.dataframe(combined_table_data_monthly)