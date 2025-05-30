#!/usr/bin/env -S python -m streamlit run

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ProphetのFutureWarningを抑制
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(layout='wide')

# --- 定数定義 ---
ANNUAL_AVERAGE_MONTH_PROXY = 13
DEFAULT_LEARN_START_YEAR = 1900
DEFAULT_LEARN_END_YEAR = 2004
MONTHS_IN_YEAR = 12
FUTURE_PREDICTION_YEARS = 20
LINE_STYLES_FOR_YEARS = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
MARKER_SYMBOLS_FOR_YEARS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star', 'hexagram', 'hourglass']
MODEL_NAMES = ["線形回帰", "2次回帰", "Prophet"]


# --- データ読み込み関数 ---
@st.cache_data
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
@st.cache_data
def prepare_and_predict_monthly_data(
        _base_df: pd.DataFrame,
        pref_name: str,
        selected_month_value: int,
        learn_start_year: int,
        learn_end_year: int,
        display_start_year: int,
        display_end_year: int
) -> tuple[pd.DataFrame, dict]:
    base_df = _base_df.copy()
    evaluation_metrics = {model: {} for model in MODEL_NAMES}

    if pref_name not in base_df.columns:
        print(f"地点 '{pref_name}' のデータが元のデータフレームに存在しません。")
        return pd.DataFrame(), evaluation_metrics

    learn_data_actual_period = base_df[base_df["年"].between(learn_start_year, learn_end_year)]
    if selected_month_value != ANNUAL_AVERAGE_MONTH_PROXY:
        learn_df_temps = learn_data_actual_period[
            learn_data_actual_period["月"] == selected_month_value
            ][["年", pref_name]].copy()
    else:
        learn_yearly_counts = learn_data_actual_period.groupby('年')[pref_name].count().rename('data_count')
        learn_yearly_means = learn_data_actual_period.groupby('年')[pref_name].mean().rename(pref_name)
        learn_df_annual_avg = pd.merge(learn_yearly_means, learn_yearly_counts, on='年', how='left').reset_index()
        learn_df_annual_avg[pref_name] = learn_df_annual_avg.apply(
            lambda row: row[pref_name] if row['data_count'] == MONTHS_IN_YEAR else np.nan, axis=1
        )
        learn_df_temps = learn_df_annual_avg[['年', pref_name]].copy()
    learn_df_temps = learn_df_temps.rename(columns={pref_name: "気温"})
    learn_df_for_fitting = learn_df_temps[learn_df_temps["気温"].notna()].copy()

    all_years_in_display_period = pd.DataFrame({'年': range(display_start_year, display_end_year + 1)})
    if selected_month_value != ANNUAL_AVERAGE_MONTH_PROXY:
        monthly_data_for_display = base_df[base_df["月"] == selected_month_value][["年", pref_name]].copy()
    else:
        display_yearly_counts = base_df.groupby('年')[pref_name].count().rename('data_count')
        display_yearly_means = base_df.groupby('年')[pref_name].mean().rename(pref_name)
        df_annual_avg_raw_display = pd.merge(display_yearly_means, display_yearly_counts, on='年',
                                             how='left').reset_index()
        df_annual_avg_raw_display[pref_name] = df_annual_avg_raw_display.apply(
            lambda row: row[pref_name] if row['data_count'] == MONTHS_IN_YEAR else np.nan, axis=1
        )
        monthly_data_for_display = df_annual_avg_raw_display[['年', pref_name]].copy()
    monthly_data_for_display = monthly_data_for_display.rename(columns={pref_name: "気温"})
    df_display_output = pd.merge(all_years_in_display_period, monthly_data_for_display, on="年", how="left")
    df_display_output["学・検"] = "検証"
    df_display_output.loc[df_display_output["年"].between(learn_start_year, learn_end_year), "学・検"] = "学習"

    pred_cols = {"線形回帰": "予測（線形）", "2次回帰": "予測（２次）", "Prophet": "予測（Prophet）"}
    for model_name in MODEL_NAMES:
        df_display_output[pred_cols[model_name]] = np.nan

    if len(learn_df_for_fitting) >= 2:
        coef1 = np.polyfit(learn_df_for_fitting["年"], learn_df_for_fitting["気温"], 1)
        df_display_output[pred_cols["線形回帰"]] = np.polyval(coef1, df_display_output["年"])
        coef2 = np.polyfit(learn_df_for_fitting["年"], learn_df_for_fitting["気温"], 2)
        df_display_output[pred_cols["2次回帰"]] = np.polyval(coef2, df_display_output["年"])
        prophet_learn_df = learn_df_for_fitting[["年", "気温"]].rename(columns={"年": "ds", "気温": "y"})
        prophet_learn_df["ds"] = pd.to_datetime(prophet_learn_df["ds"], format="%Y")
        model_prophet = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
        try:
            model_prophet.fit(prophet_learn_df)
            future_dates_prophet = pd.DataFrame({"ds": pd.to_datetime(df_display_output["年"], format="%Y")})
            forecast_prophet = model_prophet.predict(future_dates_prophet)
            df_display_output[pred_cols["Prophet"]] = forecast_prophet["yhat"].values
        except Exception as e:
            print(f"Prophetモデルエラー ({pref_name} - 月:{selected_month_value}): {e}")
            df_display_output[pred_cols["Prophet"]] = np.nan

    for model_name in MODEL_NAMES:
        pred_col_name = pred_cols[model_name]
        error_col_name = f"誤差（{model_name.split('（')[0]}）"
        df_display_output[error_col_name] = np.where(
            df_display_output["気温"].isnull() | df_display_output[pred_col_name].isnull(),
            np.nan, (df_display_output[pred_col_name] - df_display_output["気温"]).abs()
        )
        learn_predictions = df_display_output[df_display_output['年'].isin(learn_df_for_fitting['年'])][
            ['年', pred_col_name]]
        learn_eval_df = pd.merge(learn_df_for_fitting, learn_predictions, on='年', how='inner')
        comp_learn = learn_eval_df[['気温', pred_col_name]].dropna()
        if not comp_learn.empty and len(comp_learn) > 0:
            evaluation_metrics[model_name]["learn_mae"] = mean_absolute_error(comp_learn['気温'],
                                                                              comp_learn[pred_col_name])
            evaluation_metrics[model_name]["learn_rmse"] = np.sqrt(
                mean_squared_error(comp_learn['気温'], comp_learn[pred_col_name]))
        else:
            evaluation_metrics[model_name]["learn_mae"] = np.nan
            evaluation_metrics[model_name]["learn_rmse"] = np.nan
        validation_data_eval = df_display_output[df_display_output["学・検"] == "検証"]
        comp_val = validation_data_eval[['気温', pred_col_name]].dropna()
        if not comp_val.empty and len(comp_val) > 0:
            evaluation_metrics[model_name]["validation_mae"] = mean_absolute_error(comp_val['気温'],
                                                                                   comp_val[pred_col_name])
            evaluation_metrics[model_name]["validation_rmse"] = np.sqrt(
                mean_squared_error(comp_val['気温'], comp_val[pred_col_name]))
        else:
            evaluation_metrics[model_name]["validation_mae"] = np.nan
            evaluation_metrics[model_name]["validation_rmse"] = np.nan
    return df_display_output, evaluation_metrics


# --- Prophetモデル学習関数（月別気温予測用、学習済みモデルをキャッシュ）---
@st.cache_resource
def get_trained_monthly_prophet_model(_historical_data, pref_name, _last_actual_data_year):
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
@st.cache_data
def get_or_predict_specific_year_monthly_data(
        _historical_pop_data_for_pred, pref_name: str, target_year: int, _last_actual_data_year_for_pred
):
    months_df = pd.DataFrame({'月': range(1, MONTHS_IN_YEAR + 1)})
    if target_year <= _last_actual_data_year_for_pred:
        actual_df_year = _historical_pop_data_for_pred[
            _historical_pop_data_for_pred["年"] == target_year
            ][['月', pref_name]].copy()
        if actual_df_year.empty: return months_df.assign(気温=np.nan)
        actual_df_year = actual_df_year.rename(columns={pref_name: '気温'})
        return pd.merge(months_df, actual_df_year, on='月', how='left').sort_values(by='月')
    else:
        trained_model = get_trained_monthly_prophet_model(
            _historical_pop_data_for_pred, pref_name, _last_actual_data_year_for_pred
        )
        if trained_model is None: return months_df.assign(気温=np.nan)
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
pop_data = get_temperature_data()
if pop_data.empty: st.error("気温データの読み込みに失敗したため、アプリケーションを起動できません。"); st.stop()
pref_names = [col for col in pop_data.columns if col not in ["年", "月"]]
if not pref_names: st.error("データから有効な地点情報が取得できませんでした。"); st.stop()

TEMPERATURE_DATA_SOURCE_URL = 'https://www.data.jma.go.jp/risk/obsdl/index.php'
header_cols = st.columns(2)
header_cols[0].markdown('<h2 style="margin-bottom: 0;">月別平均気温（1900～現在）</h2>', unsafe_allow_html=True)
header_cols[1].link_button(':material/link: 気象庁 気温データ', TEMPERATURE_DATA_SOURCE_URL,
                           help=f'出典: {TEMPERATURE_DATA_SOURCE_URL}')

available_years = sorted(pop_data["年"].unique())
if not available_years: st.error("データから有効な年情報が取得できませんでした。"); st.stop()
max_data_year = available_years[-1]

display_mode = st.radio(
    "表示モードを選択", ["全期間データ表示", "特定月・年平均の分析・予測", "特定年の月別データ"],
    horizontal=True, key="display_mode_radio"
)

if display_mode == "全期間データ表示":
    st.markdown("---")
    start_year_display, end_year_display = st.select_slider(
        "表示する期間（年）を選択", options=available_years, value=(available_years[0], available_years[-1]),
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
            "表示する地点を選択してください（未選択時は全地点表示）", options=pref_names, key="all_data_prefs_multiselect"
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
                x_axis_dates = pd.to_datetime(date_strings_all);
                y_axis_values = year_filtered_all_data[col_name].values
                fig_all_data.add_trace(go.Scatter(x=x_axis_dates, y=y_axis_values, mode='lines+markers', name=col_name,
                                                  marker=dict(color=plot_color, size=4),
                                                  line=dict(color=plot_color, width=1.5)))
            fig_all_data.update_layout(xaxis_title="年月", yaxis_title="平均気温（℃）", height=600,
                                       legend_title_text="地点")
            st.plotly_chart(fig_all_data, use_container_width=True)

elif display_mode == "特定月・年平均の分析・予測":
    st.markdown("---")
    selected_month = st.selectbox(
        "分析・予測対象の月を選択（「年平均」も選択可）", options=[*range(1, 13), ANNUAL_AVERAGE_MONTH_PROXY],
        format_func=lambda x: f"{x}月" if x != ANNUAL_AVERAGE_MONTH_PROXY else "年平均", key="specific_month_selectbox"
    )
    selected_prefs_from_ui_specific_month = st.multiselect(
        "分析・予測する地点を選択してください（未選択時は全地点）", options=pref_names,
        key="specific_month_multiselect_prefs"
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
        default_display_start_val = available_years[0];
        default_display_end_val = max_prediction_display_year
        display_start_year, display_end_year = st.select_slider(
            "グラフに表示/予測する期間（年）", options=display_slider_options_years,
            value=(default_display_start_val, default_display_end_val),
            key="display_period_slider_specific_month_future"
        )
    st.markdown("---")
    all_prefectures_processed_data = {};
    all_evaluation_metrics = {}
    if not target_prefs_for_analysis:
        st.info("分析・予測する地点データがありません。")
    else:
        with st.spinner("予測モデルを計算中...⏳"):
            for pref_name_analysis in target_prefs_for_analysis:
                try:
                    df_result, eval_metrics = prepare_and_predict_monthly_data(
                        pop_data, pref_name_analysis, selected_month, learn_start_year, learn_end_year,
                        display_start_year, display_end_year
                    )
                    all_prefectures_processed_data[pref_name_analysis] = df_result
                    all_evaluation_metrics[pref_name_analysis] = eval_metrics
                except Exception as e:
                    st.error(f"データ処理中に予期せぬエラー ({pref_name_analysis}): {e}")
                    all_prefectures_processed_data[pref_name_analysis] = pd.DataFrame()
                    all_evaluation_metrics[pref_name_analysis] = {model: {} for model in MODEL_NAMES}
        table_tab_specific, graph_tab_specific = st.tabs(['📈📉 予測結果（表と評価）', '📊 予測グラフ'])
        with table_tab_specific:
            st.subheader("各地点の気温データと予測結果")
            if not all_prefectures_processed_data or all(df.empty for df in all_prefectures_processed_data.values()):
                st.info("表示できるデータがありません。地点や期間を確認してください。")
            else:
                for pref_name_table, df_result_table in all_prefectures_processed_data.items():
                    st.markdown(f"#### {pref_name_table}")
                    if df_result_table.empty: st.warning("この地点のデータは処理できませんでした。"); continue

                    pred_col_names_in_table = {model_key: f"予測（{model_key.split('（')[0]}）" for model_key in
                                               MODEL_NAMES}
                    error_col_names_in_table = {model_key: f"誤差（{model_key.split('（')[0]}）" for model_key in
                                                MODEL_NAMES}

                    cols_to_show_in_table = ["年", "気温", "学・検"] + \
                                            [pred_col_names_in_table[m] for m in MODEL_NAMES] + \
                                            [error_col_names_in_table[m] for m in MODEL_NAMES]
                    existing_cols_table = [col for col in cols_to_show_in_table if col in df_result_table.columns]
                    st.dataframe(df_result_table[existing_cols_table].style.format(na_rep='-', precision=2),
                                 hide_index=True, use_container_width=True)

                    st.markdown("##### 📊 予測モデル評価指標")
                    current_eval_metrics = all_evaluation_metrics.get(pref_name_table, {})
                    if not current_eval_metrics or all(not v for v in current_eval_metrics.values()):
                        st.info("評価指標を計算できませんでした。")
                    else:
                        eval_data_for_display = []
                        for model_name_metric, metrics_metric in current_eval_metrics.items():
                            eval_data_for_display.append({
                                "モデル": model_name_metric, "学習MAE": metrics_metric.get("learn_mae", np.nan),
                                "学習RMSE": metrics_metric.get("learn_rmse", np.nan),
                                "検証MAE": metrics_metric.get("validation_mae", np.nan),
                                "検証RMSE": metrics_metric.get("validation_rmse", np.nan),
                            })
                        eval_df = pd.DataFrame(eval_data_for_display)
                        format_dict = {"学習MAE": "{:.2f}", "学習RMSE": "{:.2f}", "検証MAE": "{:.2f}",
                                       "検証RMSE": "{:.2f}"}
                        st.dataframe(
                            eval_df.style.format(format_dict, na_rep='-').highlight_min(subset=['検証RMSE', '学習RMSE'],
                                                                                        color='lightgreen', axis=0),
                            hide_index=True, use_container_width=True
                        )
                        best_model_name = "N/A";
                        best_model_metric_value = float('inf')
                        best_model_metric_name = "N/A";
                        best_model_period_label = ""
                        for model_name_eval in MODEL_NAMES:
                            metrics_eval = current_eval_metrics.get(model_name_eval, {})
                            val_rmse = metrics_eval.get("validation_rmse", float('inf'))
                            if pd.notna(val_rmse) and val_rmse < best_model_metric_value:
                                best_model_metric_value = val_rmse;
                                best_model_name = model_name_eval
                                best_model_metric_name = "検証RMSE";
                                best_model_period_label = "検証期間"
                        if best_model_name == "N/A" or best_model_period_label != "検証期間":
                            best_model_metric_value = float('inf')
                            for model_name_eval in MODEL_NAMES:
                                metrics_eval = current_eval_metrics.get(model_name_eval, {})
                                learn_rmse = metrics_eval.get("learn_rmse", float('inf'))
                                if pd.notna(learn_rmse) and learn_rmse < best_model_metric_value:
                                    best_model_metric_value = learn_rmse;
                                    best_model_name = model_name_eval
                                    best_model_metric_name = "学習RMSE";
                                    best_model_period_label = "学習期間"
                        if best_model_name != "N/A":
                            st.success(
                                f"🏆 **この条件での推奨予測モデル**: **{best_model_name}** ({best_model_metric_name}: {best_model_metric_value:.2f})")
                            other_metrics_strings = []
                            for model_to_compare in MODEL_NAMES:
                                if model_to_compare != best_model_name:
                                    metrics_to_compare = current_eval_metrics.get(model_to_compare, {})
                                    value_to_compare = np.nan
                                    if best_model_period_label == "検証期間":
                                        value_to_compare = metrics_to_compare.get("validation_rmse", np.nan)
                                    elif best_model_period_label == "学習期間":
                                        value_to_compare = metrics_to_compare.get("learn_rmse", np.nan)
                                    if pd.notna(value_to_compare): other_metrics_strings.append(
                                        f"{model_to_compare}: {value_to_compare:.2f}")
                            if other_metrics_strings: st.markdown(
                                f"   <small>（他のモデルの{best_model_metric_name}: {', '.join(other_metrics_strings)}）</small>",
                                unsafe_allow_html=True)
                        else:
                            st.info("有効な評価指標に基づいて推奨モデルを特定できませんでした。")
                    st.markdown("---")
        with graph_tab_specific:
            st.subheader("気温推移と予測の比較グラフ")
            if not all_prefectures_processed_data or all(df.empty for df in all_prefectures_processed_data.values()):
                st.info("グラフを表示できるデータがありません。")
            else:
                fig_specific_month = go.Figure()
                fig_specific_month.add_vrect(
                    x0=learn_start_year, x1=learn_end_year, fillcolor="rgba(0, 128, 0, 0.1)",
                    layer="below", line_width=0, annotation_text="学習期間", annotation_position="top left",
                    annotation_font_size=10, annotation_font_color="darkgreen"
                )
                color_palette_graph = px.colors.qualitative.Set1
                line_styles_graph_pred = {"線形回帰予測": "dash", "2次回帰予測": "dot", "Prophet予測": "dashdot"}
                pred_col_map_graph = {"線形回帰": "予測（線形）", "2次回帰": "予測（２次）", "Prophet": "予測（Prophet）"}
                plot_idx = 0
                for pref_name_graph, df_result_graph in all_prefectures_processed_data.items():
                    if df_result_graph.empty: continue
                    current_pref_color_graph = color_palette_graph[plot_idx % len(color_palette_graph)]
                    if "気温" in df_result_graph.columns and not df_result_graph["気温"].isnull().all():
                        fig_specific_month.add_trace(
                            go.Scatter(x=df_result_graph["年"], y=df_result_graph["気温"], mode="markers",
                                       name=f"{pref_name_graph}：実測",
                                       marker=dict(color=current_pref_color_graph, size=7, symbol='circle')))
                    for model_key_name_graph, pred_col_val_name_graph in pred_col_map_graph.items():
                        style_key_for_line_graph = f"{model_key_name_graph}予測"
                        if pred_col_val_name_graph in df_result_graph.columns and not df_result_graph[
                            pred_col_val_name_graph].isnull().all():
                            fig_specific_month.add_trace(
                                go.Scatter(x=df_result_graph["年"], y=df_result_graph[pred_col_val_name_graph],
                                           mode="lines", name=f"{pref_name_graph}：{model_key_name_graph}",
                                           line=dict(color=current_pref_color_graph,
                                                     dash=line_styles_graph_pred[style_key_for_line_graph])))
                    plot_idx += 1
                month_name_display = f"{selected_month}月" if selected_month != ANNUAL_AVERAGE_MONTH_PROXY else "年平均"

                # --- X軸の表示範囲を明示的に設定 ---
                fig_specific_month.update_xaxes(range=[display_start_year, display_end_year])
                # --- 設定ここまで ---

                fig_specific_month.update_layout(title_text=f"各地点の気温推移と予測比較（{month_name_display}）",
                                                 xaxis_title="年", yaxis_title="平均気温（℃）", height=700,
                                                 legend_title_text='凡例')
                st.plotly_chart(fig_specific_month, use_container_width=True)

elif display_mode == "特定年の月別データ":
    # (このブロックは変更なし)
    st.markdown("---")
    max_displayable_year_monthly = max_data_year + FUTURE_PREDICTION_YEARS
    years_options_monthly = list(range(available_years[0], max_displayable_year_monthly + 1))
    selected_years_for_monthly_display = st.multiselect(
        "比較表示する年を選択（複数選択可）", options=years_options_monthly,
        default=[max_data_year] if max_data_year in years_options_monthly else (
            [years_options_monthly[-1]] if years_options_monthly else []),
        key="specific_years_multiselect_future"
    )
    st.caption(f"💡 {max_data_year}年を超える年は、Prophetによる予測値で表示されます。")
    if not selected_years_for_monthly_display:
        st.info("比較表示する年を1つ以上選択してください。")
    else:
        selected_prefs_from_ui_specific_years = st.multiselect(
            f"表示する地点を選択（未選択時は全地点）", options=pref_names, key="specific_years_prefs_multiselect"
        )
        target_prefs_for_specific_years = selected_prefs_from_ui_specific_years if selected_prefs_from_ui_specific_years else pref_names
        if not target_prefs_for_specific_years:
            st.info("表示する地点を1つ以上選択してください。")
        else:
            st.subheader("選択年の月別気温比較（実測値と予測値）")
            fig_combined_years_monthly = go.Figure();
            color_palette_prefs_monthly = px.colors.qualitative.Plotly
            all_monthly_data_for_table = []
            with st.spinner("実測値取得および未来年予測を実行中...⏳"):
                for year_idx, current_selected_year in enumerate(selected_years_for_monthly_display):
                    year_type_label = "予測" if current_selected_year > max_data_year else "実測"
                    for pref_idx, current_pref_name in enumerate(target_prefs_for_specific_years):
                        monthly_temps_df = get_or_predict_specific_year_monthly_data(pop_data, current_pref_name,
                                                                                     current_selected_year,
                                                                                     max_data_year)
                        if monthly_temps_df.empty or monthly_temps_df['気温'].isnull().all(): continue
                        current_pref_color = color_palette_prefs_monthly[pref_idx % len(color_palette_prefs_monthly)]
                        current_year_line_style = LINE_STYLES_FOR_YEARS[year_idx % len(LINE_STYLES_FOR_YEARS)]
                        current_year_marker_symbol = MARKER_SYMBOLS_FOR_YEARS[year_idx % len(MARKER_SYMBOLS_FOR_YEARS)]
                        fig_combined_years_monthly.add_trace(go.Scatter(
                            x=monthly_temps_df['月'], y=monthly_temps_df['気温'], mode='lines+markers',
                            name=f"{current_pref_name} ({current_selected_year}年 {year_type_label})",
                            line=dict(color=current_pref_color, dash=current_year_line_style),
                            marker=dict(color=current_pref_color, symbol=current_year_marker_symbol, size=6),
                        ))
                        table_df_temp = monthly_temps_df.copy();
                        table_df_temp['年'] = current_selected_year;
                        table_df_temp['地点'] = current_pref_name;
                        table_df_temp['種類'] = year_type_label
                        all_monthly_data_for_table.append(table_df_temp[['月', '年', '地点', '気温', '種類']])
            if not fig_combined_years_monthly.data:
                st.info("選択された条件で表示できるグラフデータがありません。")
            else:
                fig_combined_years_monthly.update_layout(
                    title_text=f"選択年の月別平均気温比較", xaxis_title="月", yaxis_title="平均気温（℃）",
                    xaxis=dict(tickmode='array', tickvals=list(range(1, 13)),
                               ticktext=[f"{m}月" for m in range(1, 13)]),
                    height=700, legend_title_text="凡例（地点 - 年 - 種別）"
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
                        pivot_table_monthly_combined = combined_table_data_monthly.pivot_table(index='月',
                                                                                               columns=['地点', '年',
                                                                                                        '種類'],
                                                                                               values='気温').sort_index(
                            axis=0)
                        if isinstance(pivot_table_monthly_combined.columns, pd.MultiIndex):
                            pivot_table_monthly_combined = pivot_table_monthly_combined.sort_index(axis=1,
                                                                                                   level=['地点', '年',
                                                                                                          '種類'])
                        st.dataframe(pivot_table_monthly_combined.style.format("{:.1f}", na_rep='-'),
                                     use_container_width=True)
                    except Exception as e:
                        st.error(f"表の作成中にエラーが発生しました: {e}"); st.dataframe(combined_table_data_monthly)