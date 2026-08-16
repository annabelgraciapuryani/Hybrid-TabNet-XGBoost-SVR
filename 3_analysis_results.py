"""
Halaman Analysis & Results — 4 tab: Preprocessing, Feature Engineering,
Model Evaluation, Prediction.
"""

import io
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from core.models import save_model_and_scaler
from core.preprocessing import build_clean_dataframe, compute_missing_summary
from core.state import (
    CH_LAG,
    KEY_COLUMN_MAPPING,
    KEY_DF_CLEAN,
    KEY_DF_RAW,
    KEY_SELECTED_MODEL,
    eval_results_key,
    feature_list_key,
    fe_info_key,
    model_key,
    scaler_key,
)

st.title("Analysis & Results")

if KEY_DF_RAW not in st.session_state or KEY_COLUMN_MAPPING not in st.session_state:
    st.warning("Selesaikan pengaturan di halaman Data Input terlebih dahulu.")
    st.stop()

if KEY_SELECTED_MODEL not in st.session_state:
    st.warning("Jalankan pipeline di halaman Configuration terlebih dahulu.")
    st.stop()

selected_model = st.session_state[KEY_SELECTED_MODEL]
col_map = st.session_state[KEY_COLUMN_MAPPING]
col_tanggal = col_map["col_tanggal"]
col_target = col_map["col_target"]
df_raw = st.session_state[KEY_DF_RAW]

tab_preprocessing, tab_fe, tab_eval, tab_pred = st.tabs([
    "Preprocessing",
    "Feature Engineering",
    "Model Evaluation",
    "Prediction",
])


with tab_preprocessing:
    st.subheader("Perbandingan Dataset Sebelum dan Sesudah Preprocessing")

    df_clean = st.session_state.get(KEY_DF_CLEAN)

    if df_clean is None:
        st.info("Pipeline belum dijalankan. Jalankan pipeline di halaman Configuration.")
    else:
        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown("**Sebelum Preprocessing**")
            st.metric("Jumlah Baris", df_raw.shape[0])
            st.metric("Jumlah Kolom", df_raw.shape[1])
            st.metric("Total Missing Value", int(df_raw.isnull().sum().sum()))
            st.metric("Jumlah Duplikat Tanggal", int(df_raw.duplicated(subset=[col_tanggal]).sum()))

        with col_a:
            st.markdown("**Sesudah Preprocessing**")
            st.metric("Jumlah Baris", df_clean.shape[0])
            st.metric("Jumlah Kolom", df_clean.shape[1])
            st.metric("Total Missing Value", int(df_clean.isnull().sum().sum()))
            st.metric("Jumlah Duplikat Tanggal", int(df_clean.duplicated(subset=[col_tanggal]).sum()))

        st.subheader("Preview Dataset Hasil Preprocessing")
        st.dataframe(df_clean.head(20), use_container_width=True)

        st.subheader("Download Dataset Hasil Preprocessing")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            csv_buffer = io.StringIO()
            df_clean.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="dataset_bersih.csv",
                mime="text/csv",
            )

        with col_dl2:
            xlsx_buffer = io.BytesIO()
            with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
                df_clean.to_excel(writer, index=False)
            st.download_button(
                label="Download XLSX",
                data=xlsx_buffer.getvalue(),
                file_name="dataset_bersih.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


with tab_fe:
    st.subheader(f"Feature Engineering — {selected_model}")

    fe_info = st.session_state.get(fe_info_key(selected_model))
    final_features = st.session_state.get(feature_list_key(selected_model))

    if fe_info is None or final_features is None:
        st.info("Pipeline belum dijalankan. Jalankan pipeline di halaman Configuration.")
    else:
        note = fe_info.get("note")
        if note:
            st.info(note)

        if final_features:
            st.markdown("**Daftar Fitur Final yang Digunakan Model**")
            feat_df = pd.DataFrame({"No": range(1, len(final_features) + 1), "Nama Fitur": final_features})
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

        pearson_df = fe_info.get("pearson_df")
        if pearson_df is not None:
            with st.expander("Hasil Pearson Correlation Selection"):
                st.dataframe(pearson_df, use_container_width=True)

        importance_df = fe_info.get("importance_df")
        if importance_df is not None:
            with st.expander("Hasil TabNet Feature Importance"):
                st.dataframe(importance_df, use_container_width=True)

        lag_df = fe_info.get("lag_df")
        if lag_df is not None:
            with st.expander("Hasil CCF — Lag Terpilih per Fitur"):
                st.dataframe(lag_df, use_container_width=True)

        st.markdown("**Fitur Musiman**")
        st.write(
            f"month, month_sin, month_cos — selalu disertakan di semua model. "
            f"CH_lag{CH_LAG} juga ditambahkan sebagai fitur autoregresif."
        )


with tab_eval:
    st.subheader(f"Evaluasi Model — {selected_model}")

    eval_data = st.session_state.get(eval_results_key(selected_model))

    if eval_data is None:
        st.info("Pipeline belum dijalankan. Jalankan pipeline di halaman Configuration.")
    else:
        metrics = eval_data["metrics"]
        result_df = eval_data["result_df"]
        metadata = eval_data["model_metadata"]

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("RMSE", f"{metrics['RMSE']:.4f} mm")
        with m_col2:
            st.metric("MAE", f"{metrics['MAE']:.4f} mm")
        with m_col3:
            mape_val = metrics["MAPE"]
            st.metric("MAPE", f"{mape_val:.2f} %" if not np.isnan(mape_val) else "N/A")

        with st.expander("Metadata Model"):
            st.json(metadata)

        st.subheader("Grafik Actual vs Prediksi")
        chart_df = result_df.set_index("Tanggal")[["Aktual (mm)", "Prediksi (mm)"]]
        st.line_chart(chart_df)

        st.subheader("Tabel Actual vs Prediksi")
        st.dataframe(result_df, use_container_width=True)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv_buf = io.StringIO()
            result_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="Download CSV",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name=f"evaluasi_{selected_model.replace(' ', '_')}.csv",
                mime="text/csv",
            )
        with dl_col2:
            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False)
            st.download_button(
                label="Download Excel",
                data=xlsx_buf.getvalue(),
                file_name=f"evaluasi_{selected_model.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        model_instance = st.session_state.get(model_key(selected_model))
        scaler_instance = st.session_state.get(scaler_key(selected_model))




with tab_pred:
    st.subheader(f"Prediksi Intensitas Curah Hujan — {selected_model}")

    model_instance = st.session_state.get(model_key(selected_model))
    scaler_instance = st.session_state.get(scaler_key(selected_model))
    final_features = st.session_state.get(feature_list_key(selected_model))
    df_clean = st.session_state.get(KEY_DF_CLEAN)

    if model_instance is None or final_features is None or df_clean is None:
        st.info("Pipeline belum dijalankan. Jalankan pipeline di halaman Configuration.")
    else:
        eval_data = st.session_state.get(eval_results_key(selected_model))
        df_model_full = eval_data.get("test_df") if eval_data else None

        if eval_data and "train_df" in eval_data:
            df_model_full = pd.concat([eval_data["train_df"], eval_data["test_df"]], ignore_index=True)
        elif df_model_full is not None:
            pass

        available_dates = df_clean[col_tanggal].dropna().sort_values().unique()
        min_date = pd.Timestamp(available_dates[0]).date()
        max_date = pd.Timestamp(available_dates[-1]).date()

        selected_date = st.date_input(
            "Pilih tanggal prediksi",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

        if st.button("Prediksi", type="primary"):
            try:
                target_ts = pd.Timestamp(selected_date)

                if df_model_full is None:
                    st.error("Data model tidak tersedia. Jalankan ulang pipeline di halaman Configuration.")
                    st.stop()

                row = df_model_full[df_model_full[col_tanggal] == target_ts]

                if row.empty:
                    missing_feats = [f for f in final_features if f not in df_clean.columns]
                    if missing_feats:
                        st.error(
                            f"Prediksi gagal. Dataset tidak memiliki fitur yang diperlukan: "
                            f"{', '.join(missing_feats)}"
                        )
                    else:
                        st.error(
                            f"Prediksi gagal. Data untuk tanggal {selected_date} tidak tersedia "
                            "atau data historis tidak cukup untuk membentuk lag feature yang diperlukan."
                        )
                    st.stop()

                missing_in_row = [f for f in final_features if f not in row.columns or row[f].isnull().any()]
                if missing_in_row:
                    st.error(
                        f"Prediksi gagal. Fitur berikut tidak tersedia atau bernilai NaN "
                        f"untuk tanggal ini: {', '.join(missing_in_row)}"
                    )
                    st.stop()

                X_pred = row[final_features].values

                if scaler_instance is not None:
                    X_pred = scaler_instance.transform(X_pred)

                prediction = float(model_instance.predict(X_pred)[0])
                prediction = max(0.0, prediction)

                st.success(
                    f"Prediksi intensitas curah hujan = {prediction:,.2f} mm"
                    f" (Tanggal: {selected_date})"
                )

            except Exception:
                st.error("Prediksi gagal. Terjadi kesalahan saat memproses data.")
                st.error(traceback.format_exc())
