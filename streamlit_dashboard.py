"""
Unified Streamlit dashboard for pharmaceutical packaging inspection demo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Pharma Edge-AI Digital Twin", page_icon="💊", layout="wide")


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def classify_ocr_state(row: Dict) -> str:
    if row.get("status") == "PASS":
        return "PASS"
    if row.get("status") == "FAIL":
        return "UNREADABLE"
    return "ERROR"


def main() -> None:
    st.title("💊 Pharmaceutical Packaging Defect Monitoring")
    st.caption("YOLOv8 + OCR + PatchCore + Federated Learning + OEE")

    with st.sidebar:
        st.header("Data Inputs")
        ocr_path = st.text_input("OCR results", value="ocr_results.json")
        patch_scores_path = st.text_input("PatchCore scores", value="patchcore_results/scores.json")
        patch_threshold_path = st.text_input("PatchCore threshold", value="patchcore_results/threshold.json")
        fl_path = st.text_input("Federated metrics", value="fl_results_10round_cpu/fl_metrics.json")
        oee_path = st.text_input("SimPy/OEE metrics", value="simpy_oee_metrics.json")
        st.markdown("---")
        st.info("Use the slider in PatchCore panel to tune anomaly threshold live.")

    ocr_data = load_json(Path(ocr_path), [])
    patch_scores = load_json(Path(patch_scores_path), [])
    patch_threshold = load_json(Path(patch_threshold_path), {"optimal_threshold": 0.5, "roc_auc": 0.0})
    fl_data = load_json(Path(fl_path), {"history": []})
    oee_data = load_json(
        Path(oee_path),
        {"throughput_per_hour": 0, "defect_rate": 0, "oee_percent": 0, "reject_count": 0},
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Throughput/hr", f"{oee_data.get('throughput_per_hour', 0)}")
    c2.metric("Defect Rate", f"{oee_data.get('defect_rate', 0)}%")
    c3.metric("OEE", f"{oee_data.get('oee_percent', 0)}%")
    c4.metric("Reject Count", f"{oee_data.get('reject_count', 0)}")

    st.markdown("---")

    cam1, cam2 = st.columns(2)
    with cam1:
        st.subheader("Camera 1: Frontside YOLO")
        st.write("Display frontside YOLO detections and defect class counts from your detector logs.")
        st.caption("Hook in live feed/frame outputs here (OpenCV or RTSP source).")

    with cam2:
        st.subheader("Camera 2: Backside YOLO + OCR")
        if not ocr_data:
            st.warning("No OCR data found yet.")
        else:
            ocr_df = pd.DataFrame(ocr_data)
            ocr_df["ocr_state"] = ocr_df.apply(classify_ocr_state, axis=1)
            # 3-state OCR logic requested in project scope.
            state_counts = ocr_df["ocr_state"].value_counts()
            st.write("OCR 3-state Summary")
            st.bar_chart(state_counts)
            st.dataframe(
                ocr_df[["image_name", "status", "ocr_state", "lot_detected", "exp_detected", "message"]],
                use_container_width=True,
                height=260,
            )

    st.markdown("---")
    patch_col, fl_col = st.columns(2)

    with patch_col:
        st.subheader("PatchCore Anomaly Panel")
        if not patch_scores:
            st.warning("No PatchCore scores found yet.")
        else:
            p_df = pd.DataFrame(patch_scores)
            default_thr = float(patch_threshold.get("optimal_threshold", 0.5))
            score_min = float(p_df["anomaly_score"].min())
            score_max = float(p_df["anomaly_score"].max())
            threshold = st.slider(
                "Anomaly threshold",
                min_value=score_min,
                max_value=score_max,
                value=min(max(default_thr, score_min), score_max),
            )
            p_df["anomaly_flag"] = p_df["anomaly_score"] >= threshold
            st.write(
                f"ROC AUC: **{patch_threshold.get('roc_auc', 'N/A')}** | "
                f"Flagged images: **{int(p_df['anomaly_flag'].sum())}/{len(p_df)}**"
            )
            st.dataframe(
                p_df[["image_name", "anomaly_score", "true_label", "predicted_label", "anomaly_flag"]],
                use_container_width=True,
                height=260,
            )

    with fl_col:
        st.subheader("Federated Learning Panel")
        history: List[Dict] = fl_data.get("history", [])
        if not history:
            st.warning("No FL metrics found yet.")
        else:
            fl_df = pd.DataFrame(history)
            metric_cols = [col for col in ["global_map50", "global_map50_95"] if col in fl_df.columns]
            if metric_cols:
                st.line_chart(fl_df.set_index("round")[metric_cols])
            else:
                st.warning("FL history found, but no global mAP metric columns were present.")

            loss_rows = []
            for row in history:
                for site_id, loss_val in row.get("client_losses", {}).items():
                    loss_rows.append({"round": row["round"], "site": site_id, "loss": loss_val})
            if loss_rows:
                loss_df = pd.DataFrame(loss_rows)
                pivot = loss_df.pivot(index="round", columns="site", values="loss").sort_index()
                st.line_chart(pivot)
            st.caption(fl_data.get("privacy_note", ""))

    st.markdown("---")
    st.subheader("SimPy / OEE Panel")
    st.json(oee_data)


if __name__ == "__main__":
    main()
