"""
Unified Streamlit dashboard for pharmaceutical packaging inspection demo.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="Pharma Edge-AI Digital Twin", page_icon="P", layout="wide")


DIGITAL_TWIN_QUEUE_KEY = "digital_twin_events"
UPLOAD_DIR = Path("streamlit_uploads")
DEFAULT_FRONT_YOLO_WEIGHTS = Path("test_results/yolo_front/weights/best.pt")
DEFAULT_BACK_YOLO_WEIGHTS = Path("test_results/yolo_back/weights/best.pt")
PATCHCORE_CKPT = Path("patchcore_results/anomalib_logs/Patchcore/blister/v0/weights/lightning/model.ckpt")
YOLO_IMGSZ = 320  # Smaller input = faster inference (default 640 is overkill for defect detection)


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


def decide_final_action(
    front_status: str,
    back_status: str,
    ocr_state: str,
    patchcore_score: float | None,
    patchcore_threshold: float,
) -> str:
    patchcore_defect = patchcore_score is not None and patchcore_score >= patchcore_threshold
    if front_status == "DEFECT" or back_status == "DEFECT" or patchcore_defect:
        return "REJECT"
    if ocr_state in {"UNREADABLE", "ERROR"}:
        return "MANUAL_REVIEW"
    return "PASS"


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    from ultralytics import YOLO

    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {path}")
    model = YOLO(str(path))
    # Warm up the model with a dummy inference so the first real call is fast
    import numpy as np
    try:
        dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
        model(dummy, imgsz=YOLO_IMGSZ, half=True, verbose=False)
    except Exception:
        pass  # Warmup is best-effort
    return model


@st.cache_resource(show_spinner=False)
def load_patchcore_model(side="back"):
    """Load trained PatchCore model from checkpoint for live inference."""
    # Choose checkpoint based on side
    if side == "front":
        ckpt_path = next(Path("test_results/patchcore_front").glob("run_*/anomalib_logs/Patchcore/blister/v0/weights/lightning/model.ckpt"), None)
    else:
        # back side
        ckpt_path = next(Path("test_results/patchcore_back").glob("run_*/anomalib_logs/Patchcore/blister/v0/weights/lightning/model.ckpt"), None)
        
    if ckpt_path is None or not ckpt_path.exists():
        return None, None
        
    try:
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
        import torch
        torch.set_float32_matmul_precision("high")
        model = Patchcore(backbone="resnet18", layers=["layer2", "layer3"], num_neighbors=9, coreset_sampling_ratio=0.01)
        engine = Engine(accelerator="auto", devices=1, max_epochs=1)
        
        # Warmup
        try:
            import numpy as np
            import cv2
            dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
            dummy_path = Path("patchcore_dummy_warmup.jpg")
            if not dummy_path.exists():
                cv2.imwrite(str(dummy_path), dummy_img)
            engine.predict(model=model, ckpt_path=str(ckpt_path), data_path=str(dummy_path))
        except Exception:
            pass # Ignore warmup errors
            
        return model, engine, ckpt_path
    except Exception:
        return None, None


def run_patchcore_live(image_path: Path, side: str, threshold: float, preloaded_res=None) -> Dict:
    """Run PatchCore inference on a single image and return score + heatmap data URI."""
    res = preloaded_res if preloaded_res is not None else load_patchcore_model(side)
    if res == (None, None):
        return {"score": None, "heatmap": "", "anomaly_flag": False}
    model, engine, ckpt_path = res
    try:
        preds = engine.predict(model=model, ckpt_path=str(ckpt_path), data_path=str(image_path))
        batch = preds[0]
        score = float(batch.pred_score.cpu().item())
        heatmap_uri = ""
        # Generate heatmap overlay
        if hasattr(batch, "anomaly_map") and batch.anomaly_map is not None:
            import cv2
            import numpy as np
            amap = batch.anomaly_map.cpu().numpy()[0].squeeze()
            amap_norm = ((amap - amap.min()) / (amap.max() - amap.min() + 1e-8) * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(amap_norm, cv2.COLORMAP_JET)
            orig = cv2.imread(str(image_path))
            if orig is not None:
                orig_resized = cv2.resize(orig, (heatmap_color.shape[1], heatmap_color.shape[0]))
                overlay = cv2.addWeighted(orig_resized, 0.5, heatmap_color, 0.5, 0)
                heatmap_uri = image_array_to_data_uri(overlay)
        return {
            "score": round(score, 4),
            "heatmap": heatmap_uri,
            "anomaly_flag": score >= threshold,
        }
    except Exception as exc:
        return {"score": None, "heatmap": "", "anomaly_flag": False, "error": str(exc)}


def safe_package_id(package_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", package_id.strip())
    return cleaned or "PKG_UPLOAD"


def save_uploaded_image(uploaded_file, package_id: str, side: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    out_path = UPLOAD_DIR / f"{safe_package_id(package_id)}_{side}{suffix}"
    out_path.write_bytes(uploaded_file.getvalue())
    return out_path


def class_name(model, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def is_defect_detection(name: str, side: str) -> bool:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    if side == "front":
        return normalized in {"defect", "no_pill", "missing_pill", "damaged"} or "defect" in normalized
    return normalized in {"defect", "no_pack", "missing_pack", "damaged"} or "defect" in normalized


def run_yolo_inspection(weights_path: str, image_path: Path, side: str, conf_threshold: float) -> Dict:
    model = load_yolo_model(weights_path)
    results = model(str(image_path), conf=conf_threshold, imgsz=YOLO_IMGSZ, half=True, verbose=False)
    try:
        annotated_image = image_array_to_data_uri(results[0].plot())
    except Exception:
        annotated_image = ""
    boxes = getattr(results[0], "boxes", None)
    detections = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            name = class_name(model, cls_id)
            detections.append({"class": name, "confidence": round(conf, 4)})

    defect_hits = [det for det in detections if is_defect_detection(det["class"], side)]
    status = "DEFECT" if defect_hits else "PASS"
    top = max(detections, key=lambda det: det["confidence"], default=None)
    summary = "No defect detections"
    if defect_hits:
        best_defect = max(defect_hits, key=lambda det: det["confidence"])
        summary = f"{best_defect['class']} ({best_defect['confidence']:.2f})"
    elif top:
        summary = f"{top['class']} ({top['confidence']:.2f})"

    return {
        "status": status,
        "detections": detections,
        "summary": summary,
        "annotated_image": annotated_image,
        "model_names": getattr(model, "names", {}),
    }


def normalize_name(value: str) -> str:
    return Path(str(value)).stem.lower()


def lookup_patchcore_score(image_name: str, patch_scores: List[Dict]) -> float | None:
    target = normalize_name(image_name)
    for row in patch_scores:
        if normalize_name(str(row.get("image_name", ""))) == target:
            try:
                return float(row.get("anomaly_score"))
            except Exception:
                return None
    return None


def run_ocr_validation_subprocess(image_path: Path, package_id: str) -> Dict:
    """Run PaddleOCR outside the Streamlit process so protobuf is configured before Paddle imports."""
    output_path = UPLOAD_DIR / f"{safe_package_id(package_id)}_ocr.json"
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["FLAGS_use_mkldnn"] = "0"
    env["MKLDNN_ENABLED"] = "0"
    env["DNNL_VERBOSE"] = "0"
    script = (
        "import json, sys; "
        "from pathlib import Path; "
        "from ocr_lot_exp import validate_lot_exp; "
        "res = validate_lot_exp(Path(sys.argv[1])); "
        "Path(sys.argv[2]).write_text(json.dumps(res), encoding='utf-8')"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, str(image_path), str(output_path)],
        cwd=Path(__file__).resolve().parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "OCR subprocess failed."
        return {"status": "ERROR", "message": message, "lot_detected": None, "exp_detected": None}
    if not output_path.exists():
        return {"status": "ERROR", "message": "OCR subprocess did not produce an output JSON.", "lot_detected": None, "exp_detected": None}
    try:
        return json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "ERROR", "message": f"Could not parse OCR output: {exc}", "lot_detected": None, "exp_detected": None}


def run_uploaded_package_inspection(
    front_upload,
    back_upload,
    package_id: str,
    front_weights: str,
    back_weights: str,
    conf_threshold: float,
    patch_scores: List[Dict],
    patch_threshold: Dict,
    status_callback=None,
) -> Dict:
    """Run YOLO + OCR inspection only (fast). PatchCore runs separately."""
    threshold = float(patch_threshold.get("optimal_threshold", 0.5))

    def _update(msg: str):
        if status_callback:
            status_callback(msg)

    _update("Saving uploaded images...")
    front_path = save_uploaded_image(front_upload, package_id, "front")
    back_path = save_uploaded_image(back_upload, package_id, "back")

    # Run front YOLO, back YOLO, OCR in parallel (fast — no PatchCore here)
    _update("Running YOLO + OCR in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        fut_front = pool.submit(run_yolo_inspection, front_weights, front_path, "front", conf_threshold)
        fut_back = pool.submit(run_yolo_inspection, back_weights, back_path, "back", conf_threshold)
        fut_ocr = pool.submit(run_ocr_validation_subprocess, back_path, package_id)

        t0 = time.time()
        front_result = fut_front.result()
        t1 = time.time()
        _update(f"Front YOLO complete ✓ ({t1-t0:.1f}s)")
        back_result = fut_back.result()
        t2 = time.time()
        _update(f"Back YOLO complete ✓ ({t2-t1:.1f}s)")
        ocr_result = fut_ocr.result()
        t3 = time.time()
        _update(f"OCR complete ✓ ({t3-t2:.1f}s)")
        _update(f"⏱ Total: {t3-t0:.1f}s")

    # PatchCore score from pre-computed lookup only (live PatchCore is separate)
    patchcore_score = lookup_patchcore_score(back_upload.name, patch_scores)
    patchcore_text = f"Matched: {patchcore_score:.4f}" if patchcore_score is not None else "N/A"

    _update("Assembling final decision...")
    event = {
        "package_id": package_id,
        "front_yolo_status": front_result["status"],
        "back_yolo_status": back_result["status"],
        "ocr_state": classify_ocr_state(ocr_result),
        "patchcore_score": patchcore_score,
        "patchcore_text": patchcore_text,
        "patchcore_threshold": round(threshold, 4),
        "front_image_name": front_upload.name,
        "back_image_name": back_upload.name,
        "front_path": str(front_path),
        "back_path": str(back_path),
        "front_image": uploaded_file_to_data_uri(front_upload),
        "back_image": uploaded_file_to_data_uri(back_upload),
        "front_annotated_image": front_result["annotated_image"],
        "back_annotated_image": back_result["annotated_image"],
        "front_yolo_summary": front_result["summary"],
        "front_yolo_detections": front_result["detections"],
        "back_yolo_summary": back_result["summary"],
        "back_yolo_detections": back_result["detections"],
        "ocr_message": str(ocr_result.get("message", "")),
        "lot_detected": ocr_result.get("lot_detected"),
        "exp_detected": ocr_result.get("exp_detected"),
        "raw_text": ocr_result.get("raw_text", ""),
    }

    event["final_action"] = decide_final_action(
        event["front_yolo_status"],
        event["back_yolo_status"],
        event["ocr_state"],
        patchcore_score,
        threshold,
    )
    event["reject_stage"] = "FRONT_CAMERA" if event["front_yolo_status"] == "DEFECT" else "FINAL_GATE"
    return event


def uploaded_file_to_data_uri(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    mime = uploaded_file.type or "image/png"
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_array_to_data_uri(image_array) -> str:
    if image_array is None:
        return ""
    try:
        import cv2

        ok, buffer = cv2.imencode(".jpg", image_array)
        if not ok:
            return ""
        encoded = base64.b64encode(buffer).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


def render_data_uri_image(data_uri: str, caption: str) -> None:
    if not data_uri:
        st.info("No boxed detection image available yet.")
        return
    st.markdown(
        f"""
        <figure style="margin:0">
          <img src="{data_uri}" style="width:100%;border-radius:8px;border:1px solid rgba(49,51,63,.18)" />
          <figcaption style="font-size:0.85rem;color:#6b7280;margin-top:0.35rem">{caption}</figcaption>
        </figure>
        """,
        unsafe_allow_html=True,
    )


def event_for_export(event: Dict) -> Dict:
    image_keys = {"front_image", "back_image", "front_annotated_image", "back_annotated_image", "patchcore_heatmap", "patchcore_front_heatmap"}
    return {key: value for key, value in event.items() if key not in image_keys}


def build_sample_events(ocr_data: List[Dict], patch_scores: List[Dict], patch_threshold: Dict, limit: int = 12) -> List[Dict]:
    threshold = float(patch_threshold.get("optimal_threshold", 0.5))
    events: List[Dict] = []
    source_count = max(len(ocr_data), len(patch_scores), limit)

    if source_count == 0:
        source_count = limit

    for idx in range(min(limit, source_count)):
        ocr_row = ocr_data[idx % len(ocr_data)] if ocr_data else {}
        patch_row = patch_scores[idx % len(patch_scores)] if patch_scores else {}
        ocr_state = classify_ocr_state(ocr_row) if ocr_row else ("UNREADABLE" if idx % 4 == 1 else "PASS")
        patchcore_score = float(patch_row.get("anomaly_score", 0.24 + (idx % 5) * 0.12))
        predicted_label = str(patch_row.get("predicted_label", patch_row.get("true_label", "good"))).lower()
        anomaly_flag = patchcore_score >= threshold or predicted_label == "defect"
        front_status = "DEFECT" if anomaly_flag and idx % 3 == 0 else "PASS"
        back_status = "DEFECT" if anomaly_flag and idx % 3 != 0 else "PASS"
        package_id = f"PKG-{idx + 1:03d}"
        final_action = decide_final_action(front_status, back_status, ocr_state, patchcore_score, threshold)
        events.append(
            {
                "package_id": package_id,
                "front_yolo_status": front_status,
                "back_yolo_status": back_status,
                "ocr_state": ocr_state,
                "patchcore_score": round(patchcore_score, 4),
                "patchcore_threshold": round(threshold, 4),
                "final_action": final_action,
                "front_image_name": "sample_front.jpg",
                "back_image_name": str(ocr_row.get("image_name", patch_row.get("image_name", "sample_back.jpg"))),
                "front_image": "",
                "back_image": "",
            }
        )
    return events


def render_digital_twin_html(events: List[Dict], speed: float) -> str:
    safe_events = events
    events_json = json.dumps(safe_events).replace("</", "<\\/")
    speed_json = json.dumps(float(speed))
    return """
<div class="twin-shell">
  <style>
    .twin-shell {
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: #f7fafc;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.015)),
        radial-gradient(circle at 16% 8%, rgba(56, 189, 248, 0.24), transparent 28%),
        radial-gradient(circle at 86% 18%, rgba(34, 197, 94, 0.16), transparent 30%),
        linear-gradient(135deg, #0b1118 0%, #15202b 46%, #091018 100%);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 22px 70px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .twin-topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px 20px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      background: rgba(255, 255, 255, 0.035);
    }
    .twin-title {
      font-size: 18px;
      font-weight: 750;
      letter-spacing: 0;
    }
    .twin-subtitle {
      color: #b8c4d1;
      font-size: 12px;
      margin-top: 3px;
    }
    .status-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .stat-pill {
      border: 1px solid rgba(255, 255, 255, 0.16);
      background: rgba(255, 255, 255, 0.07);
      padding: 8px 10px;
      border-radius: 8px;
      min-width: 82px;
      text-align: center;
    }
    .stat-pill strong {
      display: block;
      font-size: 18px;
      line-height: 18px;
    }
    .stat-pill span {
      display: block;
      margin-top: 2px;
      color: #aebccc;
      font-size: 10px;
      text-transform: uppercase;
    }
    .stage {
      position: relative;
      height: 440px;
      overflow: hidden;
      background:
        radial-gradient(ellipse at center, rgba(255, 255, 255, 0.05), transparent 62%),
        linear-gradient(90deg, rgba(255, 255, 255, 0.045) 1px, transparent 1px) 0 0 / 52px 52px,
        linear-gradient(0deg, rgba(255, 255, 255, 0.035) 1px, transparent 1px) 0 0 / 52px 52px;
    }
    .stage:before {
      content: "";
      position: absolute;
      inset: 22px 22px 20px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 16px;
      pointer-events: none;
    }
    .floor-line {
      position: absolute;
      left: 0;
      right: 0;
      height: 1px;
      background: rgba(255, 255, 255, 0.08);
    }
    .floor-line.one { top: 118px; }
    .floor-line.two { top: 286px; }
    .conveyor {
      position: absolute;
      left: 28px;
      right: 28px;
      top: 190px;
      height: 90px;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.16), transparent 24%),
        linear-gradient(180deg, #44515e 0%, #202832 100%);
      box-shadow: inset 0 16px 24px rgba(255, 255, 255, 0.08), inset 0 -18px 26px rgba(0, 0, 0, 0.44), 0 18px 38px rgba(0, 0, 0, 0.34);
      overflow: hidden;
    }
    .conveyor:before {
      content: "";
      position: absolute;
      inset: 0;
      background:
        repeating-linear-gradient(90deg, rgba(255,255,255,0.16) 0 4px, transparent 4px 36px),
        linear-gradient(90deg, rgba(14, 165, 233, 0.08), transparent 18%, transparent 82%, rgba(34, 197, 94, 0.08));
      animation: beltMove 0.8s linear infinite;
    }
    .conveyor:after {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      top: 35px;
      height: 2px;
      background: rgba(255, 255, 255, 0.25);
    }
    .belt-rail {
      position: absolute;
      left: 38px;
      right: 38px;
      height: 7px;
      border-radius: 999px;
      background: linear-gradient(90deg, #94a3b8, #e2e8f0, #94a3b8);
      box-shadow: 0 5px 16px rgba(0, 0, 0, 0.35);
      z-index: 4;
    }
    .belt-rail.top { top: 182px; }
    .belt-rail.bottom { top: 286px; }
    .roller {
      position: absolute;
      top: 198px;
      width: 28px;
      height: 74px;
      border-radius: 999px;
      background: linear-gradient(90deg, #111827, #64748b 45%, #111827);
      border: 1px solid rgba(255, 255, 255, 0.16);
      opacity: 0.74;
      z-index: 2;
      animation: rollerSpin 0.9s linear infinite;
    }
    .roller.r1 { left: 9%; }
    .roller.r2 { left: 24%; }
    .roller.r3 { left: 39%; }
    .roller.r4 { left: 54%; }
    .roller.r5 { left: 69%; }
    .roller.r6 { left: 84%; }
    @keyframes rollerSpin {
      0% { filter: brightness(1); }
      50% { filter: brightness(1.32); }
      100% { filter: brightness(1); }
    }
    @keyframes beltMove {
      from { transform: translateX(0); }
      to { transform: translateX(36px); }
    }
    .lane {
      position: absolute;
      left: 72%;
      right: 5%;
      height: 58px;
      border-radius: 10px;
      border: 1px dashed rgba(255, 255, 255, 0.18);
      background: rgba(255, 255, 255, 0.045);
      box-shadow: inset 0 0 18px rgba(255, 255, 255, 0.04);
    }
    .lane.review { top: 81px; }
    .lane.reject { top: 316px; }
    .lane-label {
      position: absolute;
      right: 18px;
      top: 12px;
      font-size: 11px;
      color: #d8e2ec;
      text-transform: uppercase;
      letter-spacing: 0;
    }
    .flow-arrow {
      position: absolute;
      top: 222px;
      width: 46px;
      height: 24px;
      color: rgba(226, 232, 240, 0.62);
      font-size: 26px;
      line-height: 24px;
      z-index: 5;
      animation: arrowPulse 1.4s ease-in-out infinite;
    }
    .flow-arrow.a1 { left: 13%; }
    .flow-arrow.a2 { left: 36%; animation-delay: 0.2s; }
    .flow-arrow.a3 { left: 60%; animation-delay: 0.4s; }
    @keyframes arrowPulse {
      0%, 100% { transform: translateX(0); opacity: 0.34; }
      50% { transform: translateX(10px); opacity: 0.9; }
    }
    .station {
      position: absolute;
      top: 92px;
      width: 130px;
      height: 88px;
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.12), transparent 36%),
        rgba(12, 18, 26, 0.92);
      box-shadow: 0 16px 34px rgba(0, 0, 0, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.08);
      display: grid;
      place-items: center;
      text-align: center;
      z-index: 10;
    }
    .station-camera {
      width: 54px;
      height: 24px;
      margin: 0 auto 6px;
      border-radius: 8px;
      background: linear-gradient(180deg, #334155, #0f172a);
      border: 1px solid rgba(255, 255, 255, 0.18);
      position: relative;
      box-shadow: inset 0 6px 10px rgba(255, 255, 255, 0.08);
    }
    .station-camera:before {
      content: "";
      position: absolute;
      left: 18px;
      top: 5px;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: radial-gradient(circle, #93c5fd 0 22%, #1d4ed8 35%, #020617 72%);
      box-shadow: 0 0 14px rgba(96, 165, 250, 0.7);
    }
    .station-camera:after {
      content: "";
      position: absolute;
      right: 7px;
      top: 8px;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #22c55e;
      box-shadow: 0 0 12px rgba(34, 197, 94, 0.85);
    }
    .station .name {
      font-weight: 750;
      font-size: 13px;
    }
    .station .kind {
      color: #aebccc;
      font-size: 11px;
      margin-top: 2px;
    }
    .station:after {
      content: "";
      position: absolute;
      left: 50%;
      top: 70px;
      width: 80px;
      height: 138px;
      transform: translateX(-50%);
      clip-path: polygon(45% 0, 55% 0, 100% 100%, 0 100%);
      background: linear-gradient(180deg, rgba(56, 189, 248, 0.38), rgba(56, 189, 248, 0.02));
      opacity: 0.55;
      pointer-events: none;
    }
    .station.active {
      border-color: rgba(99, 179, 237, 0.85);
      box-shadow: 0 0 0 2px rgba(66, 153, 225, 0.18), 0 0 32px rgba(66, 153, 225, 0.34);
    }
    .station.active:after {
      opacity: 0.92;
      animation: scanBeam 0.55s ease-in-out infinite alternate;
    }
    @keyframes scanBeam {
      from { filter: brightness(1); transform: translateX(-50%) scaleX(0.9); }
      to { filter: brightness(1.35); transform: translateX(-50%) scaleX(1.14); }
    }
    .station.s1 { left: 22%; }
    .station.s2 { left: 46%; }
    .decision-node {
      position: absolute;
      left: 66.5%;
      top: 198px;
      width: 70px;
      height: 70px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      color: #dbeafe;
      font-size: 10px;
      font-weight: 800;
      text-align: center;
      background: radial-gradient(circle, rgba(30, 64, 175, 0.82), rgba(15, 23, 42, 0.92));
      border: 1px solid rgba(147, 197, 253, 0.5);
      box-shadow: 0 0 24px rgba(59, 130, 246, 0.28), inset 0 0 16px rgba(147, 197, 253, 0.12);
      z-index: 7;
    }
    .gate {
      position: absolute;
      left: 71%;
      top: 136px;
      width: 96px;
      height: 170px;
      pointer-events: none;
      z-index: 11;
    }
    .gate-post {
      position: absolute;
      left: 43px;
      top: 0;
      width: 10px;
      height: 170px;
      border-radius: 5px;
      background: linear-gradient(#d7dee8, #6d7b88);
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    .gate-arm {
      position: absolute;
      left: 46px;
      top: 79px;
      width: 94px;
      height: 11px;
      border-radius: 8px;
      transform-origin: 0 50%;
      transform: rotate(0deg);
      background: linear-gradient(90deg, #f6ad55, #f56565);
      box-shadow: 0 0 22px rgba(245, 101, 101, 0.34);
      transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .gate-arm.fire {
      transform: rotate(34deg);
      box-shadow: 0 0 32px rgba(245, 101, 101, 0.65);
    }
    .bin {
      position: absolute;
      right: 28px;
      width: 155px;
      height: 54px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 12px;
      letter-spacing: 0;
      border: 1px solid rgba(255, 255, 255, 0.18);
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 12px 26px rgba(0, 0, 0, 0.24);
    }
    .bin.pass { top: 208px; background: linear-gradient(135deg, rgba(22, 101, 52, 0.7), rgba(72, 187, 120, 0.22)); color: #dcfce7; }
    .bin.review { top: 83px; background: linear-gradient(135deg, rgba(133, 77, 14, 0.72), rgba(236, 201, 75, 0.2)); color: #fefcbf; }
    .bin.reject { top: 318px; background: linear-gradient(135deg, rgba(127, 29, 29, 0.78), rgba(245, 101, 101, 0.22)); color: #fee2e2; }
    .pack-layer {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    .pack {
      position: absolute;
      width: 88px;
      height: 46px;
      border-radius: 10px;
      transform: translate(-50%, -50%);
      border: 1px solid rgba(255, 255, 255, 0.5);
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.88), rgba(148, 163, 184, 0.92)),
        repeating-linear-gradient(90deg, rgba(15, 23, 42, 0.08) 0 2px, transparent 2px 8px);
      box-shadow: 0 16px 28px rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.6);
      transition: border-color 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
      z-index: 20;
    }
    .pack:before {
      content: "";
      position: absolute;
      inset: 7px;
      border-radius: 6px;
      background:
        radial-gradient(circle at 18% 50%, #edf2f7 0 7px, #718096 8px 10px, transparent 11px),
        radial-gradient(circle at 40% 50%, #edf2f7 0 7px, #718096 8px 10px, transparent 11px),
        radial-gradient(circle at 62% 50%, #edf2f7 0 7px, #718096 8px 10px, transparent 11px),
        radial-gradient(circle at 84% 50%, #edf2f7 0 7px, #718096 8px 10px, transparent 11px);
      opacity: 0.95;
    }
    .pack:after {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: 10px;
      background: linear-gradient(110deg, transparent 0 30%, rgba(255,255,255,0.45) 42%, transparent 54% 100%);
      transform: translateX(-90%);
      animation: packShine 2.2s ease-in-out infinite;
    }
    @keyframes packShine {
      0%, 45% { transform: translateX(-110%); opacity: 0; }
      60% { opacity: 0.85; }
      100% { transform: translateX(110%); opacity: 0; }
    }
    .pack.pass {
      border-color: #68d391;
      box-shadow: 0 0 0 2px rgba(72, 187, 120, 0.22), 0 16px 28px rgba(0, 0, 0, 0.32);
    }
    .pack.reject {
      border-color: #fc8181;
      box-shadow: 0 0 0 2px rgba(245, 101, 101, 0.24), 0 16px 28px rgba(0, 0, 0, 0.32);
    }
    .pack.review {
      border-color: #f6e05e;
      box-shadow: 0 0 0 2px rgba(236, 201, 75, 0.24), 0 16px 28px rgba(0, 0, 0, 0.32);
    }
    .pack-label {
      position: absolute;
      left: 50%;
      top: -20px;
      transform: translateX(-50%);
      padding: 3px 7px;
      border-radius: 6px;
      background: rgba(10, 14, 20, 0.86);
      color: #edf2f7;
      font-size: 10px;
      white-space: nowrap;
      box-shadow: 0 8px 18px rgba(0, 0, 0, 0.24);
    }
    .monitor {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 14px;
      padding: 14px 18px 18px;
      background: rgba(255, 255, 255, 0.045);
      border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    .readout {
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(8, 13, 20, 0.68);
      border-radius: 10px;
      padding: 12px;
      min-height: 112px;
    }
    .readout-title {
      color: #aebccc;
      text-transform: uppercase;
      font-size: 11px;
      margin-bottom: 8px;
    }
    .readout-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .field {
      background: rgba(255, 255, 255, 0.06);
      border-radius: 8px;
      padding: 8px;
      min-height: 48px;
    }
    .field span {
      display: block;
      color: #aebccc;
      font-size: 10px;
      text-transform: uppercase;
      margin-bottom: 3px;
    }
    .field strong {
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    .thumbs {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .thumb {
      height: 112px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      background: rgba(8, 13, 20, 0.68);
      border-radius: 10px;
      overflow: hidden;
      position: relative;
      display: grid;
      place-items: center;
      color: #7f8ea3;
      font-size: 12px;
    }
    .thumb img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: none;
    }
    .thumb.has-image img {
      display: block;
    }
    .thumb-label {
      position: absolute;
      left: 8px;
      bottom: 7px;
      right: 8px;
      padding: 4px 6px;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.55);
      color: #edf2f7;
      font-size: 10px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    @media (max-width: 760px) {
      .twin-topbar, .monitor { grid-template-columns: 1fr; display: block; }
      .status-row { justify-content: flex-start; margin-top: 10px; }
      .monitor { padding: 12px; }
      .readout-grid { grid-template-columns: 1fr 1fr; }
      .station { width: 92px; }
      .station.s1 { left: 18%; }
      .station.s2 { left: 44%; }
      .bin { width: 118px; right: 16px; }
    }
  </style>

  <div class="twin-topbar">
    <div>
      <div class="twin-title">Python Digital Twin: Blister Pack Inspection Line</div>
      <div class="twin-subtitle">Conveyor replay driven by YOLO, OCR, PatchCore, and uploaded package events.</div>
    </div>
    <div class="status-row">
      <div class="stat-pill"><strong id="countPass">0</strong><span>Pass</span></div>
      <div class="stat-pill"><strong id="countReview">0</strong><span>Review</span></div>
      <div class="stat-pill"><strong id="countReject">0</strong><span>Reject</span></div>
      <div class="stat-pill"><strong id="countLive">0</strong><span>Live</span></div>
    </div>
  </div>

  <div class="stage" id="stage">
    <div class="floor-line one"></div>
    <div class="floor-line two"></div>
    <div class="lane review"><div class="lane-label">Manual review lane</div></div>
    <div class="lane reject"><div class="lane-label">Reject lane</div></div>
    <div class="belt-rail top"></div>
    <div class="belt-rail bottom"></div>
    <div class="roller r1"></div>
    <div class="roller r2"></div>
    <div class="roller r3"></div>
    <div class="roller r4"></div>
    <div class="roller r5"></div>
    <div class="roller r6"></div>
    <div class="conveyor"></div>
    <div class="flow-arrow a1">&#8594;</div>
    <div class="flow-arrow a2">&#8594;</div>
    <div class="flow-arrow a3">&#8594;</div>
    <div class="station s1" id="cam1">
      <div>
        <div class="station-camera"></div>
        <div class="name">CAMERA 1</div>
        <div class="kind">Front YOLO</div>
      </div>
    </div>
    <div class="station s2" id="cam2">
      <div>
        <div class="station-camera"></div>
        <div class="name">CAMERA 2</div>
        <div class="kind">Back + OCR</div>
      </div>
    </div>
    <div class="decision-node">AI<br>DECISION</div>
    <div class="gate"><div class="gate-post"></div><div class="gate-arm" id="gateArm"></div></div>
    <div class="bin review">MANUAL REVIEW</div>
    <div class="bin pass">PASS OUTPUT</div>
    <div class="bin reject">REJECT BIN</div>
    <div class="pack-layer" id="packLayer"></div>
  </div>

  <div class="monitor">
    <div class="readout">
      <div class="readout-title">Current Package Telemetry</div>
      <div class="readout-grid">
        <div class="field"><span>Package</span><strong id="pkgId">Waiting</strong></div>
        <div class="field"><span>Front YOLO</span><strong id="frontStatus">-</strong></div>
        <div class="field"><span>Back YOLO</span><strong id="backStatus">-</strong></div>
        <div class="field"><span>OCR</span><strong id="ocrStatus">-</strong></div>
        <div class="field"><span>PatchCore</span><strong id="patchScore">-</strong></div>
        <div class="field"><span>Decision</span><strong id="decision">-</strong></div>
      </div>
    </div>
    <div class="thumbs">
      <div class="thumb" id="frontThumb"><span>No front image</span><img id="frontImg" alt="front"><div class="thumb-label" id="frontName">Front side</div></div>
      <div class="thumb" id="backThumb"><span>No back image</span><img id="backImg" alt="back"><div class="thumb-label" id="backName">Back side</div></div>
    </div>
  </div>

  <script>
    const events = __EVENTS__;
    const speed = __SPEED__;
    const stage = document.getElementById("stage");
    const layer = document.getElementById("packLayer");
    const cam1 = document.getElementById("cam1");
    const cam2 = document.getElementById("cam2");
    const gateArm = document.getElementById("gateArm");
    const counts = { PASS: 0, MANUAL_REVIEW: 0, REJECT: 0 };
    const active = [];
    let cursor = 0;
    let lastSpawn = 0;
    const durationMs = 7600 / speed;
    const spawnGapMs = 1350 / speed;

    function cleanAction(action) {
      if (action === "REJECT" || action === "MANUAL_REVIEW") return action;
      return "PASS";
    }

    function actionClass(action) {
      if (action === "REJECT") return "reject";
      if (action === "MANUAL_REVIEW") return "review";
      return "pass";
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value || "-";
    }

    function setThumb(prefix, src, name) {
      const wrap = document.getElementById(prefix + "Thumb");
      const img = document.getElementById(prefix + "Img");
      const label = document.getElementById(prefix + "Name");
      label.textContent = name || (prefix === "front" ? "Front side" : "Back side");
      if (src) {
        img.src = src;
        wrap.classList.add("has-image");
      } else {
        img.removeAttribute("src");
        wrap.classList.remove("has-image");
      }
    }

    function updateReadout(event) {
      setText("pkgId", event.package_id);
      setText("frontStatus", event.front_yolo_status);
      setText("backStatus", event.back_yolo_status);
      setText("ocrStatus", event.ocr_state);
      if (event.patchcore_score === null || event.patchcore_score === undefined) {
        setText("patchScore", event.patchcore_text || "N/A");
      } else {
        setText("patchScore", Number(event.patchcore_score).toFixed(3));
      }
      setText("decision", event.final_action);
      setThumb("front", event.front_image, event.front_image_name);
      setThumb("back", event.back_image, event.back_image_name);
    }

    function updateCounters() {
      document.getElementById("countPass").textContent = counts.PASS;
      document.getElementById("countReview").textContent = counts.MANUAL_REVIEW;
      document.getElementById("countReject").textContent = counts.REJECT;
      document.getElementById("countLive").textContent = active.length;
    }

    function spawn(now) {
      if (!events.length || cursor >= events.length) return;
      const event = events[cursor];
      cursor += 1;
      const el = document.createElement("div");
      const action = cleanAction(event.final_action);
      el.className = "pack";
      el.innerHTML = '<div class="pack-label"></div>';
      el.querySelector(".pack-label").textContent = event.package_id || "PKG";
      layer.appendChild(el);
      active.push({ el, event, action, start: now, counted: false });
      updateReadout(event);
    }

    function pointFor(progress, action, event) {
      const width = stage.clientWidth;
      const left = 58;
      const right = width - 155;
      const x = left + (right - left) * progress;
      let y = 235;
      const rejectStart = event.reject_stage === "FRONT_CAMERA" ? 0.40 : 0.72;
      if (action === "REJECT" && progress > rejectStart) {
        const t = Math.min(1, (progress - rejectStart) / (1 - rejectStart));
        y = 235 + 110 * t; // Target: 345
      } else if (action === "MANUAL_REVIEW" && progress > 0.72) {
        const t = Math.min(1, (progress - 0.72) / 0.28);
        y = 235 - 125 * t; // Target: 110
      }
      return { x, y };
    }

    function tick(now) {
      if ((!lastSpawn || now - lastSpawn > spawnGapMs) && cursor < events.length) {
        spawn(now);
        lastSpawn = now;
      }

      let c1Active = false;
      let c2Active = false;
      let gateActive = false;

      for (let i = active.length - 1; i >= 0; i -= 1) {
        const item = active[i];
        const progress = Math.min(1, (now - item.start) / durationMs);
        const point = pointFor(progress, item.action, item.event);
        item.el.style.left = point.x + "px";
        item.el.style.top = point.y + "px";

        if (Math.abs(progress - 0.29) < 0.045) c1Active = true;
        if (item.event.reject_stage !== "FRONT_CAMERA" && Math.abs(progress - 0.53) < 0.045) c2Active = true;
        if (
          (item.event.reject_stage === "FRONT_CAMERA" && progress > 0.34 && progress < 0.54) ||
          (item.event.reject_stage !== "FRONT_CAMERA" && progress > 0.68 && progress < 0.82 && item.action !== "PASS")
        ) {
          gateActive = true;
        }
        if (
          (item.action === "REJECT" && item.event.reject_stage === "FRONT_CAMERA" && progress > 0.31) ||
          (item.action !== "PASS" && item.event.reject_stage !== "FRONT_CAMERA" && progress > 0.55) ||
          (item.action === "PASS" && progress > 0.55)
        ) {
          item.el.className = "pack " + actionClass(item.action);
        }

        if (progress >= 1) {
          if (!item.counted) {
            counts[item.action] += 1;
            item.counted = true;
          }
          item.el.remove();
          active.splice(i, 1);
        }
      }

      cam1.classList.toggle("active", c1Active);
      cam2.classList.toggle("active", c2Active);
      gateArm.classList.toggle("fire", gateActive);
      updateCounters();
      requestAnimationFrame(tick);
    }

    updateCounters();
    requestAnimationFrame(tick);
  </script>
</div>
""".replace("__EVENTS__", events_json).replace("__SPEED__", speed_json)


def render_digital_twin_3d_html(events: List[Dict], speed: float) -> str:
    safe_events = []
    for event in events:
        safe_events.append(
            {
                "package_id": event.get("package_id", "PKG"),
                "front_yolo_status": event.get("front_yolo_status", "N/A"),
                "back_yolo_status": event.get("back_yolo_status", "N/A"),
                "ocr_state": event.get("ocr_state", "N/A"),
                "patchcore_score": event.get("patchcore_score"),
                "patchcore_text": event.get("patchcore_text", "N/A"),
                "final_action": event.get("final_action", "PASS"),
                "reject_stage": event.get("reject_stage", "FINAL_GATE"),
                "front_image": event.get("front_image", ""),
                "back_image": event.get("back_image", ""),
                "front_image_name": event.get("front_image_name", "frontside"),
                "back_image_name": event.get("back_image_name", "backside"),
            }
        )

    events_json = json.dumps(safe_events).replace("</", "<\\/")
    speed_json = json.dumps(float(speed))
    return """
<div class="twin3d-shell">
  <style>
    .twin3d-shell {
      height: 742px;
      min-height: 742px;
      color: #f8fafc;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.02)),
        linear-gradient(145deg, #101418 0%, #151b20 48%, #0d1115 100%);
      border: 1px solid rgba(255, 255, 255, 0.13);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 24px 74px rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .twin3d-topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 14px 18px 12px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      background: rgba(255, 255, 255, 0.035);
    }
    .twin3d-title {
      font-size: 18px;
      font-weight: 760;
      letter-spacing: 0;
      line-height: 1.2;
    }
    .twin3d-subtitle {
      color: #b9c4cf;
      font-size: 12px;
      margin-top: 3px;
    }
    .twin3d-stats {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }
    .twin3d-pill {
      min-width: 78px;
      padding: 8px 10px;
      text-align: center;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.065);
    }
    .twin3d-pill strong {
      display: block;
      font-size: 18px;
      line-height: 18px;
    }
    .twin3d-pill span {
      display: block;
      margin-top: 3px;
      color: #aab7c4;
      font-size: 10px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .twin3d-workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 292px;
      height: 628px;
    }
    .twin3d-stage {
      position: relative;
      min-width: 0;
      height: 628px;
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.05), transparent 18%),
        linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px),
        linear-gradient(0deg, rgba(255, 255, 255, 0.022) 1px, transparent 1px),
        #0b0f13;
      background-size: auto, 44px 44px, 44px 44px, auto;
      overflow: hidden;
    }
    .twin3d-stage canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
    }
    .twin3d-stage canvas:active {
      cursor: grabbing;
    }
    .twin3d-label {
      position: absolute;
      transform: translate(-50%, -50%);
      padding: 6px 8px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      background: rgba(10, 14, 18, 0.78);
      color: #dbe7f1;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.02em;
      pointer-events: none;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.24);
      white-space: nowrap;
    }
    .twin3d-label.pass {
      border-color: rgba(34, 197, 94, 0.45);
      color: #c8f7d9;
    }
    .twin3d-label.review {
      border-color: rgba(245, 158, 11, 0.5);
      color: #ffe0a3;
    }
    .twin3d-label.reject {
      border-color: rgba(239, 68, 68, 0.52);
      color: #ffc9c9;
    }
    .twin3d-fallback {
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 28px;
      color: #dbe7f1;
      background: linear-gradient(145deg, #111820, #0a0f14);
      text-align: center;
      font-size: 14px;
      line-height: 1.5;
    }
    .twin3d-monitor {
      border-left: 1px solid rgba(255, 255, 255, 0.1);
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.055), rgba(255, 255, 255, 0.025));
      padding: 14px;
      overflow: hidden;
    }
    .monitor-card {
      border: 1px solid rgba(255, 255, 255, 0.13);
      border-radius: 10px;
      background: rgba(7, 10, 13, 0.54);
      padding: 12px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.055);
    }
    .monitor-card + .monitor-card {
      margin-top: 10px;
    }
    .monitor-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 10px;
      color: #f8fafc;
      font-size: 12px;
      font-weight: 760;
      letter-spacing: 0.07em;
      text-transform: uppercase;
    }
    .decision-badge {
      padding: 5px 8px;
      border-radius: 8px;
      color: #0a1117;
      background: #94a3b8;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.04em;
    }
    .decision-badge.pass {
      background: #22c55e;
    }
    .decision-badge.review {
      background: #f59e0b;
    }
    .decision-badge.reject {
      background: #ef4444;
      color: #fff7f7;
    }
    .readout-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .readout-cell {
      min-height: 54px;
      padding: 8px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.045);
    }
    .readout-cell span {
      display: block;
      color: #93a2b2;
      font-size: 10px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .readout-cell strong {
      display: block;
      margin-top: 5px;
      color: #f8fafc;
      font-size: 13px;
      line-height: 16px;
      word-break: break-word;
    }
    .thumb-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 9px;
      margin-top: 9px;
    }
    .thumb {
      min-height: 110px;
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.015)),
        #10161b;
      overflow: hidden;
    }
    .thumb img {
      display: none;
      width: 100%;
      height: 84px;
      object-fit: cover;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .thumb.has-image img {
      display: block;
    }
    .thumb span {
      display: block;
      padding: 7px;
      color: #b7c4cf;
      font-size: 10px;
      line-height: 13px;
      word-break: break-word;
    }
    .empty-state {
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      width: min(360px, calc(100% - 48px));
      padding: 16px 18px;
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 10px;
      background: rgba(8, 12, 16, 0.78);
      color: #d6e3ec;
      text-align: center;
      font-size: 13px;
      line-height: 1.45;
      pointer-events: none;
      box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
    }
    .twin3d-transport {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 0 18px 10px;
      background: rgba(255, 255, 255, 0.025);
    }
    .twin3d-btn {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 7px 14px;
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.08);
      color: #e2e8f0;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.03em;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s, box-shadow 0.15s;
      font-family: inherit;
    }
    .twin3d-btn:hover {
      background: rgba(255, 255, 255, 0.14);
      border-color: rgba(255, 255, 255, 0.3);
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.18);
    }
    .twin3d-btn.active {
      background: rgba(56, 189, 248, 0.18);
      border-color: rgba(56, 189, 248, 0.45);
    }
    .twin3d-btn .icon { font-size: 14px; }
    .twin3d-speed-label {
      color: #93a2b2;
      font-size: 11px;
      margin-left: 12px;
    }
    .twin3d-speed-slider {
      width: 100px;
      accent-color: #38bdf8;
    }
    @media (max-width: 900px) {
      .twin3d-shell {
        height: auto;
      }
      .twin3d-topbar {
        align-items: flex-start;
        flex-direction: column;
      }
      .twin3d-stats {
        justify-content: flex-start;
      }
      .twin3d-workspace {
        grid-template-columns: 1fr;
        height: auto;
      }
      .twin3d-stage {
        height: 430px;
      }
      .twin3d-monitor {
        border-left: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
      }
    }
  </style>

  <div class="twin3d-topbar">
    <div>
      <div class="twin3d-title">3D Packaging Line Digital Twin</div>
      <div class="twin3d-subtitle">Upload pair -> YOLO front/back -> OCR -> route package</div>
    </div>
    <div class="twin3d-stats">
      <div class="twin3d-pill"><strong id="countLive">0</strong><span>Live</span></div>
      <div class="twin3d-pill"><strong id="countPass">0</strong><span>Pass</span></div>
      <div class="twin3d-pill"><strong id="countReview">0</strong><span>Review</span></div>
      <div class="twin3d-pill"><strong id="countReject">0</strong><span>Reject</span></div>
    </div>
  </div>

  <div class="twin3d-transport">
    <button class="twin3d-btn active" id="btnPlay" onclick="window._twinPlay()"><span class="icon">▶</span> Play</button>
    <button class="twin3d-btn" id="btnPause" onclick="window._twinPause()"><span class="icon">⏸</span> Pause</button>
    <button class="twin3d-btn" id="btnReplay" onclick="window._twinReplay()"><span class="icon">🔄</span> Replay</button>
    <span class="twin3d-speed-label">Speed:</span>
    <input type="range" class="twin3d-speed-slider" id="speedSlider" min="0.3" max="3.0" step="0.1" value="__SPEED__" oninput="window._twinSetSpeed(parseFloat(this.value))">
    <span class="twin3d-speed-label" id="speedVal">__SPEED__x</span>
  </div>

  <div class="twin3d-workspace">
    <div class="twin3d-stage" id="sceneWrap">
      <canvas id="threeScene"></canvas>
      <div class="twin3d-label" style="left:24%; top:18%;">Camera 1</div>
      <div class="twin3d-label" style="left:52%; top:18%;">Camera 2 + OCR</div>
      <div class="twin3d-label review" style="left:84%; top:15%;">Manual review</div>
      <div class="twin3d-label reject" style="left:84%; top:52%;">Reject bin</div>
      <div class="empty-state" id="emptyState">Run AI inspection to send one uploaded package through the 3D line.</div>
      <div class="twin3d-fallback" id="fallback">3D renderer could not start. Check that the browser can load Three.js, then refresh the Streamlit page.</div>
    </div>

    <aside class="twin3d-monitor">
      <div class="monitor-card">
        <div class="monitor-title">
          <span>Live package</span>
          <span id="decisionBadge" class="decision-badge">WAITING</span>
        </div>
        <div class="readout-grid">
          <div class="readout-cell"><span>Package</span><strong id="pkgId">Waiting</strong></div>
          <div class="readout-cell"><span>Front YOLO</span><strong id="frontStatus">N/A</strong></div>
          <div class="readout-cell"><span>Back YOLO</span><strong id="backStatus">N/A</strong></div>
          <div class="readout-cell"><span>OCR</span><strong id="ocrStatus">N/A</strong></div>
          <div class="readout-cell"><span>PatchCore</span><strong id="patchScore">N/A</strong></div>
          <div class="readout-cell"><span>Route</span><strong id="routeText">Idle</strong></div>
        </div>
      </div>

      <div class="monitor-card">
        <div class="monitor-title"><span>Uploaded views</span></div>
        <div class="thumb-row">
          <div class="thumb" id="frontThumb"><img id="frontImg" alt=""><span id="frontName">Frontside</span></div>
          <div class="thumb" id="backThumb"><img id="backImg" alt=""><span id="backName">Backside</span></div>
        </div>
      </div>

      <div class="monitor-card">
        <div class="monitor-title"><span>Line logic</span></div>
        <div class="readout-cell" style="min-height:92px;">
          <span>Current rule</span>
          <strong id="logicText">Structural defect rejects. OCR unreadable routes to manual review when YOLO passes.</strong>
        </div>
      </div>
    </aside>
  </div>

  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
  <script>
    const events = __EVENTS__;
    let speed = __SPEED__;
    const wrap = document.getElementById("sceneWrap");
    const canvas = document.getElementById("threeScene");
    const fallback = document.getElementById("fallback");
    const emptyState = document.getElementById("emptyState");
    const counts = { PASS: 0, MANUAL_REVIEW: 0, REJECT: 0 };
    const active = [];
    let cursor = 0;
    let lastSpawn = 0;
    let lastTime = 0;
    let paused = false;
    let getDuration = () => Math.max(4.8, 8.8 / Math.max(0.3, speed));
    let getSpawnGap = () => Math.max(1.0, 2.4 / Math.max(0.3, speed));

    // Transport controls
    window._twinPlay = function() {
      paused = false;
      document.getElementById('btnPlay').classList.add('active');
      document.getElementById('btnPause').classList.remove('active');
    };
    window._twinPause = function() {
      paused = true;
      document.getElementById('btnPause').classList.add('active');
      document.getElementById('btnPlay').classList.remove('active');
    };
    window._twinReplay = function() {
      // Clear all active packages
      for (const item of active) {
        if (item.group && item.group.parent) item.group.parent.remove(item.group);
      }
      active.length = 0;
      cursor = 0;
      lastSpawn = 0;
      counts.PASS = 0; counts.MANUAL_REVIEW = 0; counts.REJECT = 0;
      paused = false;
      document.getElementById('btnPlay').classList.add('active');
      document.getElementById('btnPause').classList.remove('active');
      updateCounters();
    };
    window._twinSetSpeed = function(val) {
      speed = val;
      document.getElementById('speedVal').textContent = val.toFixed(1) + 'x';
    };

    function cleanAction(action) {
      return ["PASS", "MANUAL_REVIEW", "REJECT"].includes(action) ? action : "PASS";
    }

    function actionClass(action) {
      if (action === "REJECT") return "reject";
      if (action === "MANUAL_REVIEW") return "review";
      if (action === "PASS") return "pass";
      return "";
    }

    function setText(id, value) {
      const el = document.getElementById(id);
      if (el) el.textContent = value === undefined || value === null || value === "" ? "N/A" : value;
    }

    function setThumb(side, src, name) {
      const box = document.getElementById(side + "Thumb");
      const img = document.getElementById(side + "Img");
      const label = document.getElementById(side + "Name");
      if (!box || !img || !label) return;
      label.textContent = name || (side === "front" ? "Frontside" : "Backside");
      if (src) {
        img.src = src;
        box.classList.add("has-image");
      } else {
        img.removeAttribute("src");
        box.classList.remove("has-image");
      }
    }

    function updateReadout(event) {
      const action = cleanAction(event.final_action);
      setText("pkgId", event.package_id || "PKG");
      setText("frontStatus", event.front_yolo_status || "N/A");
      setText("backStatus", event.back_yolo_status || "N/A");
      setText("ocrStatus", event.ocr_state || "N/A");
      if (event.patchcore_score === null || event.patchcore_score === undefined) {
        setText("patchScore", event.patchcore_text || "N/A");
      } else {
        setText("patchScore", Number(event.patchcore_score).toFixed(3));
      }
      setText("routeText", action === "MANUAL_REVIEW" ? "MANUAL REVIEW" : action);
      const badge = document.getElementById("decisionBadge");
      badge.textContent = action === "MANUAL_REVIEW" ? "REVIEW" : action;
      badge.className = "decision-badge " + actionClass(action);
      setThumb("front", event.front_image, event.front_image_name);
      setThumb("back", event.back_image, event.back_image_name);
    }

    function updateCounters() {
      setText("countLive", active.length);
      setText("countPass", counts.PASS);
      setText("countReview", counts.MANUAL_REVIEW);
      setText("countReject", counts.REJECT);
      emptyState.style.display = events.length ? "none" : "block";
    }

    function routePoint(progress, action, event) {
      const x = -5.8 + progress * 11.6;
      let z = 0;
      const rejectStart = event.reject_stage === "FRONT_CAMERA" ? 0.40 : 0.72;
      if (action === "REJECT" && progress > rejectStart) {
        const t = Math.min(1, (progress - rejectStart) / (1 - rejectStart));
        z = 1.55 * t;
      }
      if (action === "MANUAL_REVIEW" && progress > 0.72) {
        const t = Math.min(1, (progress - 0.72) / 0.28);
        z = -1.55 * t;
      }
      return { x, z };
    }

    function showFallback() {
      fallback.style.display = "flex";
      emptyState.style.display = "none";
    }

    if (!window.THREE) {
      showFallback();
    } else {
      const ThreeLib = window.THREE;
      const scene = new ThreeLib.Scene();
      scene.background = new ThreeLib.Color(0x0b0f13);
      scene.fog = new ThreeLib.Fog(0x0b0f13, 9, 23);

      const camera = new ThreeLib.PerspectiveCamera(42, 1, 0.1, 100);
      camera.position.set(0, 5.6, 8.6);
      camera.lookAt(0, 0, 0);

      const renderer = new ThreeLib.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
      });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = ThreeLib.PCFSoftShadowMap;

      const ambient = new ThreeLib.HemisphereLight(0xeaf6ff, 0x171b1f, 1.45);
      scene.add(ambient);

      const keyLight = new ThreeLib.DirectionalLight(0xffffff, 2.4);
      keyLight.position.set(-3.5, 7.4, 4.5);
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.width = 2048;
      keyLight.shadow.mapSize.height = 2048;
      scene.add(keyLight);

      const rimLight = new ThreeLib.PointLight(0x49d1ff, 1.35, 16);
      rimLight.position.set(3.6, 3.4, -3.5);
      scene.add(rimLight);

      const materials = {
        floor: new ThreeLib.MeshStandardMaterial({ color: 0x10151a, roughness: 0.74, metalness: 0.08 }),
        belt: new ThreeLib.MeshStandardMaterial({ color: 0x20262c, roughness: 0.54, metalness: 0.22 }),
        beltEdge: new ThreeLib.MeshStandardMaterial({ color: 0x56616b, roughness: 0.36, metalness: 0.5 }),
        metal: new ThreeLib.MeshStandardMaterial({ color: 0x9ca8b3, roughness: 0.3, metalness: 0.62 }),
        darkMetal: new ThreeLib.MeshStandardMaterial({ color: 0x29313a, roughness: 0.38, metalness: 0.46 }),
        blue: new ThreeLib.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.32, metalness: 0.24 }),
        camera: new ThreeLib.MeshStandardMaterial({ color: 0xd9e4ef, roughness: 0.34, metalness: 0.25 }),
        lens: new ThreeLib.MeshStandardMaterial({ color: 0x0f1720, roughness: 0.18, metalness: 0.72 }),
        packBase: new ThreeLib.MeshStandardMaterial({ color: 0xe9edf2, roughness: 0.42, metalness: 0.12 }),
        foil: new ThreeLib.MeshStandardMaterial({ color: 0xb8c1ca, roughness: 0.22, metalness: 0.58 }),
        pill: new ThreeLib.MeshStandardMaterial({ color: 0xf7fafc, roughness: 0.26, metalness: 0.05 }),
        pass: new ThreeLib.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.34, metalness: 0.1 }),
        review: new ThreeLib.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.34, metalness: 0.08 }),
        reject: new ThreeLib.MeshStandardMaterial({ color: 0xef4444, roughness: 0.34, metalness: 0.08 }),
        glowPass: new ThreeLib.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.15 }),
        glowReview: new ThreeLib.MeshBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.18 }),
        glowReject: new ThreeLib.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.18 }),
        beam: new ThreeLib.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.16, depthWrite: false, side: ThreeLib.DoubleSide }),
      };

      function makeBox(w, h, d, mat, x, y, z, cast = true, receive = true) {
        const mesh = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(w, h, d), mat);
        mesh.position.set(x, y, z);
        mesh.castShadow = cast;
        mesh.receiveShadow = receive;
        scene.add(mesh);
        return mesh;
      }

      function makeCyl(radius, length, mat, x, y, z, rotateZ = true) {
        const mesh = new ThreeLib.Mesh(new ThreeLib.CylinderGeometry(radius, radius, length, 32), mat);
        if (rotateZ) mesh.rotation.z = Math.PI / 2;
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
        return mesh;
      }

      makeBox(15, 0.08, 7.5, materials.floor, 0, -0.08, 0, false, true);
      makeBox(12.8, 0.22, 1.28, materials.belt, 0, 0.08, 0, false, true);
      makeBox(12.8, 0.28, 0.12, materials.beltEdge, 0, 0.27, -0.78, true, true);
      makeBox(12.8, 0.28, 0.12, materials.beltEdge, 0, 0.27, 0.78, true, true);
      makeBox(3.5, 0.16, 1.05, materials.glowReview, 4.1, 0.12, -1.52, false, false);
      makeBox(3.5, 0.16, 1.05, materials.glowReject, 4.1, 0.12, 1.52, false, false);
      makeBox(2.8, 0.12, 0.92, materials.glowPass, 5.2, 0.15, 0, false, false);

      const rollers = [];
      for (let x = -5.8; x <= 5.8; x += 1.15) {
        rollers.push(makeCyl(0.14, 1.52, materials.darkMetal, x, 0.24, 0));
      }

      const slats = [];
      for (let i = 0; i < 22; i += 1) {
        const slat = makeBox(0.08, 0.018, 1.16, materials.metal, -6.1 + i * 0.58, 0.36, 0, false, true);
        slats.push(slat);
      }

      function makeStation(x, titleColor) {
        const group = new ThreeLib.Group();
        const post1 = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(0.12, 1.6, 0.12), materials.metal);
        const post2 = post1.clone();
        post1.position.set(-0.5, 0.95, -0.95);
        post2.position.set(0.5, 0.95, -0.95);
        const bridge = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(1.35, 0.16, 0.16), materials.metal);
        bridge.position.set(0, 1.72, -0.95);
        const body = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(0.7, 0.34, 0.42), materials.camera);
        body.position.set(0, 1.52, -0.42);
        const lens = new ThreeLib.Mesh(new ThreeLib.CylinderGeometry(0.16, 0.16, 0.16, 28), materials.lens);
        lens.rotation.x = Math.PI / 2;
        lens.position.set(0, 1.42, -0.16);
        const accent = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(0.76, 0.04, 0.045), titleColor);
        accent.position.set(0, 1.73, -0.34);
        group.add(post1, post2, bridge, body, lens, accent);
        group.position.x = x;
        scene.add(group);
        return group;
      }

      const camera1 = makeStation(-2.65, materials.blue);
      const camera2 = makeStation(0.65, materials.review);

      function makeBeam(x) {
        const beam = new ThreeLib.Mesh(new ThreeLib.ConeGeometry(0.82, 1.52, 4, 1, true), materials.beam);
        beam.rotation.x = Math.PI;
        beam.rotation.y = Math.PI / 4;
        beam.position.set(x, 0.88, -0.08);
        scene.add(beam);
        return beam;
      }

      const beam1 = makeBeam(-2.65);
      const beam2 = makeBeam(0.65);

      const gatePivot = new ThreeLib.Group();
      const gatePost = new ThreeLib.Mesh(new ThreeLib.CylinderGeometry(0.11, 0.11, 1.1, 28), materials.darkMetal);
      gatePost.position.set(2.85, 0.72, -0.85);
      const gateArm = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(1.35, 0.08, 0.12), materials.reject);
      gateArm.position.set(0.64, 0.95, 0);
      gatePivot.position.set(2.85, 0.2, -0.85);
      gatePivot.add(gateArm);
      scene.add(gatePost, gatePivot);

      function makeBin(x, z, mat) {
        const group = new ThreeLib.Group();
        const base = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(1.2, 0.15, 1.0), mat);
        const back = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(1.2, 0.66, 0.09), mat);
        const left = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(0.09, 0.66, 1.0), mat);
        const right = left.clone();
        base.position.y = 0.08;
        back.position.set(0, 0.42, 0.46);
        left.position.set(-0.56, 0.42, 0);
        right.position.set(0.56, 0.42, 0);
        group.add(base, back, left, right);
        group.position.set(x, 0.02, z);
        group.traverse((obj) => {
          if (obj.isMesh) {
            obj.castShadow = true;
            obj.receiveShadow = true;
          }
        });
        scene.add(group);
        return group;
      }

      makeBin(5.25, -1.68, materials.review);
      makeBin(5.25, 1.68, materials.reject);

      function createPackage(event) {
        const group = new ThreeLib.Group();
        const action = cleanAction(event.final_action);
        const base = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(0.92, 0.08, 0.56), materials.packBase);
        base.position.y = 0.06;
        const foil = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(0.8, 0.04, 0.44), materials.foil);
        foil.position.y = 0.13;
        const colorMat = action === "PASS" ? materials.pass : action === "MANUAL_REVIEW" ? materials.review : materials.reject;
        const stripe = new ThreeLib.Mesh(new ThreeLib.BoxGeometry(0.92, 0.035, 0.08), colorMat);
        stripe.position.set(0, 0.175, -0.18);
        group.add(base, foil, stripe);
        for (let row = 0; row < 2; row += 1) {
          for (let col = 0; col < 4; col += 1) {
            const pill = new ThreeLib.Mesh(new ThreeLib.SphereGeometry(0.095, 20, 12), materials.pill);
            pill.scale.set(1.2, 0.35, 0.82);
            pill.position.set(-0.3 + col * 0.2, 0.2, -0.05 + row * 0.17);
            pill.castShadow = true;
            group.add(pill);
          }
        }
        group.position.set(-5.8, 0.45, 0);
        group.rotation.y = 0;
        group.userData.event = event;
        group.userData.action = action;
        group.userData.counted = false;
        scene.add(group);
        return group;
      }

      function spawn(now) {
        if (!events.length || cursor >= events.length) return;
        const event = events[cursor];
        cursor += 1;
        const group = createPackage(event);
        active.push({ group, event, action: cleanAction(event.final_action), start: now, counted: false });
        updateReadout(event);
      }

      function resize() {
        const rect = wrap.getBoundingClientRect();
        const width = Math.max(320, rect.width);
        const height = Math.max(320, rect.height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
      }

      let drag = false;
      let dragX = 0;
      let yaw = 0;
      const baseCamera = new ThreeLib.Vector3(0, 5.6, 8.6);
      canvas.addEventListener("pointerdown", (event) => {
        drag = true;
        dragX = event.clientX;
        canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener("pointermove", (event) => {
        if (!drag) return;
        yaw += (event.clientX - dragX) * 0.004;
        yaw = Math.max(-0.55, Math.min(0.55, yaw));
        dragX = event.clientX;
      });
      canvas.addEventListener("pointerup", () => {
        drag = false;
      });

      function updateCamera() {
        const radius = baseCamera.z;
        camera.position.x = Math.sin(yaw) * radius;
        camera.position.z = Math.cos(yaw) * radius;
        camera.position.y = baseCamera.y;
        camera.lookAt(0.3, 0.25, 0);
      }

      function animate(nowMs) {
        const now = nowMs / 1000;
        const dt = lastTime ? now - lastTime : 0;
        lastTime = now;

        // Always render (camera drag works while paused) but skip sim when paused
        if (!paused) {
          const spawnGap = getSpawnGap();
          if ((!lastSpawn || now - lastSpawn > spawnGap) && cursor < events.length) {
            spawn(now);
            lastSpawn = now;
          }

          rollers.forEach((roller) => {
            roller.rotation.x += dt * 6.8 * speed;
          });
          slats.forEach((slat) => {
            slat.position.x += dt * 1.45 * speed;
            if (slat.position.x > 6.2) slat.position.x = -6.2;
          });

          const duration = getDuration();
          let cam1Active = false;
          let cam2Active = false;
          let gateActive = false;

          for (let i = active.length - 1; i >= 0; i -= 1) {
            const item = active[i];
            const progress = Math.min(1, (now - item.start) / duration);
            const point = routePoint(progress, item.action, item.event);
            item.group.position.x = point.x;
            item.group.position.z = point.z;
            item.group.position.y = 0.45 + Math.sin(now * 7 + i) * 0.01;
            item.group.rotation.y = ThreeLib.MathUtils.lerp(item.group.rotation.y, point.z * -0.18, 0.08);

            if (Math.abs(progress - 0.27) < 0.055) cam1Active = true;
            if (item.event.reject_stage !== "FRONT_CAMERA" && Math.abs(progress - 0.55) < 0.055) cam2Active = true;
            if (
              (item.event.reject_stage === "FRONT_CAMERA" && progress > 0.34 && progress < 0.53) ||
              (item.event.reject_stage !== "FRONT_CAMERA" && progress > 0.69 && progress < 0.83 && item.action !== "PASS")
            ) {
              gateActive = true;
            }

            if (progress >= 1) {
              if (!item.counted) {
                counts[item.action] += 1;
                item.counted = true;
              }
              scene.remove(item.group);
              active.splice(i, 1);
            }
          }

          camera1.position.y = cam1Active ? 0.06 + Math.sin(now * 28) * 0.018 : 0;
          camera2.position.y = cam2Active ? 0.06 + Math.sin(now * 28) * 0.018 : 0;
          materials.beam.opacity = cam1Active || cam2Active ? 0.42 : 0.1;
          beam1.scale.setScalar(cam1Active ? 1.18 + Math.sin(now * 18) * 0.06 : 1.0);
          beam2.scale.setScalar(cam2Active ? 1.18 + Math.sin(now * 18) * 0.06 : 1.0);
          gatePivot.rotation.y = ThreeLib.MathUtils.lerp(gatePivot.rotation.y, gateActive ? -0.9 : 0.08, 0.1);
          updateCounters();
        }

        updateCamera();
        renderer.render(scene, camera);
        requestAnimationFrame(animate);
      }

      window.addEventListener("resize", resize);
      resize();
      updateCounters();
      requestAnimationFrame(animate);
    }
  </script>
</div>
""".replace("__EVENTS__", events_json).replace("__SPEED__", speed_json)


def render_digital_twin_panel(ocr_data: List[Dict], patch_scores: List[Dict], patch_threshold: Dict) -> None:
    st.subheader("Digital Twin: Conveyor Inspection Simulation")
    st.caption("Upload one frontside and one backside image; the app runs YOLO + OCR and sends that package through the line.")

    if DIGITAL_TWIN_QUEUE_KEY not in st.session_state:
        st.session_state[DIGITAL_TWIN_QUEUE_KEY] = []

    controls, preview = st.columns([0.34, 0.66])
    with controls:
        st.markdown("**Live AI Inspection**")
        front_upload = st.file_uploader("Frontside image", type=["jpg", "jpeg", "png"], key="frontside_upload")
        back_upload = st.file_uploader("Backside image", type=["jpg", "jpeg", "png"], key="backside_upload")
        package_id = st.text_input(
            "Package ID",
            value="PKG-LIVE-001",
        )

        with st.expander("Model settings", expanded=False):
            front_weights = st.text_input("Front YOLO weights", value=str(DEFAULT_FRONT_YOLO_WEIGHTS))
            back_weights = st.text_input("Back YOLO weights", value=str(DEFAULT_BACK_YOLO_WEIGHTS))
            conf_threshold = st.slider("YOLO confidence threshold", 0.05, 0.90, 0.25, 0.05)
            st.caption("PatchCore live scoring is used only when the uploaded filename matches an entry in scores.json.")

        inspect_disabled = front_upload is None or back_upload is None
        if st.button("🔍 Run AI Inspection", disabled=inspect_disabled, use_container_width=True, type="primary"):
            try:
                with st.status("Running AI inspection pipeline...", expanded=True) as status:
                    def _progress(msg):
                        status.update(label=msg)
                    event = run_uploaded_package_inspection(
                        front_upload=front_upload,
                        back_upload=back_upload,
                        package_id=package_id.strip() or "PKG-LIVE-001",
                        front_weights=front_weights,
                        back_weights=back_weights,
                        conf_threshold=conf_threshold,
                        patch_scores=patch_scores,
                        patch_threshold=patch_threshold,
                        status_callback=_progress,
                    )
                    status.update(label=f"✅ Complete — {event['final_action']}", state="complete")
                st.session_state[DIGITAL_TWIN_QUEUE_KEY] = [event]
                action = event["final_action"]
                if action == "PASS":
                    st.success(f"AI decision: **{action}** ✓")
                elif action == "REJECT":
                    st.error(f"AI decision: **{action}** ✗")
                else:
                    st.warning(f"AI decision: **{action}** ⚠")
            except Exception as exc:
                st.error(f"Inspection failed: {exc}")

        # Play Demo button: generate sample events from PatchCore scores
        if st.button("▶ Play Demo (Sample Batch)", use_container_width=True):
            demo_events = build_sample_events(ocr_data, patch_scores, patch_threshold, limit=10)
            st.session_state[DIGITAL_TWIN_QUEUE_KEY] = demo_events
            st.info(f"Loaded {len(demo_events)} demo packages. Watch them route through the 3D line below!")

        col_clear, col_save = st.columns(2)
        with col_clear:
            if st.button("🗑 Clear Line", use_container_width=True):
                st.session_state[DIGITAL_TWIN_QUEUE_KEY] = []
        with col_save:
            active_events = st.session_state[DIGITAL_TWIN_QUEUE_KEY]
            if st.button("💾 Save JSON", use_container_width=True):
                export_payload = [event_for_export(event) for event in active_events]
                Path("inspection_events.json").write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
                st.success("Saved inspection_events.json")

    with preview:
        front_col, back_col = st.columns(2)
        with front_col:
            if front_upload is not None:
                st.image(front_upload, caption=f"Frontside: {front_upload.name}", use_container_width=True)
            else:
                st.empty()
        with back_col:
            if back_upload is not None:
                st.image(back_upload, caption=f"Backside: {back_upload.name}", use_container_width=True)
            else:
                st.empty()

        active_events = st.session_state[DIGITAL_TWIN_QUEUE_KEY]
        event_counts = (
            pd.Series([event.get("final_action", "PASS") for event in active_events]).value_counts()
            if active_events
            else pd.Series(dtype=int)
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Queued", len(active_events))
        m2.metric("Pass", int(event_counts.get("PASS", 0)))
        m3.metric("Review", int(event_counts.get("MANUAL_REVIEW", 0)))
        m4.metric("Reject", int(event_counts.get("REJECT", 0)))

        if active_events:
            event = active_events[0]
            st.dataframe(
                pd.DataFrame([event_for_export(event)]),
                use_container_width=True,
                height=130,
            )
        else:
            st.info("Upload a frontside and backside image, then run AI inspection to animate one package.")

    active_events = st.session_state[DIGITAL_TWIN_QUEUE_KEY]
    # Speed is now controlled inside the 3D twin's transport bar
    components.html(render_digital_twin_3d_html(active_events, 1.0), height=810, scrolling=False)


def render_live_camera_panels(ocr_data: List[Dict]) -> None:
    """Show YOLO + OCR results only (2 columns)."""
    event = None
    if DIGITAL_TWIN_QUEUE_KEY in st.session_state and st.session_state[DIGITAL_TWIN_QUEUE_KEY]:
        event = st.session_state[DIGITAL_TWIN_QUEUE_KEY][0]

    cam1, cam2 = st.columns(2)
    with cam1:
        st.subheader("Camera 1: Frontside YOLO")
        if event:
            render_data_uri_image(
                event.get("front_annotated_image", ""),
                f"{event.get('front_yolo_status', 'N/A')} - {event.get('front_yolo_summary', '')}",
            )
            st.json(
                {
                    "status": event.get("front_yolo_status"),
                    "summary": event.get("front_yolo_summary"),
                    "detections": event.get("front_yolo_detections", []),
                }
            )
        else:
            st.info("Run AI inspection above to show the boxed frontside YOLO result here.")

    with cam2:
        st.subheader("Camera 2: Backside YOLO + OCR")
        if event:
            render_data_uri_image(
                event.get("back_annotated_image", ""),
                f"{event.get('back_yolo_status', 'N/A')} - {event.get('back_yolo_summary', '')}",
            )
            st.json(
                {
                    "back_yolo_status": event.get("back_yolo_status"),
                    "back_yolo_summary": event.get("back_yolo_summary"),
                    "ocr_state": event.get("ocr_state"),
                    "lot": event.get("lot_detected"),
                    "exp": event.get("exp_detected"),
                    "message": event.get("ocr_message"),
                }
            )
        else:
            st.info("Run AI inspection above to show the boxed backside YOLO result and OCR state here.")


def render_patchcore_analysis() -> None:
    """Separate PatchCore analysis section with its own Run button."""
    st.subheader("🔥 PatchCore Anomaly Analysis")
    st.caption("Run deep anomaly detection on the last inspected package. This uses unsupervised PatchCore models to generate heatmap overlays.")

    event = None
    if DIGITAL_TWIN_QUEUE_KEY in st.session_state and st.session_state[DIGITAL_TWIN_QUEUE_KEY]:
        event = st.session_state[DIGITAL_TWIN_QUEUE_KEY][0]

    if event is None or not event.get("front_path") or not event.get("back_path"):
        st.info("Run AI Inspection first, then come here to run PatchCore analysis.")
        return

    threshold = event.get("patchcore_threshold", 0.5)

    if st.button("🔬 Run PatchCore Analysis", use_container_width=True, type="primary"):
        with st.spinner("Running PatchCore on both sides in parallel..."):
            t0 = time.time()
            # Pre-load outside the executor to avoid cache locks
            res_front = load_patchcore_model("front")
            res_back = load_patchcore_model("back")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fut_front = pool.submit(run_patchcore_live, Path(event["front_path"]), "front", threshold, res_front)
                fut_back = pool.submit(run_patchcore_live, Path(event["back_path"]), "back", threshold, res_back)
                patch_front_result = fut_front.result()
                patch_back_result = fut_back.result()
            elapsed = time.time() - t0

        # Store results in the event so they persist on rerun
        event["patchcore_front_score"] = patch_front_result.get("score")
        event["patchcore_front_text"] = f"Live: {patch_front_result['score']:.4f}" if patch_front_result.get("score") is not None else "N/A"
        event["patchcore_front_heatmap"] = patch_front_result.get("heatmap", "")
        event["patchcore_score"] = patch_back_result.get("score")
        event["patchcore_text"] = f"Live: {patch_back_result['score']:.4f}" if patch_back_result.get("score") is not None else "N/A"
        event["patchcore_heatmap"] = patch_back_result.get("heatmap", "")
        st.success(f"PatchCore analysis complete in {elapsed:.1f}s")

    # Display results if they exist
    pc_front, pc_back = st.columns(2)
    with pc_front:
        st.markdown("#### Frontside Heatmap")
        pc_score = event.get("patchcore_front_score")
        pc_heatmap = event.get("patchcore_front_heatmap", "")
        if pc_score is not None:
            if pc_score >= threshold:
                st.error(f"⚠ Anomaly — score: **{pc_score:.4f}** (threshold: {threshold:.4f})")
            else:
                st.success(f"✓ Normal — score: **{pc_score:.4f}** (threshold: {threshold:.4f})")
        if pc_heatmap:
            render_data_uri_image(pc_heatmap, "Frontside anomaly heatmap")
        elif pc_score is None:
            st.info("Click 'Run PatchCore Analysis' above.")

    with pc_back:
        st.markdown("#### Backside Heatmap")
        pc_score = event.get("patchcore_score")
        pc_heatmap = event.get("patchcore_heatmap", "")
        if pc_score is not None:
            if pc_score >= threshold:
                st.error(f"⚠ Anomaly — score: **{pc_score:.4f}** (threshold: {threshold:.4f})")
            else:
                st.success(f"✓ Normal — score: **{pc_score:.4f}** (threshold: {threshold:.4f})")
        if pc_heatmap:
            render_data_uri_image(pc_heatmap, "Backside anomaly heatmap")
        elif pc_score is None:
            st.info("Click 'Run PatchCore Analysis' above.")


def main() -> None:
    st.title("Pharmaceutical Packaging Defect Monitoring")
    st.caption("YOLOv8 + OCR + PatchCore + Federated Learning + OEE")

    with st.sidebar:
        st.header("Data Inputs")
        ocr_path = st.text_input("OCR results", value="test_results/ocr/ocr_results.json")
        patch_scores_path = st.text_input("PatchCore scores", value="test_results/patchcore_back/scores.json")
        patch_threshold_path = st.text_input("PatchCore threshold", value="test_results/patchcore_back/threshold.json")
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
    render_digital_twin_panel(ocr_data, patch_scores, patch_threshold)

    st.markdown("---")
    render_live_camera_panels(ocr_data)

    st.markdown("---")
    render_patchcore_analysis()

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


            # ROC curve
            roc_path = Path("patchcore_results/roc_curve.png")
            if roc_path.exists():
                st.markdown("#### 📊 ROC Curve & Score Distribution")
                st.image(str(roc_path), use_container_width=True)

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
    render_interactive_simpy_panel(oee_data)


def run_simpy_inline(cycle_time: float, defect_prob: float, downtime_prob: float, duration_sec: int) -> Dict:
    """Run SimPy packaging line simulation inline (fast — <1s)."""
    import simpy as _simpy

    _random = random.Random(42)
    env = _simpy.Environment()

    stats = {"total": 0, "good": 0, "reject": 0, "downtime_sec": 0.0}

    def _line():
        while True:
            if _random.random() < downtime_prob:
                stop = _random.uniform(12, 30)
                stats["downtime_sec"] += stop
                yield env.timeout(stop)
            yield env.timeout(cycle_time)
            stats["total"] += 1
            if _random.random() < defect_prob:
                stats["reject"] += 1
            else:
                stats["good"] += 1

    env.process(_line())
    env.run(until=duration_sec)

    runtime = max(1e-6, duration_sec - stats["downtime_sec"])
    ideal_output = runtime / cycle_time
    availability = runtime / max(1e-6, duration_sec)
    performance = stats["total"] / max(1e-6, ideal_output)
    quality = stats["good"] / max(1, stats["total"])
    oee = availability * performance * quality

    return {
        "simulated_seconds": duration_sec,
        "total_count": stats["total"],
        "good_count": stats["good"],
        "reject_count": stats["reject"],
        "throughput_per_hour": round((stats["total"] / max(1e-6, duration_sec)) * 3600),
        "defect_rate": round((stats["reject"] / max(1, stats["total"])) * 100, 2),
        "availability_percent": round(availability * 100, 2),
        "performance_percent": round(performance * 100, 2),
        "quality_percent": round(quality * 100, 2),
        "oee_percent": round(oee * 100, 2),
    }


def render_interactive_simpy_panel(default_oee: Dict) -> None:
    st.subheader("⚙️ SimPy / OEE — Interactive Production Simulator")
    st.caption("Adjust conveyor belt speed, defect rate, and downtime to see how OEE changes in real time.")

    # Parameter controls
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        cycle_time = st.slider(
            "Cycle time (sec/pack)",
            min_value=0.3,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="How fast the conveyor moves one package through. Lower = faster belt.",
        )
    with p2:
        defect_prob = st.slider(
            "Defect probability (%)",
            min_value=1,
            max_value=20,
            value=7,
            step=1,
            help="Chance each package has a defect.",
        )
    with p3:
        downtime_prob = st.slider(
            "Downtime probability (%)",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Chance of an unplanned stop per cycle.",
        )
    with p4:
        sim_duration = st.slider(
            "Duration (minutes)",
            min_value=5,
            max_value=30,
            value=15,
            step=1,
        )

    # Run simulation
    sim_result = run_simpy_inline(
        cycle_time=cycle_time,
        defect_prob=defect_prob / 100.0,
        downtime_prob=downtime_prob / 100.0,
        duration_sec=sim_duration * 60,
    )

    # Deltas from default saved values
    def _delta(key, suffix=""):
        current = sim_result.get(key, 0)
        default = default_oee.get(key, 0)
        diff = current - default
        if abs(diff) < 0.01:
            return None
        return f"{diff:+.1f}{suffix}"

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Throughput/hr", f"{sim_result['throughput_per_hour']:,}", delta=_delta("throughput_per_hour"))
    m2.metric("Defect Rate", f"{sim_result['defect_rate']}%", delta=_delta("defect_rate", "%"), delta_color="inverse")
    m3.metric("OEE", f"{sim_result['oee_percent']}%", delta=_delta("oee_percent", "%"))
    m4.metric("Total Produced", f"{sim_result['total_count']:,}")
    m5.metric("Rejects", f"{sim_result['reject_count']}", delta=_delta("reject_count"), delta_color="inverse")

    # OEE breakdown chart
    oee_col, detail_col = st.columns([0.55, 0.45])
    with oee_col:
        st.markdown("##### OEE Breakdown")
        breakdown_df = pd.DataFrame({
            "Component": ["Availability", "Performance", "Quality", "OEE"],
            "Percent": [
                sim_result["availability_percent"],
                sim_result["performance_percent"],
                sim_result["quality_percent"],
                sim_result["oee_percent"],
            ],
        })
        st.bar_chart(breakdown_df.set_index("Component"), height=280)

    with detail_col:
        st.markdown("##### Simulation Details")
        detail_df = pd.DataFrame([
            {"Parameter": "Cycle time", "Value": f"{cycle_time:.1f} sec"},
            {"Parameter": "Defect prob.", "Value": f"{defect_prob}%"},
            {"Parameter": "Downtime prob.", "Value": f"{downtime_prob}%"},
            {"Parameter": "Duration", "Value": f"{sim_duration} min"},
            {"Parameter": "Good count", "Value": f"{sim_result['good_count']:,}"},
            {"Parameter": "Reject count", "Value": f"{sim_result['reject_count']}"},
            {"Parameter": "Availability", "Value": f"{sim_result['availability_percent']}%"},
            {"Parameter": "Performance", "Value": f"{sim_result['performance_percent']}%"},
            {"Parameter": "Quality", "Value": f"{sim_result['quality_percent']}%"},
        ])
        st.dataframe(detail_df, use_container_width=True, hide_index=True, height=320)


if __name__ == "__main__":
    main()
