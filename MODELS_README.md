# Model Weights — Download & Setup

The trained model weights are **not stored in this repository** (they are large binary
files and don't version well in git). Download the bundled `idp_models_bundle.zip`
from the link below and unzip it **into the repository root** — the archive preserves
the exact folder structure the code expects, so every file lands in the right place.

## 📥 Download

> **Models zip:** _<PASTE YOUR GOOGLE DRIVE / ONEDRIVE SHARE LINK HERE>_

## 📦 How to install

1. Download `idp_models_bundle.zip`.
2. Unzip it at the **root of this project** (the folder containing this README), keeping
   the folder structure. On Windows you can right‑click → *Extract All…* into the repo
   folder, or from a terminal at the repo root:
   ```bash
   unzip idp_models_bundle.zip -d .
   ```
3. Confirm the files landed at the paths listed below.

## 🗂️ What's in the bundle

| Model | Path (relative to repo root) | Purpose |
|-------|------------------------------|---------|
| YOLOv8n base weights | `yolov8n.pt` | Pretrained YOLOv8 nano base used as the starting point for training. |
| YOLO26n base weights | `yolo26n.pt` | Alternate YOLO base weights. |
| YOLO front (trained) | `test_results/yolo_front/weights/best.onnx` | Trained front‑side blister detector (exported ONNX). |
| YOLO back (trained) | `test_results/yolo_back/weights/best.onnx` | Trained back‑side blister detector (exported ONNX). |
| PatchCore front | `test_results/patchcore_front/run_20260507_122351/anomalib_logs/Patchcore/blister/v0/weights/lightning/model.ckpt` | Anomalib PatchCore anomaly model (front side). |
| PatchCore back | `test_results/patchcore_back/run_20260507_122023/anomalib_logs/Patchcore/blister/v0/weights/lightning/model.ckpt` | Anomalib PatchCore anomaly model (back side). |
| Federated global model | `fl_results_10round_cpu/global_round_10_fedavg.pt` | Final FedAvg global YOLO model after 10 federated rounds. |

> **Note:** The 150+ intermediate per‑round / per‑site training checkpoints are *not*
> included — they are throwaway artifacts produced during federated training and are not
> needed to run inference. Only the final/usable models are bundled.

## ▶️ Using the models

- **Detection + anomaly dashboard:** run `streamlit run streamlit_dashboard.py` — it loads
  the trained YOLO and PatchCore weights from the paths above.
- **Retraining from scratch:** see `train_yolov8.py`, `train_backside_yolov8.py`,
  `train_patchcore.py`, and `federated_learning_sim.py`. These regenerate the weights and
  will recreate the same folder structure under `test_results/` and `fl_results_*/`.

If you move the project out of OneDrive (recommended for git/training performance), unzip
the models again at the new repo root.
