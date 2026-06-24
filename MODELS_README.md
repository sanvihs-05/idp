# Model Weights — Setup

The trained model weights are shared directly (not stored in this repository).
Place the `test_results` folder you received into the **root of this project**
(the same folder that contains this README).

## 📁 Folder structure after placing

```
IDP/
├── test_results/
│   ├── yolo_front/
│   │   └── weights/
│   │       └── best.onnx        ← YOLO front-side blister detector
│   ├── yolo_back/
│   │   └── weights/
│   │       └── best.onnx        ← YOLO back-side blister detector
│   ├── patchcore_front/
│   │   └── run_20260507_122351/anomalib_logs/Patchcore/blister/v0/weights/lightning/
│   │       └── model.ckpt       ← PatchCore anomaly model (front)
│   └── patchcore_back/
│       └── run_20260507_122023/anomalib_logs/Patchcore/blister/v0/weights/lightning/
│           └── model.ckpt       ← PatchCore anomaly model (back)
├── streamlit_dashboard.py
└── ...
```

## ▶️ Running the dashboard

```bash
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

The dashboard automatically loads the YOLO and PatchCore models from the
`test_results/` paths above. No extra config needed.

## 🔁 Retraining from scratch

If you want to regenerate the models instead of using the shared weights:

```bash
python train_yolov8.py          # trains yolo_front
python train_backside_yolov8.py # trains yolo_back
python train_patchcore.py       # trains both PatchCore models
```

This will recreate the `test_results/` folder with the same structure.
