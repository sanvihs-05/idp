"""
YOLOv8-Nano Training Pipeline — Camera 2: Backside Blister Pack Defect Detection
==================================================================================
Dataset : Roboflow  ·  my-workspace-d5mot / larger-blister-pack-defect
Classes : defect, good_pack, no_pack
Model   : YOLOv8-nano (yolov8n.pt)
"""

import os
import sys
import argparse
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
ROBOFLOW_API_KEY = "dUfTkSVwWSjYZvCdhHAN"
ROBOFLOW_WORKSPACE = "my-workspace-d5mot"          # public workspace
ROBOFLOW_PROJECT  = "larger-blister-pack-defect"
ROBOFLOW_VERSION  = 1

TRAIN_EPOCHS   = 50
IMAGE_SIZE     = 640
BATCH_SIZE     = 4       # 4 GB VRAM on RTX 3050
PATIENCE       = 10      # early-stopping patience

# Auto-detect GPU
import torch
if torch.cuda.is_available():
    DEVICE = "0"       # NVIDIA GeForce RTX 3050 Laptop GPU
    print(f"🖥️  GPU detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("⚠️  No CUDA GPU detected — training on CPU (will be slower)")
PROJECT_DIR    = "C:/temp/pharma_defect"            # outside OneDrive
RUN_NAME       = "yolov8n_backside"


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download Dataset from Roboflow
# ──────────────────────────────────────────────────────────────────────────────
def download_dataset() -> str:
    """Download the dataset in YOLOv8 format and return the path to data.yaml."""
    from roboflow import Roboflow

    print("\n📦  Downloading backside dataset from Roboflow …")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download("yolov8")

    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"✅  Dataset downloaded → {dataset.location}")
    print(f"    data.yaml → {data_yaml}")

    # Quick sanity check — print the classes in the dataset
    import yaml
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"    Classes ({cfg.get('nc', '?')}): {cfg.get('names', [])}")

    return data_yaml


# ──────────────────────────────────────────────────────────────────────────────
# 2. Train YOLOv8-Nano
# ──────────────────────────────────────────────────────────────────────────────
def train(data_yaml: str):
    """Run YOLOv8-nano training and return the model object."""
    from ultralytics import YOLO

    print("\n🚀  Starting YOLOv8-nano training (backside model) …")
    print(f"    Epochs : {TRAIN_EPOCHS}")
    print(f"    ImgSz  : {IMAGE_SIZE}")
    print(f"    Batch  : {BATCH_SIZE}")
    print(f"    Device : {DEVICE}")
    print(f"    Output : {PROJECT_DIR}/{RUN_NAME}\n")

    model = YOLO("yolov8n.pt")  # downloads automatically if not cached

    results = model.train(
        data=data_yaml,
        epochs=TRAIN_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=PATIENCE,
        augment=True,
        verbose=True,
    )

    print(f"\n✅  Training complete → {PROJECT_DIR}/{RUN_NAME}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 3. Evaluate
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model):
    """Validate the trained model and print key metrics."""
    print("\n📊  Running validation …")
    metrics = model.val()

    print("\n── Results ─────────────────────────────────")
    print(f"   mAP@50      : {metrics.box.map50:.4f}")
    print(f"   mAP@50-95   : {metrics.box.map:.4f}")
    print(f"   Precision   : {metrics.box.mp:.4f}")
    print(f"   Recall      : {metrics.box.mr:.4f}")
    print("────────────────────────────────────────────\n")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 4. Export to ONNX (for edge / embedded inference)
# ──────────────────────────────────────────────────────────────────────────────
def export_onnx(model):
    """Export the trained model to ONNX format."""
    print("📦  Exporting model to ONNX …")
    onnx_path = model.export(format="onnx")
    print(f"✅  ONNX model saved → {onnx_path}")
    return onnx_path


# ──────────────────────────────────────────────────────────────────────────────
# 5. Quick Inference Demo
# ──────────────────────────────────────────────────────────────────────────────
def run_inference(model, source: str | None = None):
    """Run inference on a sample image or directory."""
    if source is None:
        val_dir = Path(PROJECT_DIR) / RUN_NAME
        source = str(val_dir)
        print(f"\n🔍  No source provided — running on validation images in {source}")
    else:
        print(f"\n🔍  Running inference on: {source}")

    results = model.predict(source=source, save=True, conf=0.25, imgsz=IMAGE_SIZE)
    print(f"✅  Predictions saved under {PROJECT_DIR}/{RUN_NAME}/predict\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8-Nano — Camera 2: Backside Blister Pack Defect Detection"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip dataset download (use existing local data.yaml)",
    )
    parser.add_argument(
        "--data-yaml", type=str, default=None,
        help="Path to an existing data.yaml (implies --skip-download)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (only evaluate / export an existing model)",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to trained weights (.pt) for eval/export without retraining",
    )
    parser.add_argument(
        "--export-onnx", action="store_true", default=True,
        help="Export to ONNX after training (default: True)",
    )
    parser.add_argument(
        "--infer", type=str, default=None,
        help="Run inference on a file or directory after training",
    )
    args = parser.parse_args()

    # ── Step 1: Dataset ──────────────────────────────────────────────────
    if args.data_yaml:
        data_yaml = args.data_yaml
        print(f"📂  Using provided data.yaml → {data_yaml}")
    elif args.skip_download:
        data_yaml = f"{ROBOFLOW_PROJECT}-{ROBOFLOW_VERSION}/data.yaml"
        print(f"📂  Skipping download — using → {data_yaml}")
    else:
        data_yaml = download_dataset()

    # ── Step 2: Train ────────────────────────────────────────────────────
    if args.skip_train:
        from ultralytics import YOLO
        weights = args.weights or f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
        print(f"⏩  Skipping training — loading weights → {weights}")
        model = YOLO(weights)
    else:
        model = train(data_yaml)

    # ── Step 3: Evaluate ─────────────────────────────────────────────────
    evaluate(model)

    # ── Step 4: Export ───────────────────────────────────────────────────
    if args.export_onnx:
        export_onnx(model)

    # ── Step 5: Inference (optional) ─────────────────────────────────────
    if args.infer:
        run_inference(model, source=args.infer)

    print("\n🎉  Camera 2 backside model — all done!")


if __name__ == "__main__":
    main()
