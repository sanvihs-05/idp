"""
PatchCore Anomaly Detection — Train & Evaluate
==================================================
Uses anomalib v2.4 with ResNet18 backbone on GPU.
Builds memory bank from 474 good blister pack images,
then evaluates on 50 good + 49 defect images.

Outputs:
  - patchcore_results/scores.json     (per-image anomaly scores)
  - patchcore_results/heatmaps/       (overlay heatmaps)
  - patchcore_results/roc_curve.png   (ROC with AUC)
  - patchcore_results/threshold.json  (optimal threshold)
"""

import json
import shutil
import stat
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

from torchvision.transforms.v2 import Resize
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

import argparse

parser = argparse.ArgumentParser(description="Train PatchCore")
parser.add_argument("--side", choices=["front", "back"], default="back", help="Side of blister pack to train on")
args = parser.parse_args()

# ─── Configuration ───────────────────────────────────────────────────────
DATA_ROOT   = Path(f"patchcore_data/{args.side}")
RESULTS_DIR = Path(f"test_results/patchcore_{args.side}")
RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR     = RESULTS_DIR / f"run_{RUN_ID}"
HEATMAP_DIR = RUN_DIR / "heatmaps"
CATEGORY    = "blister"
BACKBONE    = "resnet18"
DEVICE      = "gpu" if torch.cuda.is_available() else "cpu"

print(f"Training PatchCore for side: {args.side.upper()}")

print(f"Device: {DEVICE} ({'CUDA: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")


def _remove_readonly(func, path, exc_info):
    """Handle Windows read-only/symlink cleanup issues in rmtree."""
    try:
        Path(path).chmod(stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def _get_batch_field(batch, *names):
    """Compatibility helper for anomalib ImageBatch field naming."""
    for name in names:
        if isinstance(batch, dict) and name in batch:
            return batch[name]
        try:
            return batch[name]
        except Exception:
            pass
        if hasattr(batch, name):
            return getattr(batch, name)
    raise KeyError(f"None of the fields found in batch: {names}")

# ─── Setup output dirs ──────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# ─── Data Module (anomalib v2.4 API) ────────────────────────────────────
transform = Resize((256, 256), antialias=True)

datamodule = Folder(
    name=CATEGORY,
    root=DATA_ROOT,
    normal_dir="train/good",
    abnormal_dir="test/defect",
    normal_test_dir="test/good",
    augmentations=transform,
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=0,  # Windows compatibility
)

# ─── Model (anomalib v2.4 API) ──────────────────────────────────────────
model = Patchcore(
    backbone=BACKBONE,
    layers=["layer2", "layer3"],
    num_neighbors=9,
    coreset_sampling_ratio=0.01,
)

# ─── Train (builds memory bank) ─────────────────────────────────────────
print("\n" + "="*60)
print("Building PatchCore memory bank from 474 good images...")
print("="*60)

engine = Engine(
    accelerator="auto",
    devices=1,
    default_root_dir=str(RUN_DIR / "anomalib_logs"),
    max_epochs=1,
)

engine.fit(model=model, datamodule=datamodule)

# ─── Predict on test set ────────────────────────────────────────────────
print("\n" + "="*60)
print("Running inference on full validation mix (50 good + 49 defect)...")
print("="*60)

predictions = []
test_predictions = engine.predict(model=model, datamodule=datamodule)
predictions.extend(test_predictions)
# anomalib Folder may place part of mixed set in val_data, so include val predictions too.
try:
    val_loader = datamodule.val_dataloader()
    val_predictions = engine.predict(model=model, dataloaders=val_loader)
    predictions.extend(val_predictions)
except Exception:
    pass

# ─── Collect results ────────────────────────────────────────────────────
scores_list = []
all_scores = []
all_labels = []  # 0=good, 1=defect

for batch in predictions:
    image_paths = _get_batch_field(batch, "image_path", "image_paths")
    pred_scores = _get_batch_field(batch, "pred_score", "pred_scores").cpu().numpy()
    try:
        anomaly_maps = _get_batch_field(batch, "anomaly_map", "anomaly_maps").cpu().numpy()
    except KeyError:
        anomaly_maps = None
    labels = _get_batch_field(batch, "gt_label", "label").cpu().numpy()

    for i in range(len(image_paths)):
        img_path = Path(image_paths[i])
        score = float(pred_scores[i])
        true_label = int(labels[i])
        true_class = "defect" if true_label == 1 else "good"

        all_scores.append(score)
        all_labels.append(true_label)

        scores_list.append({
            "image_name": img_path.name,
            "anomaly_score": round(score, 4),
            "true_label": true_class,
        })

        # Save heatmap overlay
        if anomaly_maps is not None:
            heatmap = anomaly_maps[i].squeeze()
            heatmap_norm = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            orig_img = cv2.imread(str(img_path))
            if orig_img is not None:
                orig_resized = cv2.resize(orig_img, (heatmap_color.shape[1], heatmap_color.shape[0]))
                overlay = cv2.addWeighted(orig_resized, 0.5, heatmap_color, 0.5, 0)
                cv2.imwrite(str(HEATMAP_DIR / f"{true_class}_{img_path.stem}_heatmap.jpg"), overlay)

print(f"\nProcessed {len(scores_list)} images total")

# ─── Compute optimal threshold from ROC ─────────────────────────────────
all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

# Youden's J statistic for optimal threshold
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
optimal_threshold = float(thresholds[best_idx])

print(f"\nROC AUC: {roc_auc:.4f}")
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# ─── Assign predicted labels ────────────────────────────────────────────
pred_labels = (all_scores >= optimal_threshold).astype(int)
for i, entry in enumerate(scores_list):
    entry["predicted_label"] = "defect" if pred_labels[i] == 1 else "good"

# ─── Save scores JSON ───────────────────────────────────────────────────
with open(RUN_DIR / "scores.json", "w") as f:
    json.dump(scores_list, f, indent=4)
print(f"Scores saved to {RUN_DIR / 'scores.json'}")

# ─── Save threshold JSON ────────────────────────────────────────────────
threshold_data = {
    "optimal_threshold": round(optimal_threshold, 4),
    "roc_auc": round(roc_auc, 4),
    "total_test_images": len(scores_list),
    "good_count": int(np.sum(all_labels == 0)),
    "defect_count": int(np.sum(all_labels == 1)),
}
with open(RUN_DIR / "threshold.json", "w") as f:
    json.dump(threshold_data, f, indent=4)

# ─── Plot ROC Curve + Score Distribution ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(fpr, tpr, color="#4CAF50", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
axes[0].scatter(fpr[best_idx], tpr[best_idx], color="red", s=100, zorder=5,
               label=f"Optimal Threshold = {optimal_threshold:.3f}")
axes[0].set_xlabel("False Positive Rate", fontsize=12)
axes[0].set_ylabel("True Positive Rate", fontsize=12)
axes[0].set_title("PatchCore ROC Curve", fontsize=13, fontweight="bold")
axes[0].legend(loc="lower right", fontsize=11)
axes[0].grid(alpha=0.3)

good_scores = all_scores[all_labels == 0]
defect_scores = all_scores[all_labels == 1]
axes[1].hist(good_scores, bins=20, alpha=0.7, color="#4CAF50", label="Good", edgecolor="black")
axes[1].hist(defect_scores, bins=20, alpha=0.7, color="#f44336", label="Defect", edgecolor="black")
axes[1].axvline(x=optimal_threshold, color="orange", linestyle="--", lw=2, label=f"Threshold = {optimal_threshold:.3f}")
axes[1].set_xlabel("Anomaly Score", fontsize=12)
axes[1].set_ylabel("Count", fontsize=12)
axes[1].set_title("Anomaly Score Distribution", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(RUN_DIR / "roc_curve.png"), dpi=150, bbox_inches="tight")
print(f"ROC curve saved to {RUN_DIR / 'roc_curve.png'}")

# ─── Classification Report ──────────────────────────────────────────────
print("\n" + "="*60)
print("Classification Report (using optimal threshold)")
print("="*60)
print(classification_report(all_labels, pred_labels, target_names=["good", "defect"]))

cm = confusion_matrix(all_labels, pred_labels)
print("Confusion Matrix:")
print(cm)
print(f"\nAccuracy: {np.sum(np.diag(cm)) / np.sum(cm) * 100:.1f}%")
print(f"\nAll outputs saved to: {RUN_DIR.resolve()}")

# Also update dashboard-friendly "latest" copies.
for artifact in ["scores.json", "threshold.json", "roc_curve.png"]:
    src = RUN_DIR / artifact
    dst = RESULTS_DIR / artifact
    shutil.copy2(src, dst)

# Copy heatmaps to top-level directory so the dashboard can always find them.
latest_heatmap_dir = RESULTS_DIR / "heatmaps"
latest_heatmap_dir.mkdir(parents=True, exist_ok=True)
run_heatmaps = HEATMAP_DIR if HEATMAP_DIR.exists() else RUN_DIR / "heatmaps"
if run_heatmaps.exists():
    for hm in run_heatmaps.iterdir():
        if hm.is_file():
            shutil.copy2(hm, latest_heatmap_dir / hm.name)
    print(f"Copied {sum(1 for _ in latest_heatmap_dir.iterdir())} heatmaps to {latest_heatmap_dir.resolve()}")

print(f"Updated latest artifacts in: {RESULTS_DIR.resolve()}")
