"""
Prepare PatchCore data directory structure.
Copies good_pack images for training and good+defect for testing.
"""
import shutil
from pathlib import Path

SRC = Path("Larger-Blister-Pack-Defect--1")
DST = Path("patchcore_data")

# Clean previous runs
if DST.exists():
    shutil.rmtree(DST)

# Create MVTec-style directory structure
train_good = DST / "blister" / "train" / "good"
test_good  = DST / "blister" / "test" / "good"
test_defect = DST / "blister" / "test" / "defect"

train_good.mkdir(parents=True)
test_good.mkdir(parents=True)
test_defect.mkdir(parents=True)

# Training: all 474 good_pack images from train split
count = 0
for img in (SRC / "train" / "images").glob("blister_good_*.jpg"):
    shutil.copy2(img, train_good / img.name)
    count += 1
print(f"Train (good): {count} images")

# Test good: 50 good_pack from valid split
count = 0
for img in (SRC / "valid" / "images").glob("blister_good_*.jpg"):
    shutil.copy2(img, test_good / img.name)
    count += 1
print(f"Test (good): {count} images")

# Test defect: 49 defect from valid split
count = 0
for img in (SRC / "valid" / "images").glob("blister_defect_*.jpg"):
    shutil.copy2(img, test_defect / img.name)
    count += 1
print(f"Test (defect): {count} images")

print("\nPatchCore data prepared at:", DST.resolve())
