"""
Prepare PatchCore data directory structure for both frontside and backside models.
"""
import shutil
import glob
from pathlib import Path

DST = Path("patchcore_data")

# --- BACKSIDE DATA (from Larger-Blister-Pack-Defect--1) ---
SRC_BACK = Path("Larger-Blister-Pack-Defect--1")
dst_back = DST / "back"
if dst_back.exists():
    shutil.rmtree(dst_back)

train_good_back = dst_back / "train" / "good"
test_good_back  = dst_back / "test" / "good"
test_defect_back = dst_back / "test" / "defect"

train_good_back.mkdir(parents=True)
test_good_back.mkdir(parents=True)
test_defect_back.mkdir(parents=True)

count = 0
for img in (SRC_BACK / "train" / "images").glob("blister_good_*.jpg"):
    shutil.copy2(img, train_good_back / img.name)
    count += 1
print(f"Backside Train (good): {count} images")

count = 0
for img in (SRC_BACK / "valid" / "images").glob("blister_good_*.jpg"):
    shutil.copy2(img, test_good_back / img.name)
    count += 1
print(f"Backside Test (good): {count} images")

count = 0
for img in (SRC_BACK / "valid" / "images").glob("blister_defect_*.jpg"):
    shutil.copy2(img, test_defect_back / img.name)
    count += 1
print(f"Backside Test (defect): {count} images")

# --- FRONTSIDE DATA (from BLISTER-1) ---
SRC_FRONT = Path("BLISTER-1")
dst_front = DST / "front"
if dst_front.exists():
    shutil.rmtree(dst_front)

train_good_front = dst_front / "train" / "good"
test_good_front  = dst_front / "test" / "good"
test_defect_front = dst_front / "test" / "defect"

train_good_front.mkdir(parents=True)
test_good_front.mkdir(parents=True)
test_defect_front.mkdir(parents=True)

def process_frontside_split(split_name, good_dir, defect_dir=None):
    good_cnt = 0
    defect_cnt = 0
    for label_file in (SRC_FRONT / split_name / "labels").glob("*.txt"):
        with open(label_file) as f:
            labels = [int(line.split()[0]) for line in f]
        img_name = label_file.name.replace(".txt", ".jpg")
        img_path = SRC_FRONT / split_name / "images" / img_name
        
        # Classes 2 and 3 are defects ("defect", "no_pill")
        if 2 in labels or 3 in labels:
            if defect_dir is not None:
                shutil.copy2(img_path, defect_dir / img_name)
                defect_cnt += 1
        else:
            shutil.copy2(img_path, good_dir / img_name)
            good_cnt += 1
    return good_cnt, defect_cnt

g, _ = process_frontside_split("train", train_good_front)
print(f"Frontside Train (good): {g} images")

g, d = process_frontside_split("valid", test_good_front, test_defect_front)
print(f"Frontside Test (good): {g} images")
print(f"Frontside Test (defect): {d} images")

print("\nPatchCore data prepared at:", DST.resolve())
