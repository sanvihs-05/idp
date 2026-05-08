import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["FLAGS_use_mkldnn"] = "0"
from ocr_lot_exp import validate_lot_exp
import glob

imgs = glob.glob("Larger-Blister-Pack-Defect--1/valid/images/blister_good*.jpg")
if imgs:
    res = validate_lot_exp(imgs[0])
    print(f"Status: {res['status']}, LOT: {res.get('lot_detected')}, Message: {res.get('message')}")
else:
    print("No test images found")
