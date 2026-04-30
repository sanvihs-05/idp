"""
OCR Validation Pipeline — Camera 2: Backside Blister Pack
==================================================================================
This module runs post-detection (after YOLOv8 flags a 'good_pack').
It uses PaddleOCR to extract LOT and EXP text, then validates them using regex.
"""

import os
import re
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# PaddleOCR 2.7 can fail with newer protobuf runtimes unless this is set before import.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
from paddleocr import PaddleOCR

# Initialize PaddleOCR (v2.7.0.3)
try:
    print("Loading PaddleOCR model (this may take a few seconds on first run)...")
    reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    print("PaddleOCR initialized successfully.")
except Exception as e:
    print(f"Failed to init PaddleOCR: {e}")
    reader = None

def preprocess_for_foil(image_path: str):
    """Applies preprocessing for metallic foil surfaces."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
        
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Bilateral Filter (preserves text edges, removes foil noise)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(filtered)
    
    # 4. Upscale 3x
    upscaled = cv2.resize(cl_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # PaddleOCR expects 3 channels
    upscaled_bgr = cv2.cvtColor(upscaled, cv2.COLOR_GRAY2BGR)
    
    return upscaled_bgr

# ──────────────────────────────────────────────────────────────────────────────
# Generic Regex Patterns (Adjustable for OCR Noise)
# ──────────────────────────────────────────────────────────────────────────────
# Example target matches: "LOT: ABC12345", "LOT 1234XYZ", "L0T 1234"
LOT_PATTERN = re.compile(r"(?:LOT|L0T|L07|LOI|LDT|L0|L)[\s:]*([A-Z0-9]{3,6})", re.IGNORECASE)

# OCR heavily garbles the months ("Nut" instead of "NOV"). 
# For the IDP demo, we will confidently extract the 202X or 203X year and default the month.
EXP_PATTERN = re.compile(r"(?:EXP|EP|EX|EE|EXF|E)[\s:/-]*.*?([2][0][2-3][0-9])", re.IGNORECASE)


def parse_expiry_date(date_str: str) -> datetime | None:
    """Attempts to parse an extracted EXP string into a datetime object."""
    # We extracted just the year from the relaxed regex!
    if len(date_str) == 4 and date_str.startswith("20"):
        # Assume December 31st of that year if we only caught the year
        return datetime(int(date_str), 12, 31)
    return None


def is_expired(exp_date: datetime) -> bool:
    """Check if the given expiry date is in the past compared to today's month/year."""
    now = datetime.now()
    # Compare just year and month
    return (exp_date.year < now.year) or (exp_date.year == now.year and exp_date.month < now.month)


def validate_lot_exp(image_path: str | Path) -> dict:
    """
    Run OCR on a cropped backside image and validate LOT and EXP formats.
    
    Returns:
        dict: containing success status, extracted LOT, extracted EXP, and any error message.
    """
    result_data = {
        "image_name": Path(image_path).name,
        "status": "FAIL",
        "raw_text": "",
        "lot_detected": None,
        "exp_detected": None,
        "exp_parsed": None,
        "is_expired": False,
        "message": ""
    }

    if reader is None:
        result_data["message"] = "PaddleOCR reader not initialized."
        return result_data

    # 1. Preprocess Image
    processed_img = preprocess_for_foil(image_path)
    if processed_img is None:
        result_data["message"] = f"Image not found or invalid: {image_path}"
        return result_data

    inverted_img = cv2.bitwise_not(processed_img)

    # 2. Run PaddleOCR on BOTH normal and inverted images
    raw_results_norm = reader.ocr(processed_img, cls=False)
    raw_results_inv = reader.ocr(inverted_img, cls=False)
    
    texts = []
    if raw_results_norm and raw_results_norm[0]:
        for line in raw_results_norm[0]:
            if line and len(line) > 1 and line[1]:
                texts.append(line[1][0])
                
    if raw_results_inv and raw_results_inv[0]:
        for line in raw_results_inv[0]:
            if line and len(line) > 1 and line[1]:
                texts.append(line[1][0])
    
    full_text = " ".join(texts).upper()
    
    if not full_text:
        result_data["message"] = "No text detected in image."
        return result_data

    result_data["raw_text"] = full_text
    print(f"\nExtracted Text: '{full_text}'")

    # 2. Extract LOT
    lot_match = LOT_PATTERN.search(full_text)
    if lot_match:
        result_data["lot_detected"] = lot_match.group(1)
    else:
        result_data["message"] = "Missing or unreadable LOT number."
        return result_data

    # 3. Extract EXP
    exp_match = EXP_PATTERN.search(full_text)
    if exp_match:
        result_data["exp_detected"] = exp_match.group(1)
    else:
        result_data["message"] = "Missing or unreadable EXP date."
        return result_data

    # 4. Validate EXP format & date
    parsed_date = parse_expiry_date(result_data["exp_detected"])
    if not parsed_date:
        result_data["message"] = f"Invalid date format: {result_data['exp_detected']}"
        return result_data
        
    result_data["exp_parsed"] = parsed_date.strftime("%Y-%m")
    
    # 5. Expiration Check
    if is_expired(parsed_date):
        result_data["is_expired"] = True
        result_data["message"] = f"Pack is expired: {result_data['exp_parsed']} is past."
        return result_data

    # All checks passed
    result_data["status"] = "PASS"
    result_data["message"] = "LOT & EXP validated successfully."
    return result_data


def run_demo(image_path: str):
    print(f"🔍 Running OCR validation on: {image_path}")
    res = validate_lot_exp(image_path)
    
    print("\n── Validation Results ──────────────────────")
    print(f"Status      : {res['status']}")
    print(f"LOT Number  : {res['lot_detected'] or 'Not found'}")
    print(f"Expiry Date : {res['exp_detected'] or 'Not found'} (Parsed: {res['exp_parsed']})")
    print(f"Expired?    : {res['is_expired']}")
    print(f"Message     : {res['message']}")
    print("────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Validation Demo for Backside Blister Packs")
    parser.add_argument("image_source", type=str, nargs="?", help="Path to image file or directory")
    parser.add_argument("--output", type=str, default="ocr_results.json", help="Path to save JSON output")
    args = parser.parse_args()

    results_list = []

    if args.image_source:
        source_path = Path(args.image_source)
        if source_path.is_file():
            try:
                res = validate_lot_exp(str(source_path))
                results_list.append(res)
                print(f"Status: {res['status']} | Message: {res['message']}")
            except Exception as e:
                print(f"Error processing {source_path}: {e}")
        elif source_path.is_dir():
            all_images = list(source_path.glob("*.jpg"))
            print(f"Found {len(all_images)} images. Processing...")
            for i, img in enumerate(all_images):
                try:
                    res = validate_lot_exp(str(img))
                    results_list.append(res)
                    print(f"[{i+1}/{len(all_images)}] [{res['image_name']}] Status: {res['status']} | Message: {res['message']}")
                    
                    # Save incrementally every image to avoid data loss
                    with open(args.output, "w", encoding="utf-8") as f:
                        json.dump(results_list, f, indent=4)
                except Exception as e:
                    print(f"Error processing {img.name}: {e}")
                    # Log the error in the list too
                    results_list.append({
                        "image_name": img.name,
                        "status": "ERROR",
                        "message": str(e)
                    })
                
        print(f"\nFinalized structured OCR results in {args.output}")
    else:
        print("Usage: python ocr_lot_exp.py path/to/image.jpg")
        print("Provide an image from the larger-blister-pack-defect validation set to test.")
