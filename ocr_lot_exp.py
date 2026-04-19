"""
OCR Validation Pipeline — Camera 2: Backside Blister Pack
==================================================================================
This module runs post-detection (after YOLOv8 flags a 'good_pack').
It uses EasyOCR to extract LOT and EXP text, then validates them using regex.
"""

import re
import cv2
import json
import easyocr
import argparse
from pathlib import Path
from datetime import datetime

# Initialize EasyOCR reader (loads PyTorch models into GPU if available)
# Doing this at the module level so it's loaded once for the whole pipeline
try:
    print("⏳ Loading EasyOCR model (this may take a few seconds on first run)...")
    reader = easyocr.Reader(['en'], gpu=True)
    print("✅ EasyOCR initialized successfully.")
except Exception as e:
    print(f"⚠️ Failed to init EasyOCR: {e}")
    reader = None

# ──────────────────────────────────────────────────────────────────────────────
# Generic Regex Patterns (Adjustable)
# ──────────────────────────────────────────────────────────────────────────────
# Example target matches: "LOT: ABC12345" or "LOT 1234XYZ"
LOT_PATTERN = re.compile(r"LOT[:\s]*([A-Z0-9]+)", re.IGNORECASE)

# Example target matches: "EXP: 12/2026", "EXP 2026-12", "EXP: 12-25"
EXP_PATTERN = re.compile(r"EXP[:\s]*(\d{2,4}[/-]\d{2,4})", re.IGNORECASE)


def parse_expiry_date(date_str: str) -> datetime | None:
    """Attempts to parse an extracted EXP string into a datetime object."""
    # Common formats: MM/YYYY, MM/YY, YYYY-MM
    date_str = date_str.replace("-", "/")
    
    formats_to_try = ["%m/%Y", "%m/%y", "%Y/%m"]
    
    for fmt in formats_to_try:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
            
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
        result_data["message"] = "EasyOCR reader not initialized."
        return result_data

    if not Path(image_path).exists():
        result_data["message"] = f"Image not found: {image_path}"
        return result_data

    # 1. Run EasyOCR
    raw_results = reader.readtext(str(image_path))
    
    # Extract just the text from the results list of (bbox, text, confidence)
    extracted_texts = [res[1] for res in raw_results]
    full_text = " ".join(extracted_texts).upper()
    
    if not full_text:
        result_data["message"] = "No text detected in image."
        return result_data

    result_data["raw_text"] = full_text
    print(f"\n📄 Extracted Text: '{full_text}'")

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
            res = validate_lot_exp(str(source_path))
            results_list.append(res)
            print(f"Status: {res['status']} | Message: {res['message']}")
        elif source_path.is_dir():
            for img in source_path.glob("*.jpg"):
                res = validate_lot_exp(str(img))
                results_list.append(res)
                print(f"[{res['image_name']}] Status: {res['status']} | Message: {res['message']}")
                
        # Save to JSON
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=4)
        print(f"\n💾 Saved structured OCR results to {args.output}")
    else:
        print("Usage: python ocr_lot_exp.py path/to/image.jpg")
        print("Provide an image from the larger-blister-pack-defect validation set to test.")
