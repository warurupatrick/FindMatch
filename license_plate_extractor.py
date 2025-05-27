import numpy as np
import cv2
import pytesseract
import re
from ultralytics import YOLO

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLOv8 plate detection model
plate_model = YOLO(r"C:\Users\REGINAH\Desktop\ml\models\best.pt")

def extract_number_plate_text(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Could not read image. Check path:", image_path)
        return None

    # Run YOLOv8 to detect plate
    results = plate_model.predict(source=image, conf=0.3, iou=0.5, save=False, verbose=False)

    if not results or len(results[0].boxes.xyxy) == 0:
        print("No plate detected.")
        return None

    # Assume first box is plate
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    plate_img = image[y1:y2, x1:x2]

    # Convert to grayscale for OCR
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR
    raw_text = pytesseract.image_to_string(
        thresh,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        lang='eng'
    )

    # Clean text (only keep alphanumeric)
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    return cleaned if cleaned else None

# 🔍 Test it
image_path = r'C:\Users\user\Downloads\Licence-plate\License-plate\Car Images\33.png'
text = extract_number_plate_text(image_path)

if text:
    print("Number Plate Text:", text)
else:
    print("No number plate text detected.")
