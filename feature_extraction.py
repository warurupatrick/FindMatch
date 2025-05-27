import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pytesseract
import pickle
from skimage.feature import local_binary_pattern
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from ultralytics import YOLO
import re

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Autoencoder for deep color embedding ---
class ColorAutoencoder(nn.Module):
    def __init__(self):
        super(ColorAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # B, 16, 64, 64
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # B, 32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # B, 64, 16, 16
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128)
        )

    def forward(self, x):
        return self.encoder(x)

# --- EfficientNet-based Car Feature Extractor ---
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(model.children())[:-1])  # Remove classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)

# --- Main Vehicle Recognizer Class ---
class VehicleRecognizer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.feature_extractor = FeatureExtractor().to(device).eval()
        self.color_autoencoder = ColorAutoencoder().to(device).eval()
        self.plate_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.plate_model = YOLO(r"C:\Users\REGINAH\Desktop\ml\models\best.pt") 

    def extract_car_model_embedding(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor(img_tensor).cpu().numpy().flatten()
            return features
        except Exception as e:
            print(f"Error extracting car model embedding for {img_path}: {str(e)}")
            return None

    def extract_license_plate_text(self, img_path):
        image = cv2.imread(img_path)
        if image is None:
            return "Unknown"

        results = self.plate_model.predict(source=image, conf=0.4, iou=0.5, save=False, verbose=False)
        if not results or len(results[0].boxes.xyxy) == 0:
            return "Unknown"

        x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        plate_img = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        raw_text = pytesseract.image_to_string(
            thresh,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            lang='eng'
        )

        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
        return cleaned if cleaned else "Unknown"

    def extract_license_plate_embedding(self, plate_text):
        return self.plate_embedder.encode(plate_text).tolist()

    def extract_color_histogram(self, img_path, bins=(8, 8, 8)):
        img = cv2.imread(img_path)
        if img is None:
            return [0] * (bins[0] * bins[1] * bins[2])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        hist = cv2.calcHist([lab], [0, 1, 2], None, bins,
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist.astype(float).tolist()

    def extract_deep_color_embedding(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor()
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.color_autoencoder(img_tensor).cpu().numpy().flatten()
            return embedding.tolist()
        except Exception as e:
            print(f"Error extracting deep color embedding for {img_path}: {str(e)}")
            return [0] * 128

    def extract_texture_embedding(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(26).tolist()

        img = cv2.resize(img, (128, 128))
        lbp = local_binary_pattern(img, P=24, R=3, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist.tolist()

    def process_dataset(self, dataset_path, output_path):
        features = {}
        for img_name in os.listdir(dataset_path):
            img_path = os.path.join(dataset_path, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            car_model_embedding = self.extract_car_model_embedding(img_path)
            if car_model_embedding is None:
                continue

            license_plate_text = self.extract_license_plate_text(img_path)
            license_plate_embedding = self.extract_license_plate_embedding(license_plate_text)
            color_hist_embedding = self.extract_color_histogram(img_path)
            deep_color_embedding = self.extract_deep_color_embedding(img_path)
            texture_embedding = self.extract_texture_embedding(img_path)

            features[img_name] = {
                "image_id": img_name,
                "car_model_embedding": car_model_embedding,
                "license_plate_text": license_plate_text,
                "license_plate_embedding": license_plate_embedding,
                "color_embedding": color_hist_embedding + deep_color_embedding,  # Combined
                "texture_embedding": texture_embedding
            }

            print(f"Processed {img_name}")

        with open(output_path, 'wb') as f:
            pickle.dump(features, f)

        print(f"\n✅ Saved features for {len(features)} images to: {output_path}")
        
        


if __name__ == "__main__":
    recognizer = VehicleRecognizer()
    BASE_DIR = r"C:\Users\REGINAH\Desktop\cv"
    DATASET_PATH = r"C:\Users\REGINAH\Desktop\cv\cropped"
    EMBEDDINGS_PATH = r"C:\Users\REGINAH\Desktop\ml\embeddings"
    os.makedirs(EMBEDDINGS_PATH, exist_ok=True)
    OUTPUT_PKL = os.path.join(EMBEDDINGS_PATH, "vehicle_features.pkl")
    recognizer.process_dataset(DATASET_PATH, OUTPUT_PKL)
