import os
import numpy as np
import cv2
from pinecone import Pinecone
from feature_extraction import VehicleRecognizer  # Your main feature extractor class

def main():
    # --- Step 1: Initialize Pinecone ---
    api_key = "pcsk_42kwJk_GY99ZrVeeZd4DQYHxhctxCeJqcwRLJthVMikL4jiPUfbvvhxjabmTEdSAreJj4V"
    pc = Pinecone(api_key=api_key)
    index = pc.Index("vehicle-features")

    # --- Step 2: Load feature extractor ---
    recognizer = VehicleRecognizer()

    # Path to the folder containing cropped images
    folder_path = r"C:\Users\REGINAH\Desktop\cv\cropped"

    # Get all image files in the folder (sorted alphabetically)
    image_files = sorted([
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(image_files) < 2:
        raise ValueError("Not enough images in the folder to separate query and dataset.")

    query_img_path = os.path.join(folder_path, image_files[-1])
    dataset_dir = [os.path.join(folder_path, file) for file in image_files[:-1]]

    print("Query image:", query_img_path)
    print("Dataset images:", dataset_dir)

    results_folder = r"C:\Users\REGINAH\Desktop\ml\results"
    os.makedirs(results_folder, exist_ok=True)
    log_file = r"C:\Users\REGINAH\Desktop\ml\log.txt"

    # --- Step 4: Extract features ---
    car_model_embedding = recognizer.extract_car_model_embedding(query_img_path)
    license_text = recognizer.extract_license_plate_text(query_img_path)
    plate_embedding = recognizer.extract_license_plate_embedding(license_text)
    color_histogram = recognizer.extract_color_histogram(query_img_path)  # 512D
    deep_color_embedding = recognizer.extract_deep_color_embedding(query_img_path)  # 128D
    texture_embedding = recognizer.extract_texture_embedding(query_img_path)

    try:
        car_model_embedding = np.array(car_model_embedding, dtype=np.float32)
        plate_embedding = np.array(plate_embedding, dtype=np.float32)
        color_histogram = np.array(color_histogram, dtype=np.float32)
        deep_color_embedding = np.array(deep_color_embedding, dtype=np.float32)
        texture_embedding = np.array(texture_embedding, dtype=np.float32)

        assert car_model_embedding.shape[0] == 1280
        assert plate_embedding.shape[0] == 384
        assert color_histogram.shape[0] == 512
        assert deep_color_embedding.shape[0] == 128
        assert texture_embedding.shape[0] == 26

        color_embedding = np.concatenate([color_histogram, deep_color_embedding])
        query_vector = np.concatenate([
            car_model_embedding,
            plate_embedding,
            color_embedding,
            texture_embedding
        ]).tolist()

        print("Final query vector shape:", len(query_vector))  # Should be 2330

        results = index.query(vector=query_vector, top_k=5, include_metadata=True)

        query_image = cv2.imread(query_img_path)

        with open(log_file, "w") as log:
            log.write("Query Image: " + query_img_path + "\n")
            log.write("Top 5 Matches:\n")

            for i, match in enumerate(results["matches"], 1):
                match_id = match["id"]
                score = match["score"]
                plate = match["metadata"].get("license_plate", "Unknown")

                match_full_path = os.path.join(folder_path, match_id)
                if not os.path.exists(match_full_path):
                    print(f"Warning: Match ID {match_id} not found in dataset directory")
                    continue

                match_image = cv2.imread(match_full_path)
                if match_image is None:
                    print(f"Warning: Could not read match image {match_id}")
                    continue

                height = 300
                query_resized = cv2.resize(query_image, (int(query_image.shape[1] * height / query_image.shape[0]), height))
                match_resized = cv2.resize(match_image, (int(match_image.shape[1] * height / match_image.shape[0]), height))

                cv2.putText(match_resized, f"Plate: {plate}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                comparison_image = cv2.hconcat([query_resized, match_resized])

                filename = f"comparison_{i}_{os.path.basename(match_id)}.jpg"
                save_path = os.path.join(results_folder, filename)
                cv2.imwrite(save_path, comparison_image)

                log.write(f"Match {i}: ID: {match_id} | Score: {score:.4f} | Plate: {plate}\n")
                print(f"Match {i}: ID: {match_id} | Score: {score:.4f} | Plate: {plate}")
                print(f"Saved comparison image as {save_path}")

        print(f"Matched comparison images saved to {results_folder}")
        print(f"Results logged to {log_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
