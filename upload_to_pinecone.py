import os
import pickle
import numpy as np
from pinecone import Pinecone, ServerlessSpec

def sanitize_id(image_id: str) -> str:
    # Optional: sanitize IDs to remove problematic characters for Pinecone
    return image_id.replace(" ", "_").replace("\\", "_").replace("/", "_")

def main():
    # Step 1: Initialize Pinecone
    api_key = "pcsk_42kwJk_GY99ZrVeeZd4DQYHxhctxCeJqcwRLJthVMikL4jiPUfbvvhxjabmTEdSAreJj4V"
    pc = Pinecone(api_key=api_key)

    # Step 2: Create the index if it doesn't exist or delete if wrong dimension
    index_name = "vehicle-features"
    dimension = 2330  # 1280 + 384 + 640 + 26

    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        # Connect to index to check dimension or just delete and recreate (safe)
        print(f"Deleting old index '{index_name}' to match new dimension ({dimension})")
        pc.delete_index(index_name)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Index '{index_name}' created")

    # Step 3: Connect to the index
    index = pc.Index(index_name)

    # Step 4: Load embeddings from pickle file
    pkl_path = r"C:\Users\REGINAH\Desktop\ml\embeddings\vehicle_features.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Step 5: Prepare and upsert vectors with thorough checks
    vectors = []
    for image_id, feats in data.items():
        try:
            required_keys = [
                "car_model_embedding",
                "license_plate_embedding",
                "color_embedding",
                "texture_embedding"
            ]
            if not all(key in feats for key in required_keys):
                raise KeyError(f"Missing keys in features for {image_id}")

            car_model = np.array(feats["car_model_embedding"], dtype=np.float32)
            plate_embed = np.array(feats["license_plate_embedding"], dtype=np.float32)
            color = np.array(feats["color_embedding"], dtype=np.float32)
            texture = np.array(feats["texture_embedding"], dtype=np.float32)

            # Validate sizes and shapes
            if car_model.size == 0 or car_model.shape[0] != 1280:
                raise ValueError(f"Car model embedding missing or wrong shape: {car_model.shape}")
            if plate_embed.size == 0 or plate_embed.shape[0] != 384:
                raise ValueError(f"Plate embedding missing or wrong shape: {plate_embed.shape}")
            if color.size == 0 or color.shape[0] != 640:
                raise ValueError(f"Color embedding missing or wrong shape: {color.shape}")
            if texture.size == 0 or texture.shape[0] != 26:
                raise ValueError(f"Texture embedding missing or wrong shape: {texture.shape}")

            vector = np.concatenate([car_model, plate_embed, color, texture]).tolist()

            if len(vector) != dimension:
                raise ValueError(f"Combined vector length mismatch: {len(vector)}")

            vectors.append({
                "id": sanitize_id(image_id),
                "values": vector,
                "metadata": {
                    "license_plate": feats.get("license_plate_text", "")
                }
            })

        except Exception as e:
            print(f"Skipping {image_id} due to error: {e}")

    # Step 6: Upload in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i // batch_size + 1}")

    print("All valid embeddings uploaded to Pinecone.")

if __name__ == "__main__":
    main()
