import os
import pickle
import numpy as np
from pinecone import Pinecone, ServerlessSpec

# --- Step 1: Initialize Pinecone ---
api_key = "pcsk_42kwJk_GY99ZrVeeZd4DQYHxhctxCeJqcwRLJthVMikL4jiPUfbvvhxjabmTEdSAreJj4V"
pc = Pinecone(api_key=api_key)

# --- Step 2: Create the index if it doesn't exist ---
index_name = "vehicle-features"
dimension = 1693  # 1280 (model) + 384 (license plate) + 3 (color) + 26 (texture)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"✅ Index '{index_name}' created")

# --- Step 3: Connect to the index ---
index = pc.Index(index_name)

# --- Step 4: Load embeddings from pickle file ---
pkl_path = r"C:\Users\PATO\OneDrive\Desktop\ml\embeddings\vehicle_features.pkl"
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# --- Step 5: Prepare and upsert vectors with shape checking ---
vectors = []
for image_id, feats in data.items():
    try:
        car_model = np.array(feats["car_model_embedding"], dtype=np.float32)
        plate_embed = np.array(feats["license_plate_embedding"], dtype=np.float32)
        color = np.array(feats["color_embedding"], dtype=np.float32)
        texture = np.array(feats["texture_embedding"], dtype=np.float32)

        # Validate shapes
        if car_model.shape[0] != 1280:
            raise ValueError(f"❌ Car model embedding shape wrong: {car_model.shape}")
        if plate_embed.shape[0] != 384:
            raise ValueError(f"❌ Plate embedding shape wrong: {plate_embed.shape}")
        if color.shape[0] != 3:
            raise ValueError(f"❌ Color embedding shape wrong: {color.shape}")
        if texture.shape[0] != 26:
            raise ValueError(f"❌ Texture embedding shape wrong: {texture.shape}")

        # Combine into single vector
        vector = np.concatenate([car_model, plate_embed, color, texture]).tolist()

        # Add to upload list
        vectors.append({
            "id": image_id,
            "values": vector,
            "metadata": {
                "license_plate": feats["license_plate_text"]
            }
        })

    except Exception as e:
        print(f"❌ Failed to process {image_id}: {e}")

# --- Step 6: Upload in batches ---
batch_size = 100
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i + batch_size]
    index.upsert(vectors=batch)
    print(f"✅ Uploaded batch {i // batch_size + 1}")

print("🎉 All valid embeddings uploaded to Pinecone!")
