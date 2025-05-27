import pickle

# Load the pickle file
pkl_path = r"C:\Users\REGINAH\Desktop\ml\embeddings\vehicle_features.pkl"
with open(pkl_path, 'rb') as f:
    features = pickle.load(f)

# Loop through all the images in the pickle file and display their embeddings
for image_name, embeddings in features.items():
    print(f"Embeddings for {image_name}:")
    print("Car Model Embedding:", embeddings["car_model_embedding"])
    print("License Plate Text:", embeddings["license_plate_text"])
    print("License Plate Embedding:", embeddings["license_plate_embedding"])
    print("Color Embedding:", embeddings["color_embedding"])
    print("Texture Embedding:", embeddings["texture_embedding"])
    print("-" * 50)  # Separator for better readability
