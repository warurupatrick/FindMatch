import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define a list of predefined color labels (RGB ranges for simplicity)
color_names = {
    'Black': [0, 0, 0],
    'White': [255, 255, 255],
    'Red': [255, 0, 0],
    'Green': [0, 255, 0],
    'Blue': [0, 0, 255],
    'Yellow': [255, 255, 0],
    'Orange': [255, 165, 0],
    'Gray': [169, 169, 169]
}

def get_closest_color(rgb):
    min_distance = float('inf')
    closest_color = None
    for color, value in color_names.items():
        # Calculate the Euclidean distance between the RGB values
        distance = np.linalg.norm(np.array(rgb) - np.array(value))
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    return closest_color

def extract_dominant_color(img_path, n_clusters=3):
    """
    Extract the dominant color of the image using KMeans clustering.
    Returns the RGB value of the dominant color.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))  # Resize for faster processing

    # Reshape the image to a 2D array of pixels (each pixel as a row)
    img = img.reshape((-1, 3))

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(img)

    # Get the RGB value of the dominant color (cluster center with the most pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant_color.astype(int)

def process_dataset(dataset_path):
    print(f"Processing dataset at: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset folder not found at {dataset_path}")
        return

    # List to store the results
    results = []

    # Process each image in the dataset
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        
        if os.path.isfile(img_path):
            dominant_color = extract_dominant_color(img_path)
            closest_color = get_closest_color(dominant_color)
            
            # Append result to the list
            results.append({
                'image': img_name,
                'dominant_color': dominant_color.tolist(),
                'closest_color': closest_color
            })
    
    return results

def save_results(results, output_file):
    import pandas as pd
    # Create a DataFrame and save to a CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Define the dataset path (adjust this to your local path)
    dataset_path = 'C:\Users\REGINAH\Desktop\ml\dataset\train'
    output_file = 'vehicle_colors.csv'  # Output file for the results

    # Process the dataset
    results = process_dataset(dataset_path)

    # Save the results to a CSV file
    if results:
        save_results(results, output_file)
