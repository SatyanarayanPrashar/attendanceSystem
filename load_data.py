import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path="data/", image_size=(200, 200)):
    """
    Loads images and labels from the dataset directory.

    Parameters:
    - data_path (str): Path to the dataset.
    - image_size (tuple): Desired image size (width, height).

    Returns:
    - X (np.array): Array of image data.
    - y (np.array): Array of labels.
    - label_map (dict): Mapping of labels to user names.
    """
    X, y = [], []
    label_map = {}
    current_label = 0

    # Iterate through directories in the dataset path
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Map folder names to labels
        label_map[current_label] = folder_name
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".jpg"):
                # Read and preprocess image
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, image_size)
                X.append(image)
                y.append(current_label)
        
        current_label += 1

    # Convert X to a numpy array and normalize pixel values
    X = np.array(X).reshape(-1, image_size[0], image_size[1], 1) / 255.0  # Normalize pixel values to range [0, 1]
    
    # Convert y to a numpy array
    y = np.array(y)
    
    return X, y, label_map

# Load data and split into training and validation sets
X, y, label_map = load_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data loaded. {len(X_train)} training samples and {len(X_val)} validation samples.")
