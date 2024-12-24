import os
import cv2
import numpy as np

def load_face_data(data_path="data/", image_size=(64, 64)):
    """
    Loads face images and their labels into a matrix and label array.

    Parameters:
    - data_path (str): Path to the folder containing face images.
    - image_size (tuple): The size to which images are resized (e.g., 64x64).

    Returns:
    - face_matrix (numpy.ndarray): Matrix of flattened face images (num_samples x num_features).
    - labels (list): List of labels corresponding to the faces.
    - label_names (list): Unique names corresponding to labels.
    """
    face_matrix = []
    labels = []
    label_names = []
    
    for label, person_name in enumerate(os.listdir(data_path)):
        person_folder = os.path.join(data_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue
        
        label_names.append(person_name)
        
        for file in os.listdir(person_folder):
            file_path = os.path.join(person_folder, file)
            if file_path.endswith(".jpg"):
                # Load the image in grayscale
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                # Resize the image to the standard size
                image_resized = cv2.resize(image, image_size)
                # Flatten the image
                face_matrix.append(image_resized.flatten())
                labels.append(label)
    
    # Convert to NumPy arrays
    face_matrix = np.array(face_matrix)
    labels = np.array(labels)
    
    return face_matrix, labels, label_names

if __name__ == "__main__":
    data, labels, names = load_face_data()
    print(f"Loaded {len(data)} face images for {len(names)} people.")
