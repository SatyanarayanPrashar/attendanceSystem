from load_data import load_face_data
from pca import perform_pca
from utils import save_model

def train_model(data_path="data/", model_path="models/"):
    """
    Trains the Eigenfaces model using PCA on face data.

    Parameters:
    - data_path (str): Path to the folder containing face images.
    - model_path (str): Path to save the trained model.
    """
    # Step 1: Load face data
    print("Loading face data...")
    data_matrix, labels, label_names = load_face_data(data_path)
    print(f"Loaded {len(data_matrix)} images from {len(label_names)} classes.")

    # Step 2: Perform PCA
    print("Performing PCA...")
    eigenfaces, mean_face, projections = perform_pca(data_matrix)
    print(f"PCA completed. {eigenfaces.shape[1]} eigenfaces retained.")

    # Step 3: Save the model
    print("Saving the model...")
    save_model(eigenfaces, mean_face, projections, labels, model_path)
    print("Training completed and model saved successfully!")

if __name__ == "__main__":
    train_model()
