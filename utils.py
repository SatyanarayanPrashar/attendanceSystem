import numpy as np

def save_model(eigenfaces, mean_face, projections, labels, model_path="models/"):
    """
    Saves the PCA-based Eigenfaces model.
    """
    np.save(f"{model_path}eigenfaces_model.npy", {
        "eigenfaces": eigenfaces,
        "mean_face": mean_face,
        "projections": projections
    })
    np.save(f"{model_path}face_labels.npy", labels)
    print("Model saved successfully.")

def load_model(model_path="models/"):
    """
    Loads the PCA-based Eigenfaces model.
    """
    model = np.load(f"{model_path}eigenfaces_model.npy", allow_pickle=True).item()
    labels = np.load(f"{model_path}face_labels.npy", allow_pickle=True)
    print("Model loaded successfully.")
    return model["eigenfaces"], model["mean_face"], model["projections"], labels
