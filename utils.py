import json
import numpy as np
from keras._tf_keras.keras.models import save_model, load_model

def save_model_and_data(model, label_map, labels, model_path):
    model.save(model_path + ".keras")  # Saving the model in .keras format
    
    # Save the label map as a JSON file
    with open(model_path + "_label_map.json", "w") as f:
        json.dump(label_map, f)
        
    # Save the labels as a .npy file
    np.save(model_path + "_labels.npy", labels)

def load_model_and_data(model_path):
    """
    Load the Keras model and associated data (label map and labels) from disk.

    Args:
        model_path: The base path where the model and data are stored.

    Returns:
        model: The loaded Keras model.
        label_map: The loaded label map.
        labels: The loaded labels.
    """
    # Load the Keras model from the .keras file
    model = load_model(model_path + ".keras")  # Loading the model from .keras format
    
    # Load the label map from the JSON file
    with open(model_path + "_label_map.json", "r") as f:
        label_map = json.load(f)
        
    # Load the labels from the .npy file
    labels = np.load(model_path + "_labels.npy")
    
    return model, label_map, labels

