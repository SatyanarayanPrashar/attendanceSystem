import os
from sklearn.model_selection import train_test_split
from cnn import create_cnn_model
from load_data import load_data
from utils import save_model_and_data

from keras._tf_keras.keras.utils import to_categorical

def train_model(data_path="data/", model_path="models/"):
    # Step 1: Load face data
    print("Loading face data...")
    X, y, label_map = load_data(data_path)
    print(f"Data loaded: {len(X)} samples and {len(label_map)} unique labels.")

    # Step 2: Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"{len(X_train)} training samples and {len(X_val)} validation samples.")

    # Step 3: One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=len(label_map))
    y_val = to_categorical(y_val, num_classes=len(label_map))
    print("Labels one-hot encoded.")

    # Step 4: Create CNN model
    cnn_model = create_cnn_model(input_shape=(200, 200, 1), num_classes=len(label_map))
    cnn_model.summary()

    # Step 5: Train the model
    from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint  # Correct import

    checkpoint_path = os.path.join(model_path, "face_recognition_model.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss")
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    history = cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[checkpoint, early_stopping]
    )

    print("Training completed. Best model saved at:", checkpoint_path)

    # Step 6: Save model, label_map, and labels
    save_model_and_data(cnn_model, label_map, y, model_path)  # Save model and data
    print("Model, label map, and labels saved.")

if __name__ == "__main__":
    train_model()
