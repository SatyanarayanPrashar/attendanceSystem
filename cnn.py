from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Input,Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from load_data import load_data

import numpy as np
import matplotlib.pyplot as plt

def create_cnn_model(input_shape=(200, 200, 1), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model with categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Debug: Print model summary
    print("\nModel Summary:")
    model.summary()
    
    return model

def debug_data(X, y):
    """
    Debugging function to inspect input data and labels.
    """
    print("\nDebugging Data:")
    print(f"Input shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")
    
    # Visualize a few samples
    print("\nVisualizing sample images with labels:")
    for i in range(5):
        plt.imshow(X[i].reshape(200, 200), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
        plt.show()

def debug_predictions(model, X, y):
    """
    Debugging function to check model predictions on a subset of data.
    """
    print("\nDebugging Predictions:")
    predictions = model.predict(X[:5])
    for i, prediction in enumerate(predictions):
        print(f"Image {i}: Predicted = {prediction}, Actual = {y[i]}")
