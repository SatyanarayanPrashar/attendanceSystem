import cv2
import numpy as np
import os
from datetime import datetime
from utils import load_model
import openpyxl
import time
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import geocoder

def get_user_location():
    """
    Fetch the user's current location using geocoder (IP-based).
    
    Returns:
    - tuple: (latitude, longitude) of the user's location.
    """
    try:
        g = geocoder.ip('me')
        if g.ok:
            print(g.latlng)
            return g.latlng
        else:
            print("Could not fetch location. Ensure you are connected to the internet.")
            return None
    except Exception as e:
        print(f"Failed to get user location: {e}")
        return None

def is_in_classroom(user_coords, classroom_coords, offset=50):
    distance = geodesic(user_coords, classroom_coords).meters
    print(f"Distance to classroom: {distance:.2f} meters")
    return distance <= offset

def load_label_map(data_path="data/"):
    """
    Dynamically loads label map from the dataset directory.

    Parameters:
    - data_path (str): Path to the dataset.

    Returns:
    - label_map (dict): Mapping of labels to user names.
    """
    label_map = {}
    current_label = 0

    # Iterate through directories in the dataset path
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path):
            label_map[current_label] = folder_name
            current_label += 1
    
    return label_map

def recognize_face(model_path="models/face_recognition_model.keras", data_path="data/", excel_file="attendance.xlsx"):
    # classroom_coords = (12.9719, 77.5936)
    classroom_coords = (16.9719, 77.5936)

    # Load the trained CNN model
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded successfully!")

    # Load the label mapping dynamically
    label_map = load_label_map(data_path)
    print(f"Label map loaded: {label_map}")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Press 'q' to quit.")

    recognized_faces = set()  # Keep track of already marked faces

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            # Crop and preprocess the face
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))  # Match the input size of the CNN model
            face_normalized = face_resized / 255.0  # Normalize pixel values
            face_input = face_normalized.reshape(1, 200, 200, 1)  # Reshape for the CNN model

            # Predict using the CNN model
            prediction = model.predict(face_input)[0]  # Assuming the model returns probabilities for each class
            predicted_label = np.argmax(prediction)  # Get the class with the highest probability
            print(f"Recognized label: {predicted_label}")

            # Get the name based on predicted label from the dynamically loaded label map
            recognized_name = label_map.get(predicted_label, "Unknown")

            # Display the result
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            if recognized_name != "Unknown" and recognized_name not in recognized_faces:
                user_coords = get_user_location()
                if user_coords and is_in_classroom(user_coords, classroom_coords, offset=50):
                    # Mark as "present" for recognized faces within location range
                    print("Marking attendance...")
                    mark_attendance(recognized_name, status="present", excel_file=excel_file)
                    recognized_faces.add(recognized_name)
                else:
                    print(f"{recognized_name} is outside the classroom location.")

        # Show the frame
        cv2.imshow("Face Recognition", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def mark_attendance(name, status, excel_file="attendance.xlsx"):
    if not os.path.exists(excel_file):
        print(f"File {excel_file} does not exist. Creating a new file...")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Attendance"
        sheet.append(["Name", "Date", "Time", "Status"])
        workbook.save(excel_file)

    # Open the existing Excel file
    workbook = openpyxl.load_workbook(excel_file)
    sheet = workbook.active

    # Get the current date and time
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check if the person is already marked present today
    for row in sheet.iter_rows(values_only=True):
        if row[0] == name and row[1] == date:
            print(f"{name} is already marked present today.")
            return  # Person is already marked present

    # Append the attendance entry
    sheet.append([name, date, time, status])
    workbook.save(excel_file)
    print(f"Marked {status} attendance for {name} at {time}.")

if __name__ == "__main__":
    recognize_face()
