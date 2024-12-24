import cv2
import numpy as np
import os
from datetime import datetime
from utils import load_model
import openpyxl
from geopy.distance import geodesic
import geocoder
import time

def get_user_location():
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

def is_in_classroom(user_coords, classroom_coords, offset=10):
    distance = geodesic(user_coords, classroom_coords).meters
    print(f"Distance to classroom: {distance:.2f} meters")
    return distance <= offset

def load_label_map(data_path="data/"):
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
    classroom_coords = (16.9719, 77.5936)

    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded successfully!")

    label_map = load_label_map(data_path)
    print(f"Label map loaded: {label_map}")

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("Press 'q' to quit.")

    recognized_timers = {}  # Dictionary to track recognition timestamps
    recognized_faces = set()  # Keep track of marked attendance

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        current_time = time.time()
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (200, 200))
            face_normalized = face_resized / 255.0
            face_input = face_normalized.reshape(1, 200, 200, 1)

            prediction = model.predict(face_input)[0]
            max_probability = np.max(prediction)
            predicted_label = np.argmax(prediction)

            if max_probability >= 0.8:
                recognized_name = label_map.get(predicted_label, "Unknown")
            else:
                recognized_name = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            if recognized_name != "Unknown" and recognized_name not in recognized_faces:
                if recognized_name not in recognized_timers:
                    recognized_timers[recognized_name] = current_time  # Start timer
                elif current_time - recognized_timers[recognized_name] >= 5:  # Check 5-second condition
                    user_coords = get_user_location()
                    if user_coords and is_in_classroom(user_coords, classroom_coords, offset=10):
                        print("Marking attendance...")
                        mark_attendance(recognized_name, "present", excel_file)
                        recognized_faces.add(recognized_name)
                    else:
                        print(f"{recognized_name} is outside the classroom location.")
            else:
                recognized_timers.pop(recognized_name, None)  # Reset timer if face is lost

        cv2.imshow("Face Recognition", frame)

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

    workbook = openpyxl.load_workbook(excel_file)
    sheet = workbook.active

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    for row in sheet.iter_rows(values_only=True):
        if row[0] == name and row[1] == date:
            print(f"{name} is already marked present today.")
            return

    sheet.append([name, date, time, status])
    workbook.save(excel_file)
    print(f"Marked {status} attendance for {name} at {time}.")

if __name__ == "__main__":
    recognize_face()
