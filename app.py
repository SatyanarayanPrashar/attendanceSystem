from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from utils import load_model
from recognize_and_mark import load_label_map, mark_attendance
import threading

app = Flask(__name__)

# Global variables
model_path = "models/face_recognition_model.keras"
data_path = "data/"
excel_file = "attendance.xlsx"
label_map = load_label_map(data_path)
model = load_model(model_path)
recognized_faces = set()
camera = cv2.VideoCapture(0)  # Initialize webcam

# Function to generate frames from the camera feed
def generate_frames():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (200, 200)) / 255.0
                face_input = face_resized.reshape(1, 200, 200, 1)
                prediction = model.predict(face_input)[0]
                max_probability = np.max(prediction)
                predicted_label = np.argmax(prediction)

                if max_probability >= 0.8:
                    recognized_name = label_map.get(predicted_label, "Unknown")
                    if recognized_name != "Unknown" and recognized_name not in recognized_faces:
                        mark_attendance(recognized_name, "present", excel_file)
                        recognized_faces.add(recognized_name)

                    # Draw rectangle and label on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to fetch camera feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to fetch recognized names dynamically
@app.route('/recognized_names', methods=['GET'])
def get_recognized_names():
    attendance_status = {name: "Present" if name in recognized_faces else "Absent" for name in label_map.values()}
    return jsonify(attendance_status)

# Home route
@app.route('/')
def index():
    # Fetch attendance status
    attendance_status = {name: "Present" if name in recognized_faces else "Absent" for name in label_map.values()}
    return render_template('index.html', attendance=attendance_status)

@app.route('/reset_status', methods=['POST'])
def reset_status():
    global recognized_faces
    recognized_faces = set()  # Reset the set
    return jsonify({"message": "Attendance status has been reset."})


if __name__ == '__main__':
    app.run(debug=True)
