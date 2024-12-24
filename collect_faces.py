import cv2
import os
import time

def collect_data(data_path="data/", max_images=50, max_duration=10):
    """
    Collects face data for training by capturing images from the webcam.

    Parameters:
    - data_path (str): Path to store the collected face images.
    - max_images (int): Maximum number of images to capture.
    - max_duration (int): Maximum duration (in seconds) for data collection.
    """
    # Prompt the user for their name and USN
    name = input("Enter your name: ").strip()
    usn = input("Enter your USN: ").strip()
    
    # Create a directory named after the user
    user_folder = os.path.join(data_path, f"{name}_{usn}")
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
        print(f"Created folder: {user_folder}")
    else:
        print(f"Folder already exists: {user_folder}. Adding more images to it.")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    print("Press 'q' to quit.")
    print(f"Starting data collection for {max_duration} seconds or {max_images} images.")
    
    image_count = 0  # Counter for the number of images collected
    start_time = time.time()  # Start timer

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            if image_count < max_images:
                # Save the cropped face
                face = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (200, 200))  # Resize to a consistent size
                image_path = os.path.join(user_folder, f"{image_count}.jpg")
                cv2.imwrite(image_path, face_resized)
                print(f"Saved: {image_path}")
                image_count += 1

        # Display the video feed
        cv2.imshow("Data Collection", frame)
        
        # Exit the loop when 'q' is pressed or when limits are reached
        if (time.time() - start_time > max_duration) or (image_count >= max_images) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Data collection completed. {image_count} images saved in {user_folder}.")

if __name__ == "__main__":
    collect_data()
