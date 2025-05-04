# Attendance System

## Project Description

The Attendance System is a real-time face recognition application that utilizes PCA (Principal Component Analysis) for the training and recognition of faces to mark attendance. The system is designed to improve attendance management by automatically recognizing individuals and logging their presence based on face duration and classroom location. The project leverages several technologies including OpenCV for image processing, Excel for attendance logging, and geocoding for location-based attendance verification.

## Table of Contents
- [Project Description](#project-description)
- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: OpenCV, Numpy, Openpyxl, Geopy, Geocoder
- Webcam for real-time recognition
- Appropriate dataset for training the model

### Installation Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd attendanceSystem
   ```
2. Install the required Python packages:
   ```bash
   pip install opencv-python numpy openpyxl geopy geocoder
   ```
3. Initialize the Excel file to log attendance:
   ```python
   python initialize_excel.py
   ```
4. Collect face data to train the model:
   ```python
   python collect_faces.py
   ```
5. Train the model:
   ```python
   python main.py
   ```
6. Start the face recognition and attendance marking process:
   ```python
   python recognize_and_mark.py
   ```

## Deployment

The system can be deployed locally on your machine. Ensure that you have followed the installation steps correctly. For cloud deployment, you can utilize platforms like Heroku or AWS after containerizing the app using Docker.

## Contributing

Contributions are welcome! Please follow these steps:
- Fork the repository
- Create a new branch for your feature or bugfix
- Commit your changes
- Push to the branch
- Open a pull request
