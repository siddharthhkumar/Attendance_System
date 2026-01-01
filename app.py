import streamlit as st
import cv2
import pandas as pd
import time
from datetime import datetime
import os
import numpy as np

# --- 1. SETUP ---
st.set_page_config(page_title="Smart Attendance", layout="wide")

# Load the trained mode
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('models/trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# File paths
STUDENT_FILE = "students.csv"
ATTENDANCE_FILE = "attendance.csv"

# --- 2. HELPER FUNCTIONS ---
def get_student_name(student_id):
    """Reads CSV to find name for a given ID"""
    if os.path.exists(STUDENT_FILE):
        df = pd.read_csv(STUDENT_FILE)
        student = df[df['Id'] == student_id]
        if not student.empty:
            return student.iloc[0]['Name']
    return "Unknown"

def mark_attendance(student_id, name):
    """Saves attendance to CSV if not already marked today"""
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["Id", "Name", "Time", "Date"])
        df.to_csv(ATTENDANCE_FILE, index=False)
    
    df = pd.read_csv(ATTENDANCE_FILE)
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Check if marked today
    if not df.empty:
        already_present = df[(df['Id'] == student_id) & (df['Date'] == today)]
        if not already_present.empty:
            return False # Already marked

    # Add new entry
    new_entry = pd.DataFrame([[student_id, name, current_time, today]], columns=["Id", "Name", "Time", "Date"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    return True

# --- 3. STREAMLIT INTERFACE ---
st.title("📸 Smart Attendance System")

menu = ["Home", "Mark Attendance", "View Reports"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Welcome Admin")
    st.write("Use the sidebar to navigate.")
    if os.path.exists(STUDENT_FILE):
        st.write("### Registered Students")
        st.dataframe(pd.read_csv(STUDENT_FILE))

elif choice == "Mark Attendance":
    st.subheader("Face Recognition Live Feed")
    run = st.checkbox('Turn on Camera')
    FRAME_WINDOW = st.image([])
    
    camera = cv2.VideoCapture(0)
    
    while run:
        ret, frame = camera.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Predict the face
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # Confidence: Lower is better (0 = perfect match)
            if confidence < 60: 
                name = get_student_name(id)
                confidence_text = f" {round(100 - confidence)}%"
                color = (0, 255, 0) # Green
                
                # Mark Attendance
                if mark_attendance(id, name):
                    st.success(f"Attendance Marked: {name}")
                    
            else:
                name = "Unknown"
                confidence_text = f" {round(100 - confidence)}%"
                color = (0, 0, 255) # Red
            
            # Draw rectangle and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    
    camera.release()

elif choice == "View Reports":
    st.subheader("Attendance Logs")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.warning("No attendance records found yet.")