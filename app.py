import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="Real-Time Eye Detection", layout="centered")
st.title("👁️ Real-Time Eye State Detection (Camera)")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

# Eye landmark indices (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = np.array(
                [(int(face_landmarks.landmark[i].x * w),
                  int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            )

            right_eye = np.array(
                [(int(face_landmarks.landmark[i].x * w),
                  int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
            )

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            if ear < 0.25:
                status = "EYES CLOSED"
                color = (0, 0, 255)
            else:
                status = "EYES OPEN"
                color = (0, 255, 0)

            cv2.putText(frame, status, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
