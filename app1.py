import streamlit as st
import cv2
import numpy as np

# Load pre-trained MobileNet SSD model and class labels
model_path = r"C:\opencv project\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weights_path = r"C:\opencv project\frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weights_path, model_path)

# Load COCO class labels from labels.txt
class_labels = []
with open(r'C:\opencv project\labels.txt', 'r') as f:
    class_labels = f.read().strip().split('\n')

# Set model configuration
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

st.title("Webcam People Counting Application")

# Create a placeholder for displaying video frames
frame_placeholder = st.empty()

# Streamlit buttons to start and stop the webcam feed
start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False

# Main loop
if start_button:
    st.session_state.running = True

if stop_button:
    st.session_state.running = False

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Detect objects in the frame
        class_ids, confidences, bbox = net.detect(frame, confThreshold=0.5)

        # Initialize a counter for people in the current frame
        person_count = 0

        # Draw bounding boxes around detected objects
        if len(class_ids) != 0:
            for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), bbox):
                if class_id == 1:  # Class ID for 'person'
                    person_count += 1
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, f'Person {confidence * 100:.2f}%', (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the number of people detected
        cv2.putText(frame, f'People Count: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert frame to RGB and display it using Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    # Release video capture
    cap.release()
    st.write("Webcam stopped.")
else:
    st.write("Click 'Start Webcam' to begin.")