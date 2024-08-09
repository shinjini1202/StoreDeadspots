import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Load the necessary files
cwd = os.getcwd()
video_path = os.path.abspath(os.path.join(cwd, 'store.mp4'))
prototxt_path = os.path.abspath(os.path.join(cwd, 'deploy.prototxt'))
model_path = os.path.abspath(os.path.join(cwd, 'mobilenet_iter_73000.caffemodel'))

# Load the video and initialize the detection model
cap = cv2.VideoCapture(video_path)
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def detect_people(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    people = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if idx == 15:  # Class label for person
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                people.append((startX, startY, endX, endY))
    return people

def draw_bounding_boxes(frame, people):
    for (startX, startY, endX, endY) in people:
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return frame

def calculate_footfall(frame, people):
    tracks = defaultdict(list)
    track_id = 0

    for (startX, startY, endX, endY) in people:
        centroid = (int((startX + endX) / 2), int((startY + endY) / 2))
        tracks[track_id].append(centroid)
        track_id += 1

    grid_size = (5, 5)  # Adjust grid size as needed
    footfall = np.zeros(grid_size)

    for track in tracks.values():
        for (x, y) in track:
            grid_x = x // (frame.shape[1] // grid_size[0])
            grid_y = y // (frame.shape[0] // grid_size[1])
            footfall[grid_y, grid_x] += 1

    return footfall

def plot_heatmap(footfall):
    plt.figure(figsize=(10, 4))
    sns.heatmap(footfall, cmap='hot', annot=True)
    plt.title('Customer Footfall Heatmap')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    return plt.gcf()

def run_app():
    st.title("Store Deadspot Detection")

    # Display the video
    st.subheader("Video Feed")
    frame_placeholder = st.empty()

    # Display the heatmap
    st.subheader("Customer Footfall Heatmap")
    heatmap_placeholder = st.empty()

    # Start the video processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people and draw bounding boxes
        people = detect_people(frame)
        frame_with_boxes = draw_bounding_boxes(frame, people)

        # Update the video feed
        frame_placeholder.image(frame_with_boxes, channels="BGR")

        # Calculate and display the heatmap
        footfall = calculate_footfall(frame, people)
        heatmap_placeholder.pyplot(plot_heatmap(footfall))

    # Release the video capture
    cap.release()

if __name__ == "__main__":
    run_app()