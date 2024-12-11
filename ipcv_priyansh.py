import streamlit as st
import cv2
import math
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import tempfile
import os

# Streamlit app definition
def main():
    st.set_page_config(page_title="YOLO Object Detection", layout="wide")

    # Apply custom styling
    st.markdown(
        """
        <style>
        .reportview-container {
            background: #f5f5f5;
        }
        .sidebar .sidebar-content {
            background-color: #333;
            color: #fff;
            padding: 20px;
        }
        .stButton>button {
            background-color: #FF5733;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button:hover {
            background-color: #C70039;
        }
        h1 {
            font-family: 'Arial', sans-serif;
            color: #1F618D;
        }
        .stTextInput, .stSelectbox, .stButton {
            font-size: 18px;
        }
        .stFileUploader {
            font-size: 18px;
        }
        .stChart {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 10px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Sidebar for file upload and model selection
    st.sidebar.title("Settings")
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

    # Load YOLO model
    @st.cache_resource
    def load_model():
        return YOLO("yolo-Weights/yolov8n.pt")

    model = load_model()

    # Define class names from YOLO model
    class_names = model.names  # Get all class names directly from the YOLO model

    # Initialize variables for tracking object counts over time and dwell time
    object_counts_over_time = []
    object_dwell_time = {}  # Track the first frame each object appears
    frame_number = 0

    # Process a single frame with YOLO model
    def process_frame(frame, model):
        results = model(frame, stream=True)
        frame_counts = defaultdict(int)
        frame_confidences = defaultdict(float)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])

                # Get the class name from YOLO's class index
                class_name = class_names[cls]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Handle confidence extraction
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf[0]
                confidence = round(confidence * 100) / 100

                frame_counts[class_name] += 1
                frame_confidences[class_name] += confidence

                # Track dwell time: initialize if not seen before
                if class_name not in object_dwell_time:
                    object_dwell_time[class_name] = frame_number

                # Display label and confidence on frame
                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, f"{class_name} {confidence:.2f}", org, font, font_scale, color, thickness)

        return frame, frame_counts, frame_confidences

    # Update and display charts
    def update_charts(frame_counts, frame_confidences, chart1_placeholder, chart2_placeholder, chart3_placeholder):
        nonlocal frame_number
        frame_number += 1
        
        # Track total object count for the frame
        total_objects = sum(frame_counts.values())
        object_counts_over_time.append(total_objects)

        # Chart 1: Object Count Over Time
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(object_counts_over_time, color='#1F618D', linewidth=2)
        ax1.set_title("Object Count Over Time", fontsize=16, color='#1F618D')
        ax1.set_xlabel("Frames", fontsize=12)
        ax1.set_ylabel("Object Count", fontsize=12)
        plt.tight_layout()
        chart1_placeholder.pyplot(fig1)
        plt.close(fig1)

        # Chart 2: Current Frame Confidence
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        avg_confidence = {cls: frame_confidences[cls] / frame_counts[cls]
                          for cls in frame_counts if frame_counts[cls] > 0}
        ax2.bar(avg_confidence.keys(), avg_confidence.values(), color='#FF5733')
        ax2.set_title("Current Frame Confidence", fontsize=16, color='#FF5733')
        ax2.set_ylabel("Confidence", fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.set_xticklabels(avg_confidence.keys(), rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        chart2_placeholder.pyplot(fig2)
        plt.close(fig2)
       
        # Chart 3: Object Dwell Time
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        dwell_times = {cls: frame_number - object_dwell_time[cls] for cls in frame_counts}
        ax3.bar(dwell_times.keys(), dwell_times.values(), color='#C70039')
        ax3.set_title("Object Dwell Time", fontsize=16, color='#C70039')
        ax3.set_xlabel("Object Class", fontsize=12)
        ax3.set_ylabel("Dwell Time (Frames)", fontsize=12)
        plt.tight_layout()
        chart3_placeholder.pyplot(fig3)
        plt.close(fig3)

    # Add Start and Stop buttons
    if uploaded_file is not None:
        st.sidebar.subheader("Controls")
        stop_button = st.sidebar.button("Stop Processing")

        # Temporary file storage and video reading
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        video_placeholder = st.empty()

        col1, col2 = st.columns([2, 3])  # Adjust column width for a more balanced layout
        chart1_placeholder = col1.empty()
        chart2_placeholder = col2.empty()
        chart3_placeholder = st.empty()

        # Process video frame by frame
        while cap.isOpened() and not stop_button:
            success, frame = cap.read()
            if not success:
                break

            # Process and display frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, frame_counts, frame_confidences = process_frame(frame, model)
            video_placeholder.image(frame, use_column_width=True)

            # Update charts
            update_charts(frame_counts, frame_confidences, chart1_placeholder, chart2_placeholder, chart3_placeholder)

        cap.release()
        os.unlink(tfile.name)

# Run the Streamlit app
if __name__ == "__main__":
    main()
