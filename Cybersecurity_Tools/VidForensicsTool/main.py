import cv2
import hashlib
import json
import os
import streamlit as st

# Function to extract video metadata
def extract_metadata(video_path):
    metadata = {}
    cap = cv2.VideoCapture(video_path)
    metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metadata['frame_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    metadata['frame_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    metadata['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return metadata

# Function to calculate video file hash (MD5)
def calculate_hash(video_path):
    hash_md5 = hashlib.md5()
    with open(video_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Function to analyze frames for alterations
def analyze_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    altered_frames = []
    prev_frame = None

    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            if prev_frame is not None:
                if not (frame == prev_frame).all():
                    altered_frames.append(i)
            prev_frame = frame.copy()

    cap.release()
    return altered_frames

# Streamlit app
st.title('Video Forensic Analysis Tool')

# File uploader
uploaded_file = st.file_uploader('Upload a video file', type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Check if 'temp' folder exists, create if not
    temp_folder = 'temp'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Save the uploaded file temporarily
    temp_file_path = os.path.join(temp_folder, uploaded_file.name)
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.read())

    st.write("**File uploaded successfully**")

    # Extract metadata
    st.write("**Extracting metadata...**")
    metadata = extract_metadata(temp_file_path)
    st.json(metadata)

    # Calculate hash
    st.write("**Calculating file hash (MD5)...**")
    video_hash = calculate_hash(temp_file_path)
    st.write(f"MD5 Hash: {video_hash}")

    # Analyze frames for alterations
    st.write("**Analyzing frames for alterations...**")
    altered_frames = analyze_frames(temp_file_path)
    st.write(f"Altered frames: {altered_frames}")

    # Generate report name based on video file name
    file_name = os.path.splitext(uploaded_file.name)[0]
    report_name = f'report-{file_name}.json'

    # Report creation
    report = {
        'metadata': metadata,
        'hash': video_hash,
        'altered_frames': altered_frames
    }

    st.write(f"**Video forensic analysis complete! Report saved as: {report_name}**")
    st.json(report)

    # Offer download of the JSON report
    report_json = json.dumps(report, indent=4)
    st.download_button(
        label="Download report as JSON",
        data=report_json,
        file_name=report_name,
        mime="application/json"
    )
