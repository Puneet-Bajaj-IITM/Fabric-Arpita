import streamlit as st
import cv2
from computer_vision_pipeline import StitchDetectionPipeline
from barcode_data_system import BarcodeDataSystem
import os
import ctypes
from pyzbar.pyzbar import decode
from datetime import datetime
import io
import barcode
from barcode.writer import ImageWriter

# Initialize BarcodeDataSystem and StitchDetectionPipeline
barcode_system = BarcodeDataSystem()
stitch_pipeline = StitchDetectionPipeline()

# Session management variables
session_id = None
current_frame = None

# Function to start a new session
def start_new_session(barcode_data):
    global session_id
    session_id = barcode_system.start_session(barcode_data['barcode_id'])

# Function to handle barcode scanning on the top camera
def scan_barcode_on_top_camera(frame):
    barcode_data = barcode_system.scan_barcode(frame)
    if barcode_data:
        start_new_session(barcode_data)
    return barcode_data

# Function to process frames for stitching and alignment
def process_frame_for_analysis(frame):
    stitch_results = stitch_pipeline.detect_stitches(frame)
    defect_results = stitch_pipeline.detect_defects(frame)
    reference_line = (100, 100, 1820, 100)
    alignment_results = stitch_pipeline.analyze_alignment(frame, reference_line)
    
    return stitch_results, defect_results, alignment_results

# Function to visualize results
def visualize_results(frame, stitch_results, defect_results, alignment_results):
    for stitch in stitch_results['stitches']:
        cv2.circle(frame, (stitch['x'], stitch['y']), 3, (0, 255, 0), -1)
    
    for defect in defect_results['defects']:
        x, y, w, h = defect['coords']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.line(frame, alignment_results['reference_line'][0], alignment_results['reference_line'][1], (255, 0, 0), 2)
    cv2.putText(frame, f"Alignment Score: {alignment_results['score']}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if alignment_results['score'] < 90:
        cv2.line(frame, alignment_results['reference_line'][0], alignment_results['reference_line'][1], (0, 0, 255), 2)
    
    return frame

# Light intensity control placeholder
def set_light_intensity(intensity):
    pass

# Streamlit interface
st.title('Sewing Quality Control System')

# Light Intensity Control
intensity = st.slider('Set Camera Light Intensity', min_value=0, max_value=100, value=50)
set_light_intensity(intensity)

# Camera Stream
st.sidebar.title("Camera Selection")
top_camera_index = st.sidebar.selectbox("Select Top Camera", options=[0, 1], index=0)
bottom_camera_index = st.sidebar.selectbox("Select Bottom Camera", options=[0, 1], index=1)

# Initialize placeholders for video frames
top_camera_placeholder = st.empty()
bottom_camera_placeholder = st.empty()

# Session control
if st.sidebar.button('Start Session'):
    st.session_state['session_active'] = True

# Process frames from cameras
if 'session_active' in st.session_state and st.session_state['session_active']:
    top_camera = cv2.VideoCapture(top_camera_index)
    bottom_camera = cv2.VideoCapture(bottom_camera_index)

    while top_camera.isOpened() and bottom_camera.isOpened():
        top_ret, top_frame = top_camera.read()
        bottom_ret, bottom_frame = bottom_camera.read()

        if not (top_ret and bottom_ret):
            st.warning("Failed to capture frames.")
            break

        stitch_results, defect_results, alignment_results = process_frame_for_analysis(top_frame)
        processed_top_frame = visualize_results(top_frame, stitch_results, defect_results, alignment_results)
        top_camera_placeholder.image(cv2.cvtColor(processed_top_frame, cv2.COLOR_BGR2RGB), caption="Top Camera with Analysis")

        stitch_results, defect_results, alignment_results = process_frame_for_analysis(bottom_frame)
        processed_bottom_frame = visualize_results(bottom_frame, stitch_results, defect_results, alignment_results)
        bottom_camera_placeholder.image(cv2.cvtColor(processed_bottom_frame, cv2.COLOR_BGR2RGB), caption="Bottom Camera with Analysis")

        if st.sidebar.button("Stop Session"):
            barcode_system.end_session(session_id)
            st.success('Session Ended and Report Generated.')
            st.session_state['session_active'] = False
            break

    top_camera.release()
    bottom_camera.release()
    cv2.destroyAllWindows()


# Add new tab for generating barcode
st.sidebar.title("Fabric Barcode Generator")
with st.expander("Generate New Barcode"):
    fabric_type = st.text_input("Fabric Type")
    width = st.number_input("Fabric Width (in cm)", min_value=0.0, step=0.1)
    gsm = st.number_input("Fabric GSM", min_value=0.0, step=0.1)
    stitch_length = st.number_input("Stitch Length (mm)", min_value=0.0, step=0.1)

    if st.button("Generate Barcode"):
        if fabric_type and width and gsm and stitch_length:
            fabric_data = {
                'fabric_type': fabric_type,
                'width': width,
                'gsm': gsm,
                'stitch_length': stitch_length
            }
            barcode_id = barcode_system.create_barcode(fabric_data)

            # Generate the barcode image
            barcode_img = barcode.get_barcode_class('code128')(barcode_id, writer=ImageWriter())
            buffer = io.BytesIO()
            barcode_img.write(buffer)
            buffer.seek(0)
            st.image(buffer, caption="Generated Barcode")

            st.success(f"Barcode {barcode_id} generated successfully!")
        else:
            st.error("Please fill in all fabric details.")

