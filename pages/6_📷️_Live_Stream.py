import av
import os
import time
import logging
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Logging configuration
logging.basicConfig(level=logging.INFO)
logging.getLogger("aiortc").setLevel(logging.ERROR)

# Check for GPU availability
if tf.config.list_physical_devices('GPU'):
    print("Running on GPU")
else:
    print("Running on CPU")

# Mediapipe Pose Initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Frame Processing Class
class ProcessFrame:
    def __init__(self, thresholds, flip_frame=False):
        self.flip_frame = flip_frame
        self.thresholds = thresholds
        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50, 50),
            'green': (0, 255, 127),
            'white': (255, 255, 255),
        }
        self.state_tracker = {'SQUAT_COUNT': 0, 'IMPROPER_SQUAT': 0}

    def process(self, frame, pose):
        # Your frame processing logic goes here
        return frame, None

# Frame Processing Callback
def video_frame_callback(frame: av.VideoFrame):
    try:
        frame = frame.to_ndarray(format="rgb24")
        frame, _ = live_process_frame.process(frame, pose)
        return av.VideoFrame.from_ndarray(frame, format="rgb24")
    except Exception as e:
        st.error(f"An error occurred while processing the frame: {e}")
        return frame

# Thresholds Configuration
def get_thresholds_beginner():
    return {'HIP_THRESH': [10, 50], 'KNEE_THRESH': [50, 70, 95]}

def get_thresholds_pro():
    return {'HIP_THRESH': [15, 50], 'KNEE_THRESH': [50, 80, 95]}

# Streamlit Layout
st.title('AI Fitness Trainer: Squats Analysis')
mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True, key="mode_selector")

# Thresholds and ProcessFrame Instance
thresholds = get_thresholds_beginner() if mode == 'Beginner' else get_thresholds_pro()
live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)

# MediaRecorder Factory
output_video_file = 'output_live.flv'
def out_recorder_factory():
    return MediaRecorder(output_video_file)

# WebRTC Setup
ctx = webrtc_streamer(
    key="Squats-pose-analysis",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"width": {"min": 480, "ideal": 640}}, "audio": False},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True),
    out_recorder_factory=out_recorder_factory
)

# Display WebRTC Stream Status
if ctx and ctx.state.playing:
    st.success("WebRTC stream is active.")
else:
    st.warning("Waiting for WebRTC connection...")

# Download Button for Processed Video
if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as f:
        st.download_button('Download Video', f, file_name='output_live.flv')

# Additional Utility Functions
def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    v1 = p1 - ref_pt
    v2 = p2 - ref_pt
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
    return angle

def draw_text(img, msg, pos=(0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, text_color=(0, 255, 0), bg_color=(0, 0, 0)):
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    x, y = pos
    cv2.rectangle(img, (x, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
    cv2.putText(img, msg, (x, y), font, font_scale, text_color, font_thickness)
    return img

# Ensuring Clean Download State
if 'download' not in st.session_state:
    st.session_state['download'] = False

if os.path.exists(output_video_file) and st.session_state['download']:
    os.remove(output_video_file)
    st.session_state['download'] = False
