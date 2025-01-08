import json
from get_stream import get_stream_capture
from Model import Model
import supervision as sv
import streamlit as st
import numpy as np
import cv2

ZONE1 = np.array([[590, 205], [690, 140], [360, 20], [250, 70]])
ZONE2 = np.array([[350, 300], [600, 300], [750, 150], [650, 150]])
ZONE3 = np.array([[80, 470], [480, 160], [610, 195], [310, 520]])

VIDEOS = {
    "Unirii": {
        "video": "https://www.youtube.com/watch?v=rs2be3mqryo",
        "zone": ZONE1
    },
    "Victoriei": {
        "url": "https://www.youtube.com/watch?v=1dLkP_nwZLo",
        "zone": ZONE2
    },
    "BaiaMare": {
        "url": "https://p.webcamromania.ro/baiamare/index.m3u8",
        "zone": ZONE3
    }
}

if 'stats' not in st.session_state:
    st.session_state['stats'] = {
        'number_of_cars': 0
    }

if 'step' not in st.session_state:
    st.session_state['step'] = 0

if 'location' not in st.session_state:
    st.session_state['location'] = "BaiaMare"

if 'playing' not in st.session_state:
    st.session_state['playing'] = False

model = Model("yolo")
cap = get_stream_capture(VIDEOS[st.session_state["location"]]["url"])
bb = sv.BoundingBoxAnnotator(thickness=1)
lbl = sv.LabelAnnotator(text_padding=0, text_scale=0.2)
FPS = int(cap.get(cv2.CAP_PROP_FPS))
tracker = sv.ByteTrack(frame_rate=2, lost_track_buffer=2)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("rerun")

with open("res.json", "w") as res_file:
    json.dump({}, res_file)
