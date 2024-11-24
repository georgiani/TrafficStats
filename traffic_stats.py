import time
import state_init
from state_init import cap, bb, lbl, tracker, model, W, H, WHITE, GREEN
import supervision as sv
import streamlit as st
import numpy as np
import json
import cv2

# ----------
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
# ----------

TS_DELTA = 4
SELECTION_W = int(W / 3)
SELECTION_H = int(H / 2)
ZONE1 = np.array([[590, 205], [690, 140], [360, 20], [250, 70]])

def save_step_to_json_file(filename, step, data):
    with open(filename, "r") as res_file_read:
        results_json = json.load(res_file_read)

    keys = [k for k in results_json.keys()]

    if not keys:
        # first time
        results_json[step] = data
        
        with open(filename, "w") as res_file_write:
            res_file_write.write(json.dumps(results_json))

        return

    if "ts" in results_json[keys[-1]] and "ts" in data:
        if data["ts"] - results_json[keys[-1]]["ts"] > TS_DELTA:
            results_json[step] = data
        
            with open(filename, "w") as res_file_write:
                res_file_write.write(json.dumps(results_json))

def process_results(results, frame, flags = None):
    ts = time.time()
    detections = results
    
    polygon_zone = sv.PolygonZone(ZONE1)
    detections = detections[polygon_zone.trigger(detections)]
    detections = tracker.update_with_detections(detections=detections)

    labels = [
        f"car{tracker_id}" for tracker_id in detections.tracker_id
    ]

    frame = bb.annotate(scene=frame, detections=detections)
    frame = lbl.annotate(scene=frame, detections=detections, labels=labels)
    frame = sv.draw_polygon(frame, ZONE1, color=sv.Color.RED, thickness=1)

    ids = [int(i) for i in detections.tracker_id]
    if len(ids) == 0:
        stats = {
            'ids': []
        }

        if st.session_state['save_stats_cb']:
            save_step_to_json_file("res.json", st.session_state['step'], stats)

        return stats, frame

    stats = {
        'ids': ids,
        'ts': ts
    }

    if st.session_state['save_stats_cb']:
        save_step_to_json_file("res.json", st.session_state['step'], stats)

    return stats, frame

def callback(image_slice: np.ndarray) -> sv.Detections:
    ultralytics_results = model.predict(image_slice)
    sv_detections = sv.Detections.from_ultralytics(ultralytics_results)
    return sv_detections

def next_frame():
    full_frame = cap.read()


    if full_frame is None:
        return None

    x_begin = st.session_state["xb"]
    x_end = st.session_state["xe"]
    y_begin = st.session_state["yb"]
    y_end = st.session_state["ye"]

    frame = full_frame[y_begin:y_end, x_begin:x_end]

    full_frame = full_frame[SELECTION_H:H, SELECTION_W:W]

    image_center = tuple(np.array(frame.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, st.session_state["rotation"], 1.0)
    frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    # Proportions for the one street that goes only straight
    going_straight_frame = frame[int((y_end - y_begin) / 3):int(0.6 * (y_end - y_begin)),:-60] 
    
    results = None
    editing_mode = st.session_state['edit_cb']
    if not editing_mode:
        slicer = sv.InferenceSlicer(callback=callback)
        results = slicer(image=full_frame)
        # results = model.predict(full_frame)
        st.session_state['stats'], full_frame = process_results(results, full_frame)

    return full_frame

#---------------------------------------------------

if 'w' not in st.session_state:
    st.session_state['w'] = 640

if 'h' not in st.session_state:
    st.session_state['h'] = 320
    
if 'stats' not in st.session_state:
    st.session_state['stats'] = {
        'number_of_cars': 0
    }

if 'step' not in st.session_state:
    st.session_state['step'] = 0

if 'rotation' not in st.session_state:
    st.session_state['rotation'] = 20

if 'xb' not in st.session_state:
    st.session_state["xb"] = 600

if 'xe' not in st.session_state:
    st.session_state["xe"] = 1150

if 'yb' not in st.session_state:
    st.session_state["yb"] = 250

if 'ye' not in st.session_state:
    st.session_state["ye"] = 700

st.set_page_config(layout="wide")
with st.container():
    controls, vis = st.columns(2, vertical_alignment="center", gap="small")

    with controls:
        left, left_center, center, right_center, right = st.columns([0.3, 0.1, 0.1, 0.1, 0.3], vertical_alignment="top", gap="small")

        with left_center:
            left_bt = st.button("⬅️", "left_bt")
            left_bt_minus = st.button("↩️", "undo_left")
            rot_bt_left = st.button("🔄️", "rot_bt_left")
            
        with center:
            up_bt = st.button("⬆️", "up_bt")
            up_bt_minus = st.button("↩️", "undo_up")
            down_bt_minus = st.button("↩️", "undo_down")
            down_bt = st.button("⬇️", "down_bt")

        with right_center:
            right_bt = st.button("➡️", "right_bt")
            right_bt_minus = st.button("↩️", "undo_right")
            rot_bt_right = st.button("🔃", "rot_bt_right")

        if left_bt:
            st.session_state["xb"] -= 40
        
        if left_bt_minus:
            st.session_state["xb"] += 40

        if right_bt:
            st.session_state["xe"] += 40
        
        if right_bt_minus:
            st.session_state["xe"] -= 40

        if up_bt:
            st.session_state["yb"] -= 40

        if up_bt_minus:
            st.session_state["yb"] += 40

        if down_bt:
            st.session_state["ye"] += 40

        if down_bt_minus:
            st.session_state["ye"] -= 40

        if rot_bt_left:
            st.session_state["rotation"] += 10
        
        if rot_bt_right:
            st.session_state["rotation"] -= 10

        edit_cb   = st.checkbox("Editing Mode", True, key="edit_cb")
        number_of_cars = st.empty()
        time_step = st.empty()
        save_stats = st.checkbox("Save Statistics", key="save_stats_cb")
        
    with vis:
        visualization = st.empty()

        if st.session_state["edit_cb"]:
            frame_results = next_frame()
            if frame_results is not None:
                with visualization:
                    st.image(frame_results)
                
                number_of_cars.write("Editing")
        else:
            while True:
                frame_results = next_frame()
                
                if frame_results is not None:
                    with visualization:
                        st.image(frame_results, use_column_width="always")
                    
                    number_of_cars.write(st.session_state["stats"])
                        
                    time_step.write(st.session_state['step'])

                    st.session_state['step'] += 1

                    time.sleep(2.5)