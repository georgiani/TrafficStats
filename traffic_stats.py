import time
import state_init
from state_init import cap, bb, lbl, tracker, model, W, H, VIDEOS
import supervision as sv
import streamlit as st
import pandas as pd
import numpy as np
import json
import cv2

TS_DELTA = 4
SELECTION_W = int(W / 3)
SELECTION_H = int(H / 2)

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
    
    polygon_zone = sv.PolygonZone(VIDEOS[st.session_state["location"]]["zone"])
    detections = detections[polygon_zone.trigger(detections)]
    detections = tracker.update_with_detections(detections=detections)

    labels = [
        f"car{tracker_id}" for tracker_id in detections.tracker_id
    ]

    frame = bb.annotate(scene=frame, detections=detections)
    frame = lbl.annotate(scene=frame, detections=detections, labels=labels)
    frame = sv.draw_polygon(frame, VIDEOS[st.session_state["location"]]["zone"], color=sv.Color.RED, thickness=1)

    ids = [int(i) for i in detections.tracker_id]
    if len(ids) == 0:
        stats = {
            'ids': []
        }

        if st.session_state['save_stats_cb']:
            save_step_to_json_file("res.json", st.session_state['step'], stats)

        return stats, frame

    stats = {
        'ids': [] if len(ids) == 0 else ids,
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

    full_frame = full_frame[SELECTION_H:H, SELECTION_W:W]
    # (540, 1280)

    results = None
    playing_mode = st.session_state['playing']
    if playing_mode:
        slicer = sv.InferenceSlicer(callback=callback, overlap_ratio_wh=None, overlap_wh=(100, 100), iou_threshold=0.4)
        results = slicer(image=full_frame)  
        st.session_state['stats'], full_frame = process_results(results, full_frame)

    return full_frame

#---------------------------------------------------

def play():
    st.session_state['playing'] = not st.session_state['playing']

with st.container():
    edit_col, save_col = st.columns(2, vertical_alignment="center", gap="small")
    with edit_col:
        st.button("⏸️" if st.session_state['playing'] else "▶️", on_click=play, type="secondary", icon=None)
    with save_col:
        save_stats = st.checkbox("💾", key="save_stats_cb")
    # stats = st.empty()

    visualization = st.empty()

    total_col, time_step_col, cars_col = st.columns(3, vertical_alignment="top", gap="small")
    with total_col:
        total = st.empty()
    with time_step_col:
        time_step = st.empty()
    with cars_col:
        cars = st.empty()

    if not st.session_state["playing"]:
        frame_results = next_frame()
        if frame_results is not None:
            with visualization:
                st.image(frame_results)
    else:
        while True:
            frame_results = next_frame()
            if frame_results is not None:
                with visualization:
                    st.image(frame_results, use_column_width="always")

                df_total = pd.DataFrame({"Total": [len(st.session_state["stats"]["ids"])]})
                total.write(df_total)
                cars.write(pd.DataFrame({"Cars": st.session_state["stats"]["ids"]}))                        
                time_step.write(pd.DataFrame({"Step": [st.session_state['step']]}))

                st.session_state['step'] += 1

                time.sleep(0.5)
