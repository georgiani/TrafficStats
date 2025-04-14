import time

from matplotlib import pyplot as plt
import torch
from generator_model.train_nn import FeedforwardNNModel
import state_init
from state_init import cap, bb, lbl, tracker, model, W, H, VIDEOS
import supervision as sv
import streamlit as st
import seaborn as sns
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

        return stats, frame

    stats = {
        'ids': [] if len(ids) == 0 else ids,
        'ts': ts
    }

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

def forecast(forecast_visualization):
    fn = FeedforwardNNModel(6, 12, 1)
    fn.load_state_dict(torch.load("./generator_model/models/NN/model_C5_10_aprilie_6", weights_only=True))
    fn.eval()

    ts_now = pd.Timestamp.now()
    cars_now = st.session_state['stats']['ids']
    forecast_period = st.session_state.period
    num_cars = len(cars_now)
    
    green_now = False # red
    seconds_passed = 0
    traffic = [[] for _ in range(forecast_period + 1)]
    normalization_hour = ts_now.hour
    for t in pd.date_range(ts_now, ts_now + pd.to_timedelta(forecast_period, unit='h'), freq='3s'):
        d = t.day_of_week
        h = t.hour
        is_night = 1 if 21 <= h or h < 5 else 0 # between 21 and 5
        is_morning = 1 if 5 <= h < 10 else 0 # between 5 and 10
        is_day = 1 if 10 <= h < 18 else 0 # between 10 and 18
        is_evening = 1 if 18 <= h < 21 else 0 # between 18 and 21
        probs = fn.predict(torch.Tensor([d, is_night, is_morning, is_day, is_evening, num_cars]))
        probs = probs.detach().numpy()
        prob_value = probs[0]
        if prob_value > 0.5 and not green_now:
            num_cars += 1
        
        if green_now:
            if num_cars - 2 > 0:
                num_cars -= 2
            else:
                num_cars = 1
        index = h - normalization_hour
        if index < 0:
            index = h - normalization_hour + 24

        traffic[index].append(num_cars)

        seconds_passed += 3
        if (green_now and seconds_passed > st.session_state.green) or (not green_now and seconds_passed > st.session_state.red):
            seconds_passed = 0
            green_now = not green_now

    for i, tf in enumerate(traffic):
        traffic[i] = np.median(tf)

    print(traffic)

    df = pd.DataFrame.from_dict({"ts": [i for i in range(0, len(traffic))], "traffic": traffic})

    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="ts", y="traffic", ax=ax)
    forecast_visualization.write(fig)
    st.pyplot(fig)

def play():
    st.session_state['playing'] = not st.session_state['playing']

with st.container():
    visualization_col, forecast_col = st.columns(2, vertical_alignment="center", gap="small") 

    with visualization_col:
        st.button("⏸️" if st.session_state['playing'] else "▶️", on_click=play, type="secondary", icon=None)
        visualization = st.empty()

        total_col, time_step_col, cars_col = st.columns(3, vertical_alignment="top", gap="small")
        with total_col:
            total = st.empty()
        with time_step_col:
            time_step = st.empty()
        with cars_col:
            cars = st.empty()

    with forecast_col:
        st.number_input("Traffic Light Red Period", 0, 240, 30, 1, key="red")
        st.number_input("Traffic Light Green Period", 0, 240, 30, 1, key="green")
        st.number_input("Forecast Period", 1, 10, 3, 1, key="period")
        forecast_visualization = st.empty()
        st.button("Forecast With Set Values", on_click=forecast, type="secondary", icon=None, args=(forecast_visualization,))

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
