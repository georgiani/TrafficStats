import time
import supervision as sv
import numpy as np
import json

from Model import Model
from get_stream import get_stream_capture

def save_step_to_json_file(filename, step, data):
    with open(filename, "r") as res_file_read:
        results_json = json.load(res_file_read)

    keys = [k for k in results_json.keys()]

    if not keys:
        # first time
        results_json[step] = data
        with open(filename, "w") as res_file_write:
            res_file_write.write(json.dumps(results_json))

    if keys:
        if "ts" in results_json[keys[-1]] and "ts" in data:
            if data["ts"] - results_json[keys[-1]]["ts"] > TS_DELTA:
                results_json[step] = data
            
                with open(filename, "w") as res_file_write:
                    res_file_write.write(json.dumps(results_json))

def process_results(results, step, zone):
    ts = time.time()
    detections = results
    
    polygon_zone = sv.PolygonZone(zone)
    detections = detections[polygon_zone.trigger(detections)]
    detections = tracker.update_with_detections(detections=detections)

    ids = [int(i) for i in detections.tracker_id]
    stats = {
        'ids': [] if len(ids) == 0 else ids,
        'ts': ts
    }

    save_step_to_json_file("res.json", step, stats)

    return stats

def callback(image_slice: np.ndarray) -> sv.Detections:
    ultralytics_results = model.predict(image_slice)
    sv_detections = sv.Detections.from_ultralytics(ultralytics_results)
    return sv_detections

def next_frame(H, W, SELECTION_H, SELECTION_W, step, zone):
    full_frame = cap.read()

    if full_frame is None:
        return None

    full_frame = full_frame[SELECTION_H:H, SELECTION_W:W]
    
    slicer = sv.InferenceSlicer(callback=callback, overlap_ratio_wh=None, overlap_wh=(128, 128), iou_threshold=0.7)
    results = slicer(image=full_frame)
    stats = process_results(results, step, zone)

    return stats


if __name__ == "__main__":
    with open("res.json", "w") as res_file:
        json.dump({}, res_file)

    # Unirii
    VIDEO_URL = "https://www.youtube.com/watch?v=rs2be3mqryo"
    VIDEO_URL2 = "https://www.youtube.com/watch?v=1dLkP_nwZLo"

    model = Model("yolo")
    cap = get_stream_capture(VIDEO_URL2)
    tracker = sv.ByteTrack(frame_rate=1)

    W = 1280
    H = 720
    TS_DELTA = 4
    SELECTION_W = int(W / 3)
    SELECTION_H = int(H / 2)
    ZONE1 = np.array([[590, 205], [690, 140], [360, 20], [250, 70]])
    ZONE2 = np.array([[0, 360], [300, 360], [650, 100], [400, 100]])

    step = 0
    while True:
        stats = next_frame(H, W, SELECTION_H, SELECTION_W, step, ZONE2)

        if stats is not None:
            print(stats)
            step += 1
            time.sleep(2.5)