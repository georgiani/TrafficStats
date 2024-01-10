# pip install git+https://github.com/yt-dlp/yt-dlp.git
# pip install git+https://github.com/georgiani/pafy.git
# pip install opencv-python
import os
import torch
from ultralytics import YOLO
import numpy as np
import gradio as gr
os.environ["PAFY_BACKEND"] = "yt-dlp"
import pafy
import cv2

# Drawing bounding boxes over the picture. Unused probably
def draw_line(img, x1, y1, x2, y2):
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

def get_stream_capture(url):
    video = pafy.new(url, basic=True)
    print(video.videostreams)
    best = video.videostreams[2]
    cap = cv2.VideoCapture(best.url)
    return cap

# "best.pt" is needed in order to run this app
def get_model():
    return YOLO("../best.pt")

# Unirii Intersection
VIDEO_URL = "https://www.youtube.com/watch?v=rs2be3mqryo"

# Try to change this
FRAME_SKIP = 60

model = get_model()
cap = get_stream_capture(VIDEO_URL)

rotation = 20
h_begin = 0
w_begin = 0
h_dif = 50
w_dif = 50
full_h = 360
full_w = 640

editing_mode = True
split_line = False

def process_results(results, flags):
    stats = {
        "all_cars": 0,
        "above_line": 0,
        "below_line": 0
    }

    if "line" in flags:
        if results:
            line_coords = flags["line_coords"]

            for b in results[0].boxes.xyxy:
                # if y1 box below line
                if b[1] > line_coords[1]:
                    stats["below_line"] += 1
                
                # if y2 box above line 
                if b[3] < line_coords[1]:
                    stats["above_line"] += 1


    stats["all_cars"] = len(results[0])
    return stats

# Get the next frame on "every"
def next_frame():
    _, full_frame = cap.read()

    global full_h, full_w
    full_h, full_w = full_frame.shape[0], full_frame.shape[1] 

    hour_frame = full_frame[300:, :100]

    def transform_frame(frame, rot):
        h = frame.shape[0]
        w = frame.shape[1]

        image_center = tuple(np.array(frame.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, rot, 1.0)
        rotated_frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)

        # cropped_frame = rotated_frame[int(h/3):int(h * 2 / 3), int(w/2):int(w * 6 / 7), :]
        cropped_frame = rotated_frame[h_begin: h_begin + h_dif, w_begin:w_begin + w_dif]

        if split_line:
            hc = cropped_frame.shape[0]
            wc = cropped_frame.shape[1]
            draw_line(cropped_frame, 0, int(hc / 2), wc, int(hc / 2))

        return cropped_frame

    full_frame = transform_frame(full_frame, rotation)
    # Conf and Iou values that I found to be
    # the better.
    # TODO: Add a slider to modify the Conf and Iou at
    #   runtime. Check gradio Slider

    results = None
    stats = None
    if not editing_mode:
        results = model.track(
            full_frame,
            persist=True,
            augment=False,
            show_labels=False,
            show_conf=False,
            conf=0.5,
            iou=0.6,
            line_width=1,
            verbose=False,
            save=True
        )

        stats = process_results(
            results,
            {
                "line": split_line,
                "line_coords": (0, int(full_frame.shape[0] / 2), full_frame.shape[1], int(full_frame.shape[0] / 2))
            }
        )

    # Skip frames since not every frame is needed and predicting
    #   on every frame would slow things down
    for _ in range(FRAME_SKIP):
        _, _ = cap.read()
    
    # Saving the prediction since "save=True" and then
    #   substituting the image in the UI with the newly saved one.
    #   Also saving the number of cars in order to update it in the UI
    if editing_mode:
        yield [
            full_frame, 
            "Editing Mode", 
            hour_frame
        ]
    else:
        if results and stats:
            if split_line:
                yield [
                    results[0].save_dir + "\\" + results[0].path, 
                    f"Cars above line: {stats['above_line']}\nCars below line: {stats['below_line']}",
                    hour_frame
                ]
            else:
                yield [
                    results[0].save_dir + "\\" + results[0].path, 
                    f"Cars on selected street: {stats['all_cars']}", 
                    hour_frame
                ]

def rotation_changed(rot):
    global rotation
    rotation = rot

def hb_changed(hb):
    global h_begin
    if h_begin + h_dif < full_h:
        h_begin = hb

def wb_changed(wb):
    global w_begin
    if w_begin + w_dif < full_w:
        w_begin = wb

def hd_changed(hd):
    global h_dif
    if h_begin + h_dif < full_h:
        h_dif = hd

def wd_changed(wd):
    global w_dif
    if w_begin + w_dif < full_w:
        w_dif = wd

def switch_editing_mode():
    global editing_mode
    editing_mode = not editing_mode

    return [
        gr.update(visible=editing_mode), 
        gr.update(visible=editing_mode), 
        gr.update(visible=editing_mode), 
        gr.update(visible=editing_mode), 
        gr.update(visible=editing_mode),
    ]

def switch_split_line():
    global split_line
    split_line = not split_line

with gr.Blocks() as demo:
    editing = gr.Button("Editing Mode")
    frame = gr.Image()

    with gr.Row():
        hour_frame = gr.Image()
        car_count = gr.Label()

    with gr.Column():
        rotation_slider = gr.Slider(0, 180, rotation, label="Rotation", visible=editing_mode)
        rotation_slider.change(rotation_changed, [rotation_slider], None)
        with gr.Row():
            w_begin_slider = gr.Slider(0, 630, w_begin, label="Width Begin", visible=editing_mode)
            w_begin_slider.change(wb_changed, [w_begin_slider], None)
            w_dif_slider = gr.Slider(0, 630, w_dif, label="Width Difference", visible=editing_mode)
            w_dif_slider.change(wd_changed, [w_dif_slider], None)

        with gr.Row():
            h_begin_slider = gr.Slider(0, 350, h_begin, label="Height Begin", visible=editing_mode)
            h_begin_slider.change(hb_changed, [h_begin_slider], None)
            h_dif_slider = gr.Slider(0, 350, h_dif, label="Height Difference", visible=editing_mode)
            h_dif_slider.change(hd_changed, [h_dif_slider], None)
        
        with gr.Row():
            split_check = gr.Checkbox(value=split_line, label="Traffic Line Split")

    split_check.change(switch_split_line)
    editing.click(switch_editing_mode, 
                  None, 
                  [
                      rotation_slider, 
                      w_begin_slider, 
                      w_dif_slider, 
                      h_begin_slider, 
                      h_dif_slider,
                  ]
    )

    demo.load(next_frame, None, [frame, car_count, hour_frame], show_progress=True, every=1, queue=True)

if __name__ == "__main__":
    demo.queue().launch(show_api=False)