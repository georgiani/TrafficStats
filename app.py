# pip install git+https://github.com/yt-dlp/yt-dlp.git
# pip install git+https://github.com/georgiani/pafy.git
import pafy
# pip install opencv-python
import cv2
import os
import torch
from ultralytics import YOLO
import numpy as np
import gradio as gr
os.environ["PAFY_BACKEND"] = "yt-dlp"

# Drawing bounding boxes over the picture. Unused probably
def draw_bounding_box(img, xmin, ymin, xmax, ymax):
    xmin = round(xmin)
    ymin = round(ymin)
    xmax = round(xmax)
    ymax = round(ymax)

    start_point = (xmin, ymin) 
    end_point = (xmax, ymax) 
    color = (255, 0, 0) 

    cv2.rectangle(img, start_point, end_point, color, 1)

def get_stream_capture(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)
    return cap

# "best.pt" is needed in order to run this app
def get_model():
    return YOLO("best.pt")

# Unirii Intersection
VIDEO_URL = "https://www.youtube.com/watch?v=rs2be3mqryo"

# Try to change this
FRAME_SKIP = 30

model = get_model()
cap = get_stream_capture(VIDEO_URL)

count_cars = gr.State(0)

# Get the next frame on "every"
def next_frame():
    success, frame = cap.read()

    w = 1280
    h = 720

    if success:
        full_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        # TODO: generalize the cropping
        quarter_frame = full_frame[250:650, 650:1200]

        image_center = tuple(np.array(quarter_frame.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, 20, 1.0)
        rotated_frame = cv2.warpAffine(quarter_frame, rot_mat, quarter_frame.shape[1::-1], flags=cv2.INTER_LINEAR)

        half_rotated_frame = rotated_frame[170:310,:400] 

        # Conf and Iou values that I found to be
        # the better.
        # TODO: Add a slider to modify the Conf and Iou at
        #   runtime. Check gradio Slider
        results = model.track(
            half_rotated_frame,
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

        # Skip frames since not every frame is needed and predicting
        #   on every frame would slow things down
        for _ in range(FRAME_SKIP):
            _, _ = cap.read()
        
        # Saving the prediction since "save=True" and then
        #   substituting the image in the UI with the newly saved one.
        #   Also saving the number of cars in order to update it in the UI
        yield [results[0].save_dir + "\\" + results[0].path, len(results[0])]
    else:
        yield None

with gr.Blocks() as demo:
    ended = False

    def end():
        cap.release()
        ended = True
        print(ended)

    end_btn = gr.Button("End").click(end)

    frame = gr.Image()

    car_count = gr.Label()

    # Refresh the image and predictions very 0.5 seconds
    #   Supposing video is 60 fps, then showing and skipping 30 frames every half a second
    #   would not delay the show, but this is not taking into account
    #   the time to predict and that every 0.5 I'm showing one frame and skipping 30, so 31 frames advance
    # I need to find a way to show the real time. Maybe by cropping the time in
    #   the bottom left corner and using it in the code
    demo.load(next_frame, None, [frame, car_count], show_progress=True, every=0.5)

if __name__ == "__main__":
    demo.queue().launch(show_api=False)