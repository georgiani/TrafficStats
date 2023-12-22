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

def get_model():
    return YOLO("best.pt")

VIDEO_URL = "https://www.youtube.com/watch?v=rs2be3mqryo"
FRAME_SKIP = 30
model = get_model()
cap = get_stream_capture(VIDEO_URL)

count_cars = gr.State(0)

def next_frame():
    success, frame = cap.read()

    w = 1280
    h = 720

    if success:
        full_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        quarter_frame = full_frame[250:650, 650:1200]

        image_center = tuple(np.array(quarter_frame.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, 20, 1.0)
        rotated_frame = cv2.warpAffine(quarter_frame, rot_mat, quarter_frame.shape[1::-1], flags=cv2.INTER_LINEAR)

        half_rotated_frame = rotated_frame[170:310,:400] 

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

        # # Visualize the results on the frame
        # for p in results[0]:
        #     boxes = p.boxes.xyxy[0]
        #     x1 = boxes[0].item()
        #     x2 = boxes[2].item()
        #     y1 = boxes[1].item()
        #     y2 = boxes[3].item()
        #     draw_bounding_box(frame, x1, y1, x2, y2)

        # cv2.namedWindow("tracking",cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("tracking", 1280, 720)
        # cv2.imshow("tracking", frame)
        for _ in range(FRAME_SKIP):
            _, _ = cap.read()
        
        # count_cars.value = len(results[0])
        yield [results[0].save_dir + "\\" + results[0].path, len(results[0])]
    else:
        yield None


def destroy():
    pass

with gr.Blocks() as demo:
    ended = False

    def end():
        cap.release()
        ended = True
        print(ended)

    end_btn = gr.Button("End").click(end)

    frame = gr.Image()

    car_count = gr.Label()

    demo.load(next_frame, None, [frame, car_count], show_progress=True, every=0.5)

if __name__ == "__main__":
    demo.queue().launch(show_api=False)