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

def draw_bounding_box(img, xmin, ymin, xmax, ymax):
    xmin = round(xmin)
    ymin = round(ymin)
    xmax = round(xmax)
    ymax = round(ymax)

    start_point = (xmin, ymin) 
    end_point = (xmax, ymax) 
    color = (255, 0, 0) 

    cv2.rectangle(img, start_point, end_point, color, 1)

model = YOLO("best.pt")
os.environ["PAFY_BACKEND"] = "yt-dlp"
url = "https://www.youtube.com/watch?v=rs2be3mqryo"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

FRAME_SKIP = 10
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    w = 1280
    h = 720

    if success:

        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        frame = frame[250:650, 650:1200]
        
        image_center = tuple(np.array(frame.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, 20, 1.0)
        frame = cv2.warpAffine(frame, rot_mat, frame.shape[1::-1], flags=cv2.INTER_LINEAR)

        half_frame = frame[170:310,:400] 

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            half_frame, 
            show=True,
            persist=True,
            augment=False,
            show_labels=False,
            show_conf=False,
            conf=0.5,
            iou=0.6,
            line_width=1,
            verbose=False
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

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

    # skip 30 frames
    for _ in range(FRAME_SKIP):
        _, _ = cap.read()

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()