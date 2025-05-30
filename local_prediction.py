from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt") 
results = model.predict(
                "images/im20.jpg",
                augment=False,
                show_labels=False,
                show_conf=False,
                conf=0.4,
                iou=0.7,
                line_width=1,
                verbose=False, 
                show=True
            )

next = input()
model = YOLO("yolov5n.pt") 
results = model.predict(
                "images/im20.jpg",
                augment=False,
                show_labels=False,
                show_conf=False,
                conf=0.4,
                iou=0.7,
                line_width=1,
                verbose=False, 
                show=True
            )

next = input()
model = YOLO("yolov8n.pt") 
results = model.predict(
                "images/im20.jpg",
                augment=False,
                show_labels=False,
                show_conf=False,
                conf=0.4,
                iou=0.7,
                line_width=1,
                verbose=False, 
                show=True
            )

next = input()