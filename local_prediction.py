from ultralytics import YOLO
import cv2

model = YOLO("models/9Nov_Unirii/best.pt") 
#model = YOLO("models/better_aug_cctv_cars/best.pt") 

while True:
    results = model.predict(
                    "images/im11.jpg",
                    augment=False,
                    show_labels=False,
                    show_conf=False,
                    conf=0.5,
                    iou=0.7,
                    line_width=1,
                    verbose=False, 
                    show=True
                )