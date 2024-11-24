from ultralytics import YOLO


model = YOLO("models/20Oct_50_epochs/best.pt") 

while True:
    results = model.predict(
                    "full_images/im0.jpg",
                    augment=False,
                    show_labels=False,
                    show_conf=False,
                    conf=0.4,
                    iou=0.7,
                    line_width=1,
                    verbose=False, 
                    show=True
                )