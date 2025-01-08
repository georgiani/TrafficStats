from ultralytics import YOLO
from ultralytics.engine.results import Results

class Model:
    def __init__(self, typ):
        self._pretrained_weights = "best.pt"        
        self._model = YOLO(self._pretrained_weights).to("cuda")

    def predict(self, frame):
        return self._model.predict(
            frame,
            conf=0.4,
            imgsz=1280,
            iou=0.7,
            verbose=False,
            save=True,
        )[0]