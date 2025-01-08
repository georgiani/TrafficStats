from ultralytics import YOLO
from ultralytics.engine.results import Results

class Model:
    def __init__(self, typ):
        self.speed = 0.0
        self._pretrained_weights = "models/21Dec_SingleClass640_75/best.pt"        
        self._model = YOLO(self._pretrained_weights).to("cuda")


    def predict(self, frame):
        result = self._model.predict(
            frame,
            conf=0.4,
            imgsz=640,
            verbose=False
        )[0]

        self.speed = max(self.speed, (result.speed['preprocess'] + result.speed['inference'] + result.speed['postprocess']))
        return result