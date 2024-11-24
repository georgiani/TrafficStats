from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.engine.results import Results

class Model:
    def __init__(self, typ):
        self._pretrained_weights = "models/15Nov_Larger/best.pt"        
        if typ == "yolo":
            self._model = YOLO(self._pretrained_weights).to("cuda")
        elif typ == "sahi":
            self._model = AutoDetectionModel.from_pretrained(
                model_type='yolov8',
                model_path=self._pretrained_weights,
                confidence_threshold=0.4,
                device="cuda:0",
            )
        
        self.typ = typ


    def predict(self, frame):
        if self.typ == "sahi":
            result = get_sliced_prediction(
                frame,
                self._model,
                slice_height = 640,
                slice_width = 640,
                overlap_height_ratio = 0.2,
                overlap_width_ratio = 0.2
            )

            boxes = [b.bbox for b in result.object_prediction_list]
            names = {0: 'car'}
            results = Results(frame, boxes=boxes, names=names)

            return result.object_prediction_list
        else:
            return self._model.predict(
                frame,
                conf=0.4,
                imgsz=1280,
                iou=0.7,
                verbose=False
            )[0]