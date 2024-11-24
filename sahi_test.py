from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path="models/9Nov_Unirii/best.pt",
    confidence_threshold=0.4,
    device="cuda:0", # or 'cpu'
)

result = get_sliced_prediction(
    "full_images/im0.jpg",
    detection_model,
    slice_height = 640,
    slice_width = 640,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

result.export_visuals(export_dir="result/")
for k in result.object_prediction_list:
    print(k.bbox)
    print()