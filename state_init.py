import json
from get_stream import get_stream_capture
from Model import Model
import supervision as sv

VIDEO_URL = "https://www.youtube.com/watch?v=rs2be3mqryo"
model = Model("yolo")
cap = get_stream_capture(VIDEO_URL)
bb = sv.BoundingBoxAnnotator(thickness=1)
lbl = sv.LabelAnnotator(text_padding=0, text_scale=0.2)
tracker = sv.ByteTrack(frame_rate=1)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
W = 1280
H = 720

print("rerun")

with open("res.json", "w") as res_file:
    json.dump({}, res_file)
