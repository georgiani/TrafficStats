from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

if __name__ == "__main__": 
    CONF = {
    "architecture": "YOLO V11",
    "dataset": "BaseDSWithUniriiIntersectionSmallerSize",
    "epochs": 50,
    }

    wandb.init(
        project="TrafficStats",
        config=CONF,
        name="YOLOV11 Single Class Train 21 December 320x320 50 Epochs YoloV11N"
    )


    model = YOLO("yolo11n.pt")

    add_wandb_callback(model, enable_model_checkpointing=True)

    results = model.train(project="TrafficStats", data="E:\Projects\datasets\Tfs21Dec\data.yaml", batch=-1, epochs=50, imgsz=320, save=True, save_period=20, device=0, single_cls=True, patience=10)

    wandb.finish()
