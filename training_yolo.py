from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolov9c.pt")
wandb.init(project="camus", job_type="training")

add_wandb_callback(model, enable_model_checkpointing=False)

results = model.train(
    project="camus",
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=[0, 1],
    workers=8,
    patience=10,
)

model.val()
wandb.finish()
