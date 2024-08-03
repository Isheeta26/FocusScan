import torch
from ultralytics import YOLO

def load_yolo_model(model_path="yolov8n.pt"):
    return YOLO(model_path)

def run_yolo_inference(model, image_path):
    return model(image_path)
