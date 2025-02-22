from ultralytics import YOLO

# Load YOLOv8 model (small version for speed, use 'yolov8n' for even faster inference)
model = YOLO("yolov8s.pt")  
