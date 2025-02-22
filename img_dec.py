import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (you can replace with a custom model)
model = YOLO("yolov8l.pt")  # Use "yolov8n.pt" for faster inference or a custom model

# Define COCO class names (YOLOv8 is trained on COCO dataset)
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
                "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
                "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
                "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
                "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
                "toothbrush"]

# Define vehicle class IDs (based on COCO dataset)
VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

# Load image
image_path = "test1.jpg"  # Change to your image path
image = cv2.imread(image_path)

# Run YOLOv8 inference
results = model(image, iou = 0.5, conf = 0.2)

# Process detections
for r in results:
    for box in r.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(cls)
        confidence = float(conf)

        # Only process vehicle classes
        if class_id in VEHICLE_CLASSES and confidence > 0.2:  # Adjust confidence threshold if needed
            label = f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            color = (0, 255, 0)  # Green color for vehicles

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Display the output
cv2.imshow("Vehicle Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
