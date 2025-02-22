import cv2
import torch
import time
from ultralytics import YOLO

# Force CUDA usage
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 model and move to GPU
model = YOLO("yolov8n.pt").to(device)  # Use 'yolov8s.pt' for better accuracy

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

# Define vehicle classes (Car, Motorcycle, Bus, Truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Open video file or webcam
video_path = "testv.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Set up video writer for output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# Process video
while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Convert OpenCV frame (BGR) to RGB and ensure correct input format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

 # Run YOLO inference on GPU with batch processing
    results = model.predict(frame_rgb, device="cuda", imgsz=640, conf=0.2, iou = 0.5, half=True, verbose=False)

    vehicle_count = 0

    # Draw bounding boxes
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = int(cls)

            # Only process vehicle classes
            if class_id in VEHICLE_CLASSES and conf > 0.2:
                vehicle_count += 1
                color = (0, 255, 0)  # Green for vehicles
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display FPS and vehicle count
    fps_text = f"FPS: {1 / (time.time() - start_time):.2f}"
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, fps_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show video output
    cv2.imshow("Live Vehicle Detection", frame)
    out.write(frame)  # Write frame to output video

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
