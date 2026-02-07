import cv2
import time
from ultralytics import YOLO

# Load SOTA Model (YOLOv8 Nano - The "Fastest" standard model)
import os

model_name = 'yolov8n.pt'
if os.path.exists(model_name):
    print(f"✅ Found Local Model: {model_name}")
else:
    print(f"⬇️ Model not found locally. Downloading {model_name} from Ultralytics...")

# Ultralytics will automatically download if not found, but we made it explicit above
model = YOLO(model_name) 

# Video Path
video_path = "website/assets/test1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening video: {video_path}")
    exit()

print("Starting SOTA Benchmark...")
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Resize to standard 640x640 for fair comparison
    frame = cv2.resize(frame, (640, 640))
    
    # Run Inference
    results = model(frame, verbose=False)
    
    # Plot Results
    annotated_frame = results[0].plot()
    
    # Calculate Real FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time
    
    # Display FPS
    cv2.putText(annotated_frame, f"SOTA FPS: {fps:.1f}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("SOTA Benchmark (YOLOv8n)", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
