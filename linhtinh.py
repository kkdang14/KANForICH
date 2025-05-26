from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load a pre-trained YOLOv8 model

results = model.predict(source="0", show=True, conf=0.25)  # Predict on webcam feed with confidence threshold of 0.25