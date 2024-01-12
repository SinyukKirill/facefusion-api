from ultralytics import YOLO
model = YOLO('./models/yolov8n-face.pt')
model.export(format='engine', imgsz=640, opset=12)
