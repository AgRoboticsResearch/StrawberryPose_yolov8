from ultralytics import YOLO

# model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
# results = model.train(data='/home/rcli/code/cfg/ko8.yaml', epochs=100, imgsz=640)


model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
# Train the model
results = model.train(data='/home/rcli/code/cfg/strawberry.yaml', epochs=300, imgsz=640)