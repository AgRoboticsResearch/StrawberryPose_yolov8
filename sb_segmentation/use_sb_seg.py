from ultralytics import YOLO

# Load a model
model = YOLO('/mnt/TempData/brl/pangolin/ruichen/Strawberry_yolov8/model/segment.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image