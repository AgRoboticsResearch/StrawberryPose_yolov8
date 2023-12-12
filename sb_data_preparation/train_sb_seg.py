from ultralytics import YOLO

model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
# Train the model
results = model.train(data='/mnt/TempData/brl/pangolin/ruichen/sb_pose_detection/cfg/seg_strawberry.yaml', epochs=300, imgsz=640)

# model = YOLO('/mnt/data0/ruichen/sb_pose_detection/results/runs/pose/train10/weights/best.pt')  # load a pretrained model (recommended for training)
# # Train the model
# results = model.train(data='/mnt/data0/ruichen/sb_pose_detection/cfg/V4.yaml', epochs=100, imgsz=640)