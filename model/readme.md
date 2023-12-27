pose: the origin model for detecting the keypoints of strawberries
pose_ud: add data augmentation of flipping up and down
segment: strawberry segmentation model trained on SDI dataset
simu: strawberry detecting model in virtual scenes
how to use:

from ultralytics import YOLO

# Load a model
model = YOLO('/mnt/TempData/brl/pangolin/ruichen/sb_pose_detection/model/segment.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

################################################################################


