# StrawberryPose_yolov8
The training dataset used in this project is kindly provided by the StrawDI Team (see https://strawdi.github.io/).

## sb_data_preparation
The main function of this folder is to convert various types of data into .txt format that YOLO can directly use. 
The existing conversion functions include the .json format labeled with LABELME and the COCO format included in the SDI dataset.

### json2yolo_example
A sample folder for converting labelme to yolo, containing three images labeled with labelme and their labels in the json folder.
The core file is Label2yolo.ipynb, after changing the Dataset_root and setting the training and testing set folders as required, the conversion results are shown in the labels folder.

### cfg
the .yaml file of the  json2yolo_example dataset.

### mul_COCO2YOLO.py
Convert all COCO format files in a certain folder to YOLO format, simply change the json_path and save_path at the beginning of the file.

### train_sb_pose.py
The training code obtained from the YOLO official website, which can be run in the system background.
Simply run the following commands: nohup python3 train_sb_pose.py &
