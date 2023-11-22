#!/usr/bin/env python
"""
Zhenghao Fei
"""

"""
Ruichen Li
added the function of outputting 3D coordinates of picking points predicted by model version_2
"""

import numpy as np
import rospy
import cv2 as cv
import copy
import math
import time
import os
import sys
from std_msgs.msg import Header, Int8
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import CameraInfo
from ultralytics import YOLO
from utils import projections as proj
from cv_bridge import CvBridge
from sb_robot_msgs.msg import Fruits
# from sb_robot_msgs.msg import FruitKeypoints
# from sb_robot_msgs.msg import FruitsKeypoints
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


class StrawberryDetection(object):
    def __init__(self, rate=30):
        rospy.init_node("strawberry_detection", anonymous=True, log_level=rospy.INFO)
        self.namespace = rospy.get_namespace()
        self.get_ros_parameters()
        self.rate = rospy.Rate(rate)
        self.initialize_variables()

        rospy.loginfo(rospy.get_name() + " Start")
        rospy.loginfo(rospy.get_name() + " Namespace: " + self.namespace)

        self.define_publishers()
        self.define_subscribers()
        self.initialize_rviz_fruits_marker()
        self.run()

    def define_publishers(self):
        # Define publisher
        fruit_topic = "/perception/fruits"
        self.fruits_publisher = rospy.Publisher(fruit_topic, Fruits, queue_size=1)

        visualize_image_topic = (
            "/" + self.robot_name + "/visualization/strawberry_detection"
        ).replace("//", "/")
        self.visualize_image_publisher = rospy.Publisher(
            visualize_image_topic, Image, queue_size=1
        )
        fruits_rviz_topic = "/rviz/fruits"
        self.fruits_rviz_publisher = rospy.Publisher(
            fruits_rviz_topic, Marker, queue_size=10
        )

    def define_subscribers(self):
        # Define subscribers
        front_rgb_topic = "/camera/color/image_raw"
        front_rgb_compressed_topic = "/camera/color/image_raw/compressed"
        front_rgb_intrinsic_topic = "/camera/color/camera_info"

        depth_image_topic = "/camera/aligned_depth_to_color/image_raw"
        depth_image_cam_info_topic = "/camera/aligned_depth_to_color/camera_info"

        if not self.use_compressed_img:
            rospy.Subscriber(
                front_rgb_topic,
                Image,
                self.front_rgb_image_callback,
                False,
                queue_size=1,
                buff_size=2**24,
            )
        else:
            rospy.Subscriber(
                front_rgb_compressed_topic,
                CompressedImage,
                self.front_rgb_image_callback,
                True,
                queue_size=1,
                buff_size=2**24,
            )

        rospy.Subscriber(
            depth_image_topic,
            Image,
            self.depth_image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        self.front_rgb_K_sub = rospy.Subscriber(
            front_rgb_intrinsic_topic,
            CameraInfo,
            self.front_rgb_intrinsic_callback,
            queue_size=1,
        )

        self.depth_img_K_sub = rospy.Subscriber(
            depth_image_cam_info_topic,
            CameraInfo,
            self.depth_image_cam_info_callback,
            queue_size=1,
        )

    def depth_image_cam_info_callback(self, cam_info_msg):
        self.depth_image_cam_info = np.asarray(cam_info_msg.K).reshape([3, 3])
        print(
            rospy.get_name() + " depth_image_cam_info: \n ",
            self.depth_image_cam_info,
        )
        self.depth_img_K_sub.unregister()

    def get_ros_parameters(self):
        self.robot_name = self.namespace[1:-1]
        self.yolo_model_path = rospy.get_param("~yolo_model_path", "")
        self.use_compressed_img = rospy.get_param("~use_compressed_img", False)
        self.classnames_file = rospy.get_param("~classnames_file", "")

        self.viz_flag = rospy.get_param("~viz_flag", True)
        self.resized_dim = rospy.get_param("~resized_dim", "420")
        self.conf = rospy.get_param("~conf", "")
        self.nms_th = rospy.get_param("~nms_th", "")
        self.profile = rospy.get_param("~profile", True)

        rospy.loginfo(
            rospy.get_name() + " yolo_model_path: " + str(self.yolo_model_path)
        )
        rospy.loginfo(
            rospy.get_name() + " use_compressed_img: " + str(self.use_compressed_img)
        )
        rospy.loginfo(
            rospy.get_name() + " classnames_file: " + str(self.classnames_file)
        )
        rospy.loginfo(rospy.get_name() + " viz_flag: " + str(self.viz_flag))
        rospy.loginfo(rospy.get_name() + " resized_dim: " + str(self.resized_dim))
        rospy.loginfo(rospy.get_name() + " conf: " + str(self.conf))
        rospy.loginfo(rospy.get_name() + " nms_th: " + str(self.nms_th))
        rospy.loginfo(rospy.get_name() + " profile: " + str(self.profile))

    def initialize_variables(self):
        self.bridge = CvBridge()
        self.front_rgb_image = None
        self.front_rgb_intrinsic = None

        self.depth_image = None
        self.depth_image_cam_info = None

        self.rgb_ready = False
        self.depth_ready = False

        self.depth_img_width = None
        self.depth_img_height = None

        self.fruit_conf = 0.5

        self.model = YOLO(self.yolo_model_path)
        self.frame_id = "camera_color_optical_frame"
        self.initialize_rviz_fruits_marker()

    def inference(self, model, source_img):
        """
        model: loaded self.model
        source_img: image to be infered
        return: image after inference
        """
        # Inference
        result = model(source_img)
        bbox_np = (result[0].boxes.data).cpu().numpy()
        return bbox_np
    
    def get_keypoints(self, model, source_img):
        keypoints_error = "keypoints_error"
        result = model(source_img)
        final_points = []
        # print("keypoints:",len(result[0].keypoints.data))
        if len(result[0].keypoints.data[0]) > 0:
            keypoints = (result[0].keypoints.data).cpu().numpy()
            print("keypoints:",keypoints)
            for i in range(len(keypoints)):
                if keypoints[i,0,0] > 0:
                    final_points.append(keypoints[i])
            return final_points   
        else:
            rospy.logwarn(rospy.get_name() + " no strawberry ")
            return keypoints_error
    
    def get_bboxes_and_keypoints(self, model, source_img):
        """
        model: loaded self.model
        source_img: image to be infered
        return: boxes and keypoints after filtering
        """
        result = model(source_img)
        bbox = (result[0].boxes.data).cpu().numpy()
        keypoints = (result[0].keypoints.data).cpu().numpy()
        bbox_np = []
        keypoints_np = []

        for i in range(len(bbox)):
            if bbox[i,5] == 0: # screen out ripe strawberry
                if bbox[i,4] > self.fruit_conf: # the confidence of bbox needs to exceed 0.5 
                    bbox_np.append(bbox[i])
                    keypoints_np.append(keypoints[i])
        print("bbox_np:",bbox_np)
        print("keypoints_np:",keypoints_np)
        return bbox_np, keypoints_np



    def publish_fruits(self, boxes, depths=None):
        pass

    def front_rgb_intrinsic_callback(self, intrinsic_msg):
        self.front_rgb_intrinsic = np.asarray(intrinsic_msg.P).reshape([3, 4])[:3, :3]
        rospy.loginfo(rospy.get_name() + " front rgb intrinsic initialized ")
        self.front_rgb_K_sub.unregister()

    def depth_image_callback(self, depth_msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )
        self.depth_img_width = self.depth_image.shape[1]
        self.depth_img_height = self.depth_image.shape[0]
        self.depth_ready = True

    def front_rgb_image_callback(self, image_msg, compressed=False):
        tic = time.time()
        if not compressed:
            self.front_rgb_image = self.bridge.imgmsg_to_cv2(
                image_msg, desired_encoding="passthrough"
            )
            self.front_rgb_image = cv.cvtColor(self.front_rgb_image, cv.COLOR_BGR2RGB)
        else:
            self.front_rgb_image = self.bridge.compressed_imgmsg_to_cv2(image_msg)
        self.rgb_img_width = self.front_rgb_image.shape[1]
        self.rgb_img_height = self.front_rgb_image.shape[0]
        self.rgb_ready = True

    def yolo_bbox_to_pixels(self, yolo_bbox, width_img, height_img):
        """
        yolo_bbox: [x_min, y_min, x_max, y_max, score, score, cls_id] in normalized size
        return: [x1, x2, y1, y2] in pixels

        """
        x1 = width_img * max(0, yolo_bbox[0])
        x2 = width_img * yolo_bbox[2]
        y1 = height_img * yolo_bbox[1]
        y2 = height_img * min(1, yolo_bbox[3])
        return [x1, y1, x2, y2]

    def get_interest_area(self, bbox_pixels, scale_factor):
        """
        bbox_pixels: [x1, y1, x2, y2] in pixels
        scale_factor: factor to scale the size of center area to calculate the depth
        """
        x1, y1, x2, y2 = bbox_pixels[:4]
        x1_dis_rec = x1 + (x2 - x1) * ((scale_factor - 1) / 2) / scale_factor
        y1_dis_rec = y1 + (y2 - y1) * ((scale_factor - 1) / 2) / scale_factor
        x2_dis_rec = x1 + (x2 - x1) * ((scale_factor - 1) / 2 + 1) / scale_factor
        y2_dis_rec = y1 + (y2 - y1) * ((scale_factor - 1) / 2 + 1) / scale_factor

        return x1_dis_rec, y1_dis_rec, x2_dis_rec, y2_dis_rec

    def get_object_depth(
        self,
        depth_image,
        bounding_boxes,
        scale_factor=10,
        depth_factor=0.001,
        profile=False,
    ):
        """
        bouding_box: [x_min, y_min, x_max, y_max, score, score, cls_id] in normalized size
        scale_factor: factor to scale the size of center area to calculate the depth
        depth_factor: factor to display the final result in the measurement of meter
        return:
        depth_array: depth of each detected item
        bbox_pixels: [x1, x2, y1, y2] in pixels
        """
        tic = time.time()
        width_img = depth_image.shape[1]
        height_img = depth_image.shape[0]
        depth_array = []

        for bounding_box in bounding_boxes:
            # print("bounding_box is: ", bounding_box)
            bbox_pixels = self.yolo_bbox_to_pixels(bounding_box, width_img, height_img)
            x1_dis_rec, y1_dis_rec, x2_dis_rec, y2_dis_rec = self.get_interest_area(
                bbox_pixels, scale_factor
            )
            depth = np.mean(
                depth_image[
                    int(y1_dis_rec) : int(y2_dis_rec), int(x1_dis_rec) : int(x2_dis_rec)
                ]
            )
            depth = depth * depth_factor
            depth_array.append(depth)
        if profile:
            rospy.logdebug(
                rospy.get_name()
                + " get_object_depth: %.1f ms" % ((time.time() - tic) * 1000)
            )
        return depth_array, bbox_pixels

    def draw_strawberry_detection_image(self, image, bbox, keypoint, fruits_3d_info, class_names):
        thresh = 0.6
        for i in range(len(bbox)):
            coco_bbox = bbox[i]
            conf = coco_bbox[4]
            cls = coco_bbox[5]
            point = keypoint[i]
            # print(i,point)
            image = cv.rectangle(
                image,
                (int(coco_bbox[0]), int(coco_bbox[1])),
                (int(coco_bbox[2]), int(coco_bbox[3])),
                (255, 255, 0), # BGR not RGB
                3,
            )
            image = cv.circle(
                image,
                (int(point[0,0]), int(point[0,1])),
                5,
                (0,0,255),
                cv.FILLED,
            )
            image = cv.circle(
                image,
                (int(point[1,0]), int(point[1,1])),
                5,
                (0,255,255),
                cv.FILLED,
            )
            image = cv.circle(
                image,
                (int(point[2,0]), int(point[2,1])),
                5,
                (0,255,0),
                cv.FILLED,
            )
            image = cv.circle(
                image,
                (int(point[3,0]), int(point[3,1])),
                5,
                (255,0,0),
                cv.FILLED,
            )
            image = cv.circle(
                image,
                (int(point[4,0]), int(point[4,1])),
                5,
                (128,0,128),
                cv.FILLED,
            )
            image = cv.line(
                image,
                (int(point[1,0]), int(point[1,1])),
                (int(point[2,0]), int(point[2,1])),
                (255, 255 ,255),
                2,
            )
            image = cv.line(
                image,
                (int(point[3,0]), int(point[3,1])),
                (int(point[4,0]), int(point[4,1])),
                (255, 255 ,255),
                2,
            )

            # draw text on box
            if len(fruits_3d_info) == len(bbox):
                fruit_center = fruits_3d_info[i][3]
                text = "%.2f, %.2f, %.2f" % (
                    fruit_center[0],
                    fruit_center[1],
                    fruit_center[2],
                )
                image = cv.putText(
                    image,
                    text,
                    (int(coco_bbox[0]), int(coco_bbox[1])),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    5,
                )
        return image

    def encode_and_publish_image(self, image, image_publisher, profile=False):
        # publish image
        tic = time.time()
        image_cv_msg = self.bridge.cv2_to_imgmsg(image, encoding="passthrough")
        image_publisher.publish(image_cv_msg)
        if profile:
            rospy.logdebug(
                rospy.get_name()
                + " encode_and_publish_image: %.1f ms" % ((time.time() - tic) * 1000)
            )

    def process_bbox(self, boxes):
        """
        Description: process original bbox so that it is numpy array in cpu, and organized as normalized size
        args
            boxes: torch list, results from yolo v4
        return:
            boxes_np: numpy array []
        """
        # process bbox
        boxes_np = boxes[0].cpu().numpy()
        # convert the xyxy in box from pixel level to normalized size
        boxes_np[:, :4] = boxes_np[:, :4] / self.resized_dim
        return boxes_np

    def preprocess_and_inference(self, image, profile=False):
        """
        return: [x_min, y_min, x_max, y_max, conf, class_id] normalized scale
                x means horizontal axis, y means vertival axis
        """
        # resize image
        tic = time.time()
        boxes = self.inference(self.model, image)
        if profile:
            rospy.logdebug(
                rospy.get_name() + " inference: %.1f ms" % ((time.time() - tic) * 1000)
            )
        return boxes

    def get_fruit_pathches(self, depth_img, bbox_np, valid_z_min=0, valid_z_max=500):
        """
        input:
            depth_img: depth image
            bbox_np: bbox from yolo [x1, y1, x2, y2, conf, cls]
            valid_z_min: minimum valid depth in mm
        return:
            list of fruit patches
        """
        fruit_patches = []
        res = 5
        for i in range(len(bbox_np)):
            coco_bbox = bbox_np[i]
            conf = coco_bbox[4]
            cls = coco_bbox[5]
            box_xs = np.arange(round(coco_bbox[0]), round(coco_bbox[2]), res) # 方框横坐标等分
            box_ys = np.arange(round(coco_bbox[1]), round(coco_bbox[3]), res) # 方框纵坐标等分
            grids = np.meshgrid(box_xs, box_ys)
            grids = np.vstack((grids[0].flatten(), grids[1].flatten())).T
            fruit_patch = np.hstack(
                (grids, depth_img[grids[:, 1], grids[:, 0]].reshape(-1, 1)) # 这里不知道depth_img格式，姑且认为是x|y|z三列
            )
            fruit_patch = fruit_patch[fruit_patch[:, 2] > valid_z_min]
            fruit_patch = fruit_patch[fruit_patch[:, 2] < valid_z_max]
            fruit_patches.append(fruit_patch)
        
        return fruit_patches

    def get_fruit_points(self, fruit_patches, depth_img, depth_K):
        """
        input:
            fruit_patches: list of fruit patches
            depth_img: depth image
            depth_K: depth camera intrinsic
        return:
            list of fruit points
        """
        fruit_points = []
        for fruit_patch in fruit_patches:
            fruit_uvs = np.vstack((fruit_patch[:, 1], fruit_patch[:, 0])).T # y|x?
            # print("fruit_uvs:", fruit_uvs)
            fruit_point = proj.reproject_depth_pixel_to_3d(
                fruit_uvs, depth_img, depth_K
            )
            fruit_points.append(fruit_point)
        return fruit_points

    def get_fruits_3d_info(self, fruit_point):
        fruit_width = np.max(fruit_point[:, 0]) - np.min(fruit_point[:, 0])
        fruit_height = np.max(fruit_point[:, 1]) - np.min(fruit_point[:, 1])
        fruit_depth = np.median(fruit_point[:, 2])
        fruit_center = [
            np.min(fruit_point[:, 0]) + fruit_width / 2,
            np.min(fruit_point[:, 1]) + fruit_height / 2,
            fruit_depth,
        ]
        return fruit_width, fruit_height, fruit_depth, fruit_center
    
    def get_fruit_picking_point(self, keypoints_np, depth_K, mean_depth): 
        """
        input:
            keypoints_np: numpy array of keypoints [x1, y1, conf]
            depth_K: depth_K: depth camera intrinsic
            mean_depth: mean depth of fruits obtained from get_fruits_3d_info()
        return:
            final_point: the 3D coordinates of picking points
        """
        error_return = "error"
        # print("keypoints_np:",keypoints_np)
        pixel_picking_points = [[0] * 2 for _ in range(len(keypoints_np))]
        for i in range(len(keypoints_np)):
            points = keypoints_np[i]
            pixel_picking_points[i][0] = int(points[0,1])
            pixel_picking_points[i][1] = int(points[0,0])
        picking_uvs = np.array(pixel_picking_points)
        # print("picking_uvs:",picking_uvs)
        # final_point = proj.reproject_depth_pixel_to_3d(picking_uvs, depth_img, depth_K)
        if len(picking_uvs) != 0:
            if keypoints_np[0][0][0] > 1.0:
                final_point = proj.reproject_pixel_to_3d_using_z(picking_uvs, depth_K, mean_depth)
                # print("picking_points:",final_point)
                return final_point
            else:
                rospy.logwarn(rospy.get_name() + " low confidence ")
                print("low confidencel")
                return error_return
        else:
            rospy.logwarn(rospy.get_name() + " not detected ")
            print("not detected")
            return error_return

    def get_fruit_radius(self, keypoints_np, depth_K, mean_depth):
        error_return = "radius_error"
        if len(mean_depth) == 0:
            rospy.logwarn(rospy.get_name() + " no depth image ")
            print("no depth image")
            return error_return
        # print("keypoints_np:",keypoints_np)
        pixel_left_points = [[0] * 2 for _ in range(len(keypoints_np))]
        pixel_right_points = [[0] * 2 for _ in range(len(keypoints_np))]
        for i in range(len(keypoints_np)):
            points = keypoints_np[i]
            pixel_left_points[i][0] = int(points[3,1])
            pixel_left_points[i][1] = int(points[3,0])
            pixel_right_points[i][0] = int(points[4,1])
            pixel_right_points[i][1] = int(points[4,0])
        # print("pixel_left_points:",pixel_left_points)
        # print("pixel_right_points:",pixel_right_points)
        left_uvs = np.array(pixel_left_points)
        right_uvs = np.array(pixel_right_points)
        # print("picking_uvs:",picking_uvs)
        if len(left_uvs) != 0:
            if keypoints_np[0][3][0] > 1.0:
                left_points = proj.reproject_pixel_to_3d_using_z(left_uvs, depth_K, mean_depth)
                print("left_points:",left_points)
            else:
                rospy.logwarn(rospy.get_name() + " low confidence ")
                print("low confidencel")
                return error_return
        else:
            rospy.logwarn(rospy.get_name() + " not detected ")
            print("not detected")
            return error_return
        
        if len(right_uvs) != 0:
            if keypoints_np[0][4][0] > 1.0:
                right_points = proj.reproject_pixel_to_3d_using_z(right_uvs, depth_K, mean_depth)
                print("right_points:",right_points)
            else:
                rospy.logwarn(rospy.get_name() + " low confidence ")
                print("low confidencel")
                return error_return
        else:
            rospy.logwarn(rospy.get_name() + " not detected ")
            print("not detected")
            return error_return
        
        radius = math.sqrt((right_points[0][0] - left_points[0][0])**2 + (right_points[0][1] - left_points[0][1])**2) * 0.5
        # print("radius:",radius)
        return radius

    def filter_bboxes(self, boxes_np, fruit_conf, fruit_id):
        """
        input:
            boxes_np: numpy array of bbox [x1, y1, x2, y2, conf, cls]
            fruit_conf: confidence threshold
            fruit_id: class id of fruit
        return:
            boxes_np: numpy array of bbox [x1, y1, x2, y2, conf, cls]
        """
        boxes_np = boxes_np[boxes_np[:, 5] == fruit_id]
        boxes_np = boxes_np[boxes_np[:, 4] > fruit_conf]
        return boxes_np

    def publish_fruits(self, boxes_np, fruits_3d_info):
        fruits_msg = Fruits()
        fruits_msg.header.stamp = rospy.Time.now()
        fruits_msg.header.frame_id = "camera_color_optical_frame"
        for fruit_3d_info in fruits_3d_info:
            fruit_center = fruit_3d_info[3]
            fruit = Point()
            fruit.x = fruit_center[0]
            fruit.y = fruit_center[1]
            fruit.z = fruit_center[2]
            fruits_msg.fruits.append(fruit)
        self.fruits_publisher.publish(fruits_msg)

    def new_publish_fruits(self, picking_points):
        fruits_msg = Fruits()
        fruits_msg.header.stamp = rospy.Time.now()
        fruits_msg.header.frame_id = "camera_color_optical_frame"
        # print("picking_points:",picking_points)
        if picking_points != "error":
            for i in range(picking_points.shape[0]):
                fruit = Point()
                fruit.x = picking_points[i,0]
                fruit.y = picking_points[i,1]
                fruit.z = picking_points[i,2]
                fruits_msg.fruits.append(fruit)
            # print(fruits_msg)
            self.publish_fruits_marker(fruits_msg)
            self.fruits_publisher.publish(fruits_msg)


    def initialize_rviz_fruits_marker(self):
        # initialize the path points to visualize
        self.fruits_marker = Marker()
        self.fruits_marker.header.frame_id = self.frame_id
        self.fruits_marker.header.stamp = rospy.Time.now()

        self.fruits_marker.ns = "fruits_marker"
        self.fruits_marker.type = Marker.POINTS
        self.fruits_marker.action = Marker.ADD
        self.fruits_marker.pose.orientation.w = 1.0
        self.fruits_marker.scale.x = 0.003
        self.fruits_marker.scale.y = 0.003
        self.fruits_marker.scale.z = 0.003
        self.fruits_marker.color.b = 0.0
        self.fruits_marker.color.r = 1.0
        self.fruits_marker.color.g = 0.0
        self.fruits_marker.color.a = 1.0

    def publish_fruits_marker(self, fruits_msg):
        self.fruits_marker.header.frame_id = self.frame_id
        self.fruits_marker.points = []
        for i in range(len(fruits_msg.fruits)):
            p = Point()
            p.x = fruits_msg.fruits[i].x
            p.y = fruits_msg.fruits[i].y
            p.z = fruits_msg.fruits[i].z
            self.fruits_marker.points.append(p)
        self.fruits_marker.header.stamp = fruits_msg.header.stamp
        self.fruits_rviz_publisher.publish(self.fruits_marker)
    
    def strawberry_detection_pipeline(self):
        if self.rgb_ready:
            # Use the copy img to avoid the RGB bug caused by different threads
            tic = time.time()
            front_rgb_image = self.front_rgb_image
            depth_get = False
            if self.depth_ready:
                depth_image = self.depth_image.copy() # 使用复制的深度图像，理由同上
                depth_get = True
                self.depth_ready = False # 避免重复进循环

            if self.viz_flag:
                visualize_image = front_rgb_image.copy()
            self.rgb_ready = False

            # YOLO
            # boxes_np = self.preprocess_and_inference(
            #     front_rgb_image, profile=self.profile
            # )
            # boxes_np = self.filter_bboxes(boxes_np, self.fruit_conf, 0) # 过滤掉不属于待识别类别以及置信度不够高的框
            # keypoints_np = self.get_keypoints(self.model, front_rgb_image)
            boxes_np, keypoints_np = self.get_bboxes_and_keypoints(self.model, front_rgb_image)
            fruits_3d_info = []  # fruit_width, fruit_height, fruit_depth, fruit_center
            fruit_Zs = []
            
            # fruit_centers = []
            # 3D info of fruit
            if depth_get:
                depth_get = False
                fruit_patches = self.get_fruit_pathches(
                    depth_image, boxes_np, valid_z_min=25
                )
                fruit_points = self.get_fruit_points(
                    fruit_patches, depth_image, self.depth_image_cam_info
                )
                for fruit_point in fruit_points:
                    (
                        fruit_width,
                        fruit_height,
                        fruit_depth,
                        fruit_center,
                    ) = self.get_fruits_3d_info(fruit_point)
                    print(
                        "w: %.2fm, h: %.2fm, d: %.2fm, center(xyz): %.2fm, %.2fm, %.2fm"
                        % (
                            fruit_width,
                            fruit_height,
                            fruit_depth,
                            fruit_center[0],
                            fruit_center[1],
                            fruit_center[2],
                        )
                    )
                    fruits_3d_info.append(
                        [fruit_width, fruit_height, fruit_depth, fruit_center]
                    )
                    fruit_Zs.append(fruit_depth)
                    # fruit_centers.append(fruit_center)
            
            if len(keypoints_np) > 0: 
                picking_points = self.get_fruit_picking_point(keypoints_np, self.depth_image_cam_info, fruit_Zs) # get 3D picking points
                # self.publish_fruits(boxes_np, fruits_3d_info)
                if picking_points != "error":
                    radius = self.get_fruit_radius(keypoints_np, self.depth_image_cam_info, fruit_Zs)
                    if radius != "radius_error":
                        picking_points[0][2] = picking_points[0][2] + radius
                        print("picking_points:", picking_points)
                        self.new_publish_fruits(picking_points)
                
                # print("fruit_centers:", np.array(fruit_centers))
                # self.new_publish_fruits(np.array(fruit_centers))

            

            if self.viz_flag is True:
                visualize_image = self.draw_strawberry_detection_image(
                    visualize_image,
                    boxes_np,
                    keypoints_np,
                    fruits_3d_info,
                    class_names=self.classnames_file,
                )
                self.encode_and_publish_image(
                    visualize_image, self.visualize_image_publisher, self.profile
                )
            if self.profile:
                rospy.logdebug(
                    rospy.get_name()
                    + " total time: %.1f ms" % ((time.time() - tic) * 1000)
                )

    def safe_sleep(self):
        try:
            self.rate.sleep()
        except:
            rospy.logwarn(rospy.get_name() + " sleep failed, use system sleep")
            time.sleep(self.rate.sleep_dur.to_sec())

    def run(self):
        while not rospy.is_shutdown():
            self.strawberry_detection_pipeline()
            self.safe_sleep()


if __name__ == "__main__":
    strawberry_detection = StrawberryDetection(rate=10)
    rospy.spin()
