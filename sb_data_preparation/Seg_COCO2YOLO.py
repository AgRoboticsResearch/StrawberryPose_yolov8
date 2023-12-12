import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
#这里根据自己的json文件位置，换成自己的就行
parser.add_argument('--json_path', default='/mnt/TempData/brl/pangolin/ruichen/strawberry-pp-w-r-dataset/annotations/SDI_annotations/val/json',type=str, help="input: coco format(json)")
#这里设置.txt文件保存位置
parser.add_argument('--save_path', default='/mnt/TempData/brl/pangolin/ruichen/strawberry-pp-w-r-dataset/annotations/SDI_annotations/labels/val', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()

# def convert_box(size, box):
#     dw = 1. / (size[0])
#     dh = 1. / (size[1])
#     x = (box[0] + box[2]) / 2.0
#     y = (box[1] + box[3]) / 2.0
#     w = box[2] - box[0]
#     h = box[3] - box[1]
# #round函数确定(xmin, ymin, xmax, ymax)的小数位数
#     x = format(x * dw ,'.5f')
#     w = format(w * dw ,'.5f')
#     y = format(y * dh ,'.5f')
#     h = format(h * dh ,'.5f')
#     return (x, y, w, h)

def convert_point(width, height, segpoints):
    list=[]
    dw = 1. / width
    dh = 1. / height
    segpoint_num = len(segpoints[0])
    for i in range(segpoint_num):
    # 每个关键点的横坐标数据
        if i % 2 == 0:
            list.append(format(segpoints[0][i]*dw,'.5f'))
    # 每个关键点的纵坐标数据
        if i % 2 == 1:
            list.append(format(segpoints[0][i]*dh,'.5f'))
    result = tuple(list)
    return result


if __name__ == '__main__':
    files = os.listdir(arg.json_path)
    for file in files:
        if file.endswith(".json"): #如果file以json结尾
            json_file =  file # COCO Object Instance 类型的标注
            ana_txt_save_path = arg.save_path  # 保存的路径
            data = json.load(open(arg.json_path + "/" + json_file, 'r'))
            if not os.path.exists(ana_txt_save_path):
                os.makedirs(ana_txt_save_path)

            # list_file = open(os.path.join(ana_txt_save_path, 'path.txt'), 'w')

            # filename = data["file_name"]
            width = data['width']   # 图像宽度
            height = data['height'] # 图像高度
            newname = file
            newname = newname.split(".")
            ana_txt_name = newname[0] + ".txt"  # 对应的txt名字，与jpg一致
            
            f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
            for ann in data["annotations"]:
                point = convert_point(width, height, ann["segmentation"])
                f_txt.write('%s' % (ann["category_id"]))
                for coordinate in point:
                    f_txt.write(' %s' % (str(coordinate)))
                f_txt.write('\n')
            f_txt.close()
            #将图片的相对路径写入train2017或val2017的路径
            # list_file.write('./images/train/%s.png\n' %(head))
            # print("convert successful!")
            print('{} --> {} 转换完成'.format(json_file, ana_txt_name))  
    # list_file.close()
