# *coding:utf-8 *
# 用于验证鱼眼展开后，图片和标注坐标点是否匹配
# 在转换后的图上画出标注点
import os
import cv2
import mmcv
import numpy as np
from utils.unfold import trans_corrdinate_without_inner
# trans_corrdinate_without_inner(pixel_points, w, h, K, D, theta, fi, FOV, R, status)
# 鱼眼相机内参矩阵
# K = np.array([[581.1058307906718, 0.0, 955.5987388116735], [0.0, 579.8976865646564, 974.0212406615763], [0.0, 0.0, 1.0]])
# 鱼眼相机畸变系数
# D = np.array([-0.015964497003735242, -0.002789473611910958, 0.005727838947159351, -0.0025185770227346576])
point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 可以为 0、4、8

# 鱼眼展开后图片的文件夹路径
# img_dir = 'D:/data/new_log_side_img/0609/train/'
# 鱼眼展开后的标注路径，via格式
# ann_dir = 'D:/data/new_log_side_img/0609/annotations/train.json'
# 生成图片的文件夹路径
# save_dir = 'D:/data/new_log_side_img/0609/train_points/'


def draw(ann_dir, img_dir, save_dir):
    data_infos = mmcv.load(ann_dir)
    print("start draw points.")
    for idx, item in enumerate(mmcv.track_iter_progress(data_infos.items())):
        # print(item[1]['filename'])
        img = cv2.imread(os.path.join(img_dir, item[1]['filename']))
        # print(img)
        for pig in item[1]['regions']:
            for point in zip(pig['shape_attributes']['all_points_x'], pig['shape_attributes']['all_points_y']):
                # point = trans_corrdinate_without_inner(point, 1080, 1080, K, D, 50, f, 90, 100, 1)
                cv2.circle(img, (point[0], point[1]), point_size, point_color, thickness)
        cv2.imwrite(os.path.join(save_dir, item[1]['filename']), img)
    print("draw points finished.")
