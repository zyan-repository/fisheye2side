# *coding:utf-8 *
import os
import cv2
import tqdm
import numpy as np
from utils.unfold import polar_unfold

fisheye_configs = {
    'plz': [(2592, 1944),  # dim of input image (w,h )
            np.array([[894.1664550880178, 0.0, 1272.584098136942], [0.0, 893.847453815009, 961.3637141182406],
                      [0.0, 0.0, 1.0]]),  # K
            np.array(
                [[-0.01880269911771689], [-0.001463985599093003], [0.0006538835249622903], [-0.00048675949913783216]]),
            # D
            0.45  # fov_scale
            ],
    'hk1920': [(1920, 1920),
               np.array([[581.1058307906718, 0.0, 955.5987388116735], [0.0, 579.8976865646564, 974.0212406615763],
                         [0.0, 0.0, 1.0]]),
               np.array([-0.015964497003735242, -0.002789473611910958, 0.005727838947159351,
                         -0.0025185770227346576]),  # D
               0.45  # fov_scale
               ],

    'hk1920_2': [(1920, 1920),
                 np.array([[597.1506937919324, 0.0, 941.8933562415162], [0.0, 596.6519047766966, 968.1380137889757],
                           [0.0, 0.0, 1.0]]),
                 np.array([[-0.012013029051805495], [-0.0057539405259074], [0.0025240923677038603],
                           [-0.0009949409261845303]]),  # D
                 1.8  # fov_scale
                 ]
}


def get_useful_area(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_fisheye = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    center, radius = cv2.minEnclosingCircle(contour_fisheye)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask = cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), (1, 1, 1), -1)
    image_useful = image * mask
    image_fisheye = image_useful[int(center[1]) - int(radius):int(center[1]) + int(radius),
                    int(center[0]) - int(radius):int(center[0]) + int(radius), :]
    return image_fisheye


def train(image):
    R = image.shape[0] // 2
    W = int(2 * np.pi * R)
    H = R
    mapx = np.zeros([H, W], dtype=np.float32)
    mapy = np.zeros([H, W], dtype=np.float32)
    for i in range(mapx.shape[0]):
        for j in range(mapx.shape[1]):
            angle = j / W * np.pi * 2
            radius = H - i
            mapx[i, j] = R + np.sin(angle) * radius
            mapy[i, j] = R - np.cos(angle) * radius
    np.save('mapx.npy', mapx)
    np.save('mapy.npy', mapy)


# img = cv2.imread('D:/20220307174420/train/SCYS_1640484016.4704242.jpg')
# img_fisheye = img[90:1830, 90:1830]  # get_useful_area(img)

# train(img_fisheye)
# mapx = np.load('mapx.npy')
# mapy = np.load('mapy.npy')

# cv2.imshow("fisheye_img", img_fisheye)
# cv2.waitKey(0)

# unfold_img = cv2.remap(img_fisheye, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
# unfold_img = polar_unfold(img_fisheye)

from utils.calibrate import cal_outside_punctuation_fisheye, get_obj_points, cal_internal_fisheye
# obj_p = get_obj_points((6, 9))
# img_dir = 'D:/camera_20211221'
# img_list = []
# for img in os.listdir(img_dir):
#     img_list.append(cv2.imread(img_dir + '/' + img))
# ret, mtx, dist = cal_internal_fisheye(obj_p, img_list, (6, 9))
# obj_p = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]]).astype(np.float)
# img_point = np.array([[[813, 192], [972, 176], [1146, 169], [1315, 167]]]).astype(np.float)
# # print(mtx)
# # print(dist.squeeze())
# ret, rvec, rmat, tvec = cal_outside_punctuation_fisheye(obj_p, img_point, mtx, dist.squeeze())
# # print(ret, rvec, rmat, tvec)
# from utils.position import position_fisheye
# ret, lst = position_fisheye(np.array([5., 2.]), mtx, np.array([dist.squeeze()]), rmat, tvec)
# print(np.array([5., 2.]), mtx, np.array([dist.squeeze()]), rmat, tvec)
# print(ret)
# print(lst)


from utils.unfold import project_unfold_fast, trans_corrdinate_without_inner

# cfgs = fisheye_configs.get("hk1920", None)
# (w, h), K, D, scale = cfgs
# unfold_img = project_unfold_fast(1740, 1740, K, D, 90, 270, 90, img_fisheye, 1)
# cv2.imshow("unfold_img", unfold_img)
# cv2.imwrite("unfold.jpg", unfold_img)
# cv2.waitKey(0)

# def project_unfold_fast(w, h, K, D, theta, fi, FOV, yxImg, R):
#     """
#     基于球面坐标系和鱼眼相机投影原理对鱼眼相机图片进行展开
#     可以根据俯仰角和水平角选定展开位置
#     根据视场角选定展开范围，将范围内的图像展开到给定大小的图像上
#     :param w: 展开图像的宽
#     :param h: 展开图像的高
#     :param K: 内参矩阵
#     :param D: 畸变系数，代表着鱼眼相机投影模型（如果是[4, ]的array，代表为多项式模型；如果是float，代表是等距投影模型，投影系数为k）
#     :param theta: 入射角，即投影平面法线与xoy平面的夹角（0~90°）
#     :param fi: 水平角，即投影平面法线投影xoy平面后，和x轴正半轴的夹角（0~360°）
#     :param FOV: 视场角，即在投影方向上展开范围的角度（0~180°）
#     :param yxImg: 原始图像
#     :param R: 球面半径，任意值
#     :return: 返回展开后的图像
#     """
#
#     'hk1920': [(1920, 1920),
#                np.array([[581.1058307906718, 0.0, 955.5987388116735], [0.0, 579.8976865646564, 974.0212406615763],
#                          [0.0, 0.0, 1.0]]),
#                np.array([[-0.015964497003735242], [-0.002789473611910958], [0.005727838947159351],
#                          [-0.0025185770227346576]]),  # D
#                0.45  # fov_scale
#                ],
#
#     'hk1920_2': [(1920, 1920),
#                  np.array([[597.1506937919324, 0.0, 941.8933562415162], [0.0, 596.6519047766966, 968.1380137889757],
#                            [0.0, 0.0, 1.0]]),
#                  np.array([[-0.012013029051805495], [-0.0057539405259074], [0.0025240923677038603],
#                            [-0.0009949409261845303]]),  # D
#                  1.8  # fov_scale
#                  ]

K = np.array([[581.1058307906718, 0.0, 955.5987388116735],
              [0.0, 579.8976865646564, 974.0212406615763],
              [0.0, 0.0, 1.0]])
D = np.array([-0.015964497003735242, -0.002789473611910958,
              0.005727838947159351, -0.0025185770227346576])
img = cv2.imread('D:/data/20220606092038/train/BYZ_SH_NK_1641266107256.jpg')

# 把图片分割成多个区域，分区域展开
# if img is None:
#     print(img)
# else:
#     print("a")
# img = img[:1740, :1740]
# print(img.shape)
# cv2.imshow('image', img)
# cv2.waitKey(100)
img = project_unfold_fast(640, 640, K, D, 50, 270, 90, img, 100)
import time
s = time.time()
trans_corrdinate_without_inner(np.array([1000, 1000]), 640, 640, K, D, 50, 0, 90, 100)
t = time.time()
print(t - s)

# print(trans_corrdinate_without_inner(np.array([1000, 1000]), 640, 640, K, D, 50, 0, 90, 100))
cv2.imwrite('555.jpg', img)
