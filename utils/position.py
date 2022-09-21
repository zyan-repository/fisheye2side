"""
相机定位工具包
"""
import cv2

import numpy as np
from numpy.linalg import solve

from scipy.optimize import newton


def equation_set_solve(point, R, T):
    """
    通过解多元一次方程组得形式得到Zc与Xw，Yw，也就是深度与世界坐标的XY，世界坐标Z默认为0
    :param point: 相机坐标系下前两维的坐标Xc,Yc，shape:[2, ]
    :param R: 外参中的旋转矩阵，如果是旋转向量需要用函数转换之后输入，shape:[3, 3]
    :param T: 外参中的位移向量，shape:[3, ]
    :return: 返回世界坐标系下XY的坐标，shape:[2, ] （此处还可以得到深度，也就是相机坐标系下的Z，暂时不需要）
    """
    # 根据定位的求解式，构建三元一次方程组的系数矩阵
    w = \
        np.array(
            [[R[0, 0], R[0, 1], -point[0]],
             [R[1, 0], R[1, 1], -point[1]],
             [R[2, 0], R[2, 1], -1]], dtype='float')
    # 对外参中的位移向量进行类型转换，float
    T = np.array(T, dtype='float')
    # 求解定位的计算式，得到一个三维向量，其中前两维是定位点的世界坐标的X与Y
    ans = solve(w, -T).T
    # 返回定位结果，世界坐标的X与Y
    return ans[0, :2]


# 牛顿迭代解一元多次方程
def inverse(f, f_prime=None):
    """
    用牛顿迭代解一元多次方程，用于反向求解畸变矫正模型，得到畸变前的角度
    :param f: 方程形式
    :param f_prime: 不晓得是干啥滴
    :return: 返回方程的解x
    """
    
    # 构建求解函数
    def solve_(y):
        # 使用牛顿迭代法进行求解
        return newton(lambda x: f(x) - y, 1., f_prime, (), 1e-10, 1000000)

    # 返回方程的解
    return solve_


# 单目相机的目标定位
def position_monocular(pixel_points, mtx, dist, rmx, tvec):
    """
    单目相机定位算法
    :param pixel_points: 待定位点的像素坐标，shape:[N, 2]
    :param mtx: 单目相机的内参矩阵，shape:[3, 3]
    :param dist: 单目相机的畸变系数，shape:[5, ]
    :param rmx: 单目相机的旋转矩阵，shape:[3, 3]
    :param tvec: 单目相机的位移矩阵，shape:[5, ]
    :return: 以列表的形式返回每一个点的世界坐标
    """
    try:
        # 使用反投影函数得到目标像素点在相机坐标系下的坐标点
        un_distort_points = cv2.undistortPoints(pixel_points.reshape((-1, 2)), mtx, dist, None, None)[:, 0, :]
        # 定义空列表，用于储存待求解坐标
        lis = []
        # 对每一个像素坐标系的像素坐标进行定位求解
        for i in range(un_distort_points.shape[0]):
            # 基于目标点在相机坐标系上的点构建三元一次方程，求解得到对应世界坐标系中的坐标
            tmp_ans = equation_set_solve(un_distort_points[i], rmx, tvec)  # 求解坐标
            # 将世界坐标系中的坐标存入列表
            lis.append(tmp_ans)
        # 返回结果，每一个待定位点在世界坐标中的坐标
        return True, lis

    except cv2.error:
        # 捕获cv异常
        return False, None

    except Exception:
        # 捕获python异常
        return False, None


# 鱼眼相机的目标定位
def position_fisheye(pixel_points, mtx, dist, rmx, tvec):
    """
    鱼眼相机定位算法
    :param pixel_points: 待定位点的像素坐标，shape:[N, 2]
    :param mtx: 鱼眼相机的内参矩阵，shape:[3, 3]
    :param dist: 鱼眼相机的畸变系数，shape:[1, 5]
    :param rmx: 鱼眼相机的旋转矩阵，shape:[3, 3]
    :param tvec: 鱼眼相机的位移矩阵，shape:[5, ]
    :return: 以列表的形式返回每一个点的世界坐标
    """
    try:
        # 去除变量中多余维度
        pixel_points = pixel_points.reshape((-1, 2))
        # 定义空列表，用于储存待求解坐标
        lis = []
        # 对每一个像素坐标系的像素坐标进行定位求解
        for i in range(pixel_points.shape[0]):
            # 使用内参矩阵将像素坐标系下的点，转换到畸变后的相机坐标系下，Z，深度都为1
            x_d, y_d = (pixel_points[i, 0] - mtx[0, 2]) / mtx[0, 0], (pixel_points[i, 1] - mtx[1, 2]) / mtx[1, 1]
            # 基于等距投影模型计算得到目标点在畸变后的角度
            theta_d = np.sqrt(x_d ** 2 + y_d ** 2)
            # 去除畸变系数array中的多余维度
            dist = dist[0]
            # 构建牛顿迭代法来逆求解畸变模型的函数
            poly_solve = inverse(
                lambda x: x + dist[0] * x ** 3 + dist[1] * x ** 5 + dist[2] * x ** 7 + dist[3] * x ** 9)
            # 使用畸变函数，将畸变后的角度求解为畸变前的角度
            theta = poly_solve(theta_d)
            # 计算目标点在相机坐标系下，映射到深度为1位置的圆面半径
            r = np.tan(theta)
            # 根据比例关系计算出畸变前相机坐标系在深度为1的位置的相机坐标x与y
            x_c, y_c = r * x_d / theta_d, r * y_d / theta_d
            # 基于目标点在相机坐标系上的点构建三元一次方程，求解得到对应世界坐标系中的坐标
            point = equation_set_solve([x_c, y_c], rmx, tvec)
            # 将世界坐标系中的坐标存入列表
            lis.append(point)
        # 返回结果，每一个待定位点在世界坐标中的坐标
        return True, lis

    except cv2.error:
        # 捕获cv异常
        return False, None

    except Exception:
        # 捕获python异常
        return False, None
