"""
鱼眼展开工具包
"""
import math

import cv2
import numpy as np
from mpmath import inverse
from numpy.linalg import solve
from scipy.optimize import newton


def polar_unfold(img):
    """
    使用极坐标转换将鱼眼相机图片转换为全景图片
    优点是运算速度快，方法简便
    缺点是只能进行固定的展开，尤其是没法设定俯仰角
    :param img: 待展开的图像
    :return: 展开后的全景图片
    """
    x0 = img.shape[0] // 2  # 得到圆形区域的中心坐标
    y0 = img.shape[1] // 2
    radius = img.shape[0] // 2
    unwrapped_height = radius
    unwrapped_width = int(2 * math.pi * radius)
    unwrapped_img = np.zeros((unwrapped_height, unwrapped_width, 3),
                             dtype="u1")  # 初始化展开后的图片
    except_count = 0
    for j in range(unwrapped_width):
        theta = 2 * math.pi * (j / unwrapped_width)
        for i in range(unwrapped_height):
            unwrapped_radius = radius - i
            x = unwrapped_radius * math.cos(theta) + x0
            y = unwrapped_radius * math.sin(theta) + y0
            x, y = int(x), int(y)
            try:
                unwrapped_img[i, j, :] = img[x, y, :]
            except Exception:
                except_count = except_count + 1
    return unwrapped_img


def project_point_withoutK(obj_points, rvec, tvec, K):
    """
    重投影函数。通过内参、外参
    将世界坐标系中的坐标点映射到鱼眼图像坐标系中
    不使用畸变系数来矫正画面
    :param obj_points: 待重投影的世界坐标系上的点坐标
    :param rvec: 旋转向量，在本模块中，重投影不需要外参，故可以省略
    :param tvec: 平移向量，在本模块中，重投影不需要外参，故可以省略
    :param K: 内参矩阵
    :return: 返回重投影得到的像素坐标
    """
    # 计算相机坐标系中的坐标（深度为1时）
    x = obj_points[0, 0, 0] / abs(obj_points[0, 0, 2])
    y = obj_points[0, 0, 1] / abs(obj_points[0, 0, 2])
    # 使用arctan计算出入射角度
    theta = np.arctan2(
        np.sqrt(obj_points[0, 0, 0] ** 2 + obj_points[0, 0, 1] ** 2),
        obj_points[0, 0, 2])
    # 使用内参矩阵计算出重投影得到的像素坐标
    u = K[0, 0] * theta * x / np.sqrt(x ** 2 + y ** 2) + K[0, 2]
    v = K[1, 1] * theta * y / np.sqrt(x ** 2 + y ** 2) + K[1, 2]
    # 返回重投影得到的像素坐标系坐标
    return [u, v]


def project_point_withK(obj_points, rvec, tvec, K, D):
    """
    重投影函数。通过内参、外参、畸变系数
    将世界坐标系中的坐标点映射到鱼眼图像坐标系中
    使用畸变系数来矫正画面
    :param obj_points: 待重投影的世界坐标系上的点坐标
    :param rvec: 旋转向量，在本模块中，重投影不需要外参，故可以省略
    :param tvec: 平移向量，在本模块中，重投影不需要外参，故可以省略
    :param K: 内参矩阵
    :param D: 畸变系数
    :return: 返回重投影得到的像素坐标
    """
    # 计算相机坐标系中的坐标（深度为1时）
    x = obj_points[0, 0, 0] / abs(obj_points[0, 0, 2])
    y = obj_points[0, 0, 1] / abs(obj_points[0, 0, 2])
    # 使用arctan计算出入射角度
    theta = np.arctan2(
        np.sqrt(obj_points[0, 0, 0] ** 2 + obj_points[0, 0, 1] ** 2),
        obj_points[0, 0, 2])
    # 基于KB畸变模型，将入射角转化为畸变后的入射角
    theta_d = theta + D[0] * theta ** 3 + D[1] * theta ** 5 + D[
        2] * theta ** 7 + D[3] * theta ** 9
    # 使用内参矩阵计算出重投影得到的像素坐标
    u = K[0, 0] * theta_d * x / np.sqrt(x ** 2 + y ** 2) + K[0, 2]
    v = K[1, 1] * theta_d * y / np.sqrt(x ** 2 + y ** 2) + K[1, 2]
    # 返回重投影得到的像素坐标系坐标
    return [u, v]


def project_unfold_fast(w, h, K, D, theta, fi, FOV, yxImg, R, status_D):
    """
    基于球面坐标系和鱼眼相机投影原理对鱼眼相机图片进行展开
    可以根据俯仰角和水平角选定展开位置
    根据视场角选定展开范围，将范围内的图像展开到给定大小的图像上
    :param w: 展开图像的宽
    :param h: 展开图像的高
    :param K: 内参矩阵
    :param D: 畸变系数，代表着鱼眼相机投影模型（如果是[4, ]的array，代表为多项式模型；如果是float，代表是等距投影模型，投影系数为k）
    :param theta: 入射角，即投影平面法线与xoy平面的夹角（0~90°）
    :param fi: 水平角，即投影平面法线投影xoy平面后，和x轴正半轴的夹角（0~360°）
    :param FOV: 视场角，即在投影方向上展开范围的角度（0~180°）
    :param yxImg: 原始图像
    :param R: 球面半径，任意值
    :return: 返回展开后的图像
    """
    # 将所有输入的角度转换为弧度制
    theta = theta / 180 * math.pi
    fi = fi / 180 * math.pi
    FOV = FOV / 180 * math.pi
    # 计算展开图像与投影半球面的切点，也是展开后图像的中心点
    x0, y0, z0 = R * math.cos(theta) * math.cos(fi), R * math.cos(
        theta) * math.sin(fi), R * math.sin(
        theta)  # 直角坐标系--->球面坐标系
    # 计算每个像素代表的球面坐标系的距离
    d_pixel = (2 * math.tan(FOV / 2) * R) / w
    # 生成展开后图像的像素坐标
    u_axis, v_axis = np.mgrid[-h / 2:h / 2:1, -w / 2:w / 2:1]
    # 合并坐标
    u_axis = np.expand_dims(u_axis, 2)
    v_axis = np.expand_dims(v_axis, 2)
    u_v = np.concatenate((u_axis, v_axis), axis=2)
    # 维度增加
    u_v = np.expand_dims(u_v, 3)
    # 构建运算矩阵，用来计算球面坐标系偏移量
    tmp_mat = np.array([[[math.sin(fi), math.sin(theta) * math.cos(fi)],
                         [- math.cos(fi), math.sin(theta) * math.sin(fi)],
                         [0, - math.cos(theta)]]])
    # 计算出展开后图像上每个像素的球面坐标
    w_points = np.matmul(tmp_mat, u_v) * d_pixel + np.array(
        [[[[x0], [y0], [z0]]]])
    w_points = w_points.squeeze()
    # 将坐标处理为相机坐标系坐标，深度为1
    w_points /= w_points[:, :, 2:3]
    # 计算半径r
    r = np.sqrt(w_points[:, :, 0] ** 2 + w_points[:, :, 1] ** 2)
    # 计算畸变前的俯仰角
    theta = np.arctan2(r, w_points[:, :, 2])
    # 计算畸变后的俯仰角
    # D = D[0]
    if status_D:
        theta_d = theta + D[0] * theta ** 3 + D[1] * theta ** 5 + D[
            2] * theta ** 7 + D[3] * theta ** 9
    else:
        theta_d = theta
    # 基于内参矩阵计算像素坐标
    u = K[0, 0] * theta_d * w_points[:, :, 0] / r + K[0, 2]
    v = K[1, 1] * theta_d * w_points[:, :, 1] / r + K[1, 2]
    # 像素坐标整数化
    u = u.astype(np.int32).T[::-1, :]
    v = v.astype(np.int32).T[::-1, :]
    # 生成展开后图像
    mapImg = yxImg[v, u]
    # 返回展开后图像
    return mapImg


def extract_fisheye_valid_region(img):
    # 判断img，若为str，则为图片路径
    if isinstance(img, str):
        # 读取图片，转换为BGR图像
        img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_COLOR)
    # 若为np数组，则已为BGR图像，跳过
    elif isinstance(img, np.ndarray):
        pass
    T = 40

    # 转换为灰度图片
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 提取原图大小
    rows, cols = img.shape[:2]

    # 从上向下扫描
    for i in range(0, rows, 1):
        for j in range(0, cols, 1):
            if img_gray[i, j] >= T:
                if img_gray[i + 1, j] >= T:
                    top = i
                    break
        else:
            continue
        break

    # 从下向上扫描
    for i in range(rows - 1, -1, -1):
        for j in range(0, cols, 1):
            if img_gray[i, j] >= T:
                if img_gray[i - 1, j] >= T:
                    bottom = i
                    break
        else:
            continue
        break

    # 从左向右扫描
    for j in range(0, cols, 1):
        for i in range(top, bottom, 1):
            if img_gray[i, j] >= T:
                if img_gray[i, j + 1] >= T:
                    left = j
                    break
        else:
            continue
        break

    # 从右向左扫描
    for j in range(cols - 1, -1, -1):
        for i in range(top, bottom, 1):
            if img_gray[i, j] >= T:
                if img_gray[i, j - 1] >= T:
                    right = j
                    break
        else:
            continue
        break
    # 计算有效区域半径
    R = max((bottom - top) / 2, (right - left) / 2)
    return R


def project_unfold_fast_without_inner(w, h, theta, fi, FOV, yxImg, R):
    """
    基于球面坐标系和鱼眼相机投影原理对鱼眼相机图片进行展开
    可以根据俯仰角和水平角选定展开位置
    根据视场角选定展开范围，将范围内的图像展开到给定大小的图像上
    :param w: 展开图像的宽
    :param h: 展开图像的高
    :param theta: 入射角，即投影平面法线与xoy平面的夹角（0~90°）
    :param fi: 水平角，即投影平面法线投影xoy平面后，和x轴正半轴的夹角（0~360°）
    :param FOV: 视场角，即在投影方向上展开范围的角度（0~180°）
    :param yxImg: 原始图像
    :param R: 球面半径，任意值
    :return: 返回展开后的图像
    """
    shape = yxImg.shape
    K = np.zeros((3, 3))
    D = np.zeros((1, 4))
    K[0, 0] = R / (math.pi / 2)
    K[0, 2] = shape[1] / 2
    K[1, 1] = R / (math.pi / 2)
    K[1, 2] = shape[0] / 2
    # 将所有输入的角度转换为弧度制
    theta = theta / 180 * math.pi
    fi = fi / 180 * math.pi
    FOV = FOV / 180 * math.pi
    # 计算展开图像与投影半球面的切点，也是展开后图像的中心点
    x0, y0, z0 = R * math.cos(theta) * math.cos(fi), R * math.cos(
        theta) * math.sin(fi), R * math.sin(
        theta)  # 直角坐标系--->球面坐标系
    # 计算每个像素代表的球面坐标系的距离
    d_pixel = (2 * math.tan(FOV / 2) * R) / w
    # 生成展开后图像的像素坐标
    u_axis, v_axis = np.mgrid[-h / 2:h / 2:1, -w / 2:w / 2:1]
    # 合并坐标
    u_axis = np.expand_dims(u_axis, 2)
    v_axis = np.expand_dims(v_axis, 2)
    u_v = np.concatenate((u_axis, v_axis), axis=2)
    # 维度增加
    u_v = np.expand_dims(u_v, 3)
    # 构建运算矩阵，用来计算球面坐标系偏移量
    tmp_mat = np.array([[[math.sin(fi), math.sin(theta) * math.cos(fi)],
                         [- math.cos(fi), math.sin(theta) * math.sin(fi)],
                         [0, - math.cos(theta)]]])
    # 计算出展开后图像上每个像素的球面坐标
    w_points = np.matmul(tmp_mat, u_v) * d_pixel + np.array(
        [[[[x0], [y0], [z0]]]])
    w_points = w_points.squeeze()
    # 将坐标处理为相机坐标系坐标，深度为1
    w_points /= w_points[:, :, 2:3]
    # 计算半径r
    r = np.sqrt(w_points[:, :, 0] ** 2 + w_points[:, :, 1] ** 2)
    # 计算畸变前的俯仰角
    theta = np.arctan2(r, w_points[:, :, 2])
    # 计算畸变后的俯仰角
    D = D[0]
    theta_d = theta + D[0] * theta ** 3 + D[1] * theta ** 5 + D[
        2] * theta ** 7 + D[3] * theta ** 9
    # 基于内参矩阵计算像素坐标
    u = K[0, 0] * theta_d * w_points[:, :, 0] / r + K[0, 2]
    v = K[1, 1] * theta_d * w_points[:, :, 1] / r + K[1, 2]
    # 像素坐标整数化
    u = u.astype(np.int32).T[::-1, :]
    v = v.astype(np.int32).T[::-1, :]
    # 生成展开后图像
    mapImg = yxImg[v, u]
    # 返回展开后图像
    return mapImg


def projection_unfold(mapImg, inWidth, inHeight, w, h, K, D, theta, fi, FOV,
                      yxImg, R):
    """
    基于球面坐标系和鱼眼相机投影原理对鱼眼相机图片进行展开
    可以根据俯仰角和水平角选定展开位置
    根据视场角选定展开范围，将范围内的图像展开到给定大小的图像上
    :param mapImg: 待展开的图像
    :param inWidth: 原鱼眼图像的高
    :param inHeight: 原鱼眼图像的宽
    :param w: 展开图像的宽
    :param h: 展开图像的高
    :param K: 内参矩阵
    :param D: 畸变系数，代表着鱼眼相机投影模型（如果是[4, ]的array，代表为多项式模型；如果是float，代表是等距投影模型，投影系数为k）
    :param theta: 入射角，即投影平面法线与xoy平面的夹角（0~90°）
    :param fi: 水平角，即投影平面法线投影xoy平面后，和x轴正半轴的夹角（0~360°）
    :param FOV: 视场角，即在投影方向上展开范围的角度（0~180°）
    :param yxImg: 原始图像
    :param R: 球面半径，任意值
    :return: 返回展开后的图像
    """
    # 定义列表，用于储存投影到鱼眼图像上的坐标。用于验证展开区域
    pixel = []
    # 计算鱼眼图像的中心点
    xx = int(inWidth / 2.0)
    yy = int(inHeight / 2.0)
    # 记录鱼眼图像中心点
    pixel.append([xx, yy])
    # 将所有输入的角度转换为弧度制
    theta = theta / 180 * math.pi
    fi = fi / 180 * math.pi
    FOV = FOV / 180 * math.pi
    # 计算展开图像与投影半球面的切点，也是展开后图像的中心点
    x0, y0, z0 = R * math.cos(theta) * math.cos(fi), R * math.cos(
        theta) * math.sin(fi), R * math.sin(
        theta)  # 直角坐标系--->球面坐标系
    # 计算每个像素代表的球面坐标系的距离
    d_pixel = (2 * math.tan(FOV / 2) * R) / w
    # 计算每一个展开后图像上的点投影对应的鱼眼图像上的点
    for m in range(int(-h / 2), int(h / 2)):
        for n in range(int(-w / 2), int(w / 2)):
            # 计算展开后图像上的点相对图像中心点的偏移量
            d_x = n * d_pixel * math.sin(theta) * math.cos(fi) + math.sin(
                fi) * m * d_pixel
            d_y = n * d_pixel * math.sin(theta) * math.sin(fi) - math.cos(
                fi) * m * d_pixel
            d_z = - n * d_pixel * math.cos(theta)
            # 计算展开后图像上的点在球面上的坐标
            x1, y1, z1 = x0 + d_x, y0 + d_y, z0 + d_z
            # 构建旋转平移向量
            rvec = np.array([0.0, 0.0, 0.0])
            tvec = np.array([0.0, 0.0, 0.0])
            tvec = tvec[None, :]
            # 构建世界坐标系中的坐标点
            obj_points = np.array([x1, y1, z1])
            obj_points = obj_points[None, None, :]
            imagePoints = project_point_withK(obj_points, rvec, tvec, K, D)
            # 得到鱼眼图像坐标系中的坐标uv
            u = int(imagePoints[0])
            v = int(imagePoints[1])
            # 计算展开后图像的中心点
            aa = int(h / 2)
            bb = int(w / 2)
            # 将重投影到的鱼眼图像的像素值赋给展开后的图像
            if (u >= 0) and (u < inWidth - 1) and (v >= 0) and (
                    v < inHeight - 1):
                mapImg[n + bb, m + aa, 0] = yxImg[v, u, 0]
                mapImg[n + bb, m + aa, 1] = yxImg[v, u, 1]
                mapImg[n + bb, m + aa, 2] = yxImg[v, u, 2]
            else:
                mapImg[n + bb, m + aa, 0] = 0
                mapImg[n + bb, m + aa, 1] = 0
                mapImg[n + bb, m + aa, 2] = 0
            # 记录鱼眼图像的位置
            pixel.append([u, v])
    # 返回展开后图像和展开位置的记录
    return mapImg, pixel


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
        return newton(lambda x: f(x) -
                                y, 1., f_prime, (), 1e-10, 1000000)

    # 返回方程的解
    return solve_


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
            [[R[0][0], R[0][1], -point[0]],
             [R[1][0], R[1][1], -point[1]],
             [R[2][0], R[2][1], -1]], dtype='float')
    # 对外参中的位移向量进行类型转换，float
    T = np.array(T, dtype='float')
    # 求解定位的计算式，得到一个三维向量，其中前两维是定位点的世界坐标的X与Y
    ans = solve(w, -T).T
    # 返回定位结果，世界坐标的X与Y
    return ans[0, :2]


def trans_corrdinate_without_inner(pixel_points, w, h, K, D, theta, fi, FOV, R):
    """鱼眼相机坐标转换算法
    :param pixel_points: 待映射点的像素坐标，shape:[2]
    :param w: 展开图像的宽
    :param h: 展开图像的高
    :param K: 内参矩阵
    :param D: 畸变系数，代表着鱼眼相机投影模型（如果是[4, ]的array，代表为多项式模型；如果是float，代表是等距投影模型，投影系数为k）
    :param theta: 入射角，即投影平面法线与xoy平面的夹角（0~90°）
    :param fi: 水平角，即投影平面法线投影xoy平面后，和x轴正半轴的夹角（0~360°）
    :param FOV: 视场角，即在投影方向上展开范围的角度（0~180°）
    :return: 返回待映射点在展开后图像上的像素坐标
    """
    THETA = theta / 180 * math.pi
    fi = fi / 180 * math.pi
    FOV = FOV / 180 * math.pi
    # 计算展开图像与投影半球面的切点，也是展开后图像的中心点
    x0, y0, z0 = R * math.cos(THETA) * math.cos(fi), R * math.cos(
        THETA) * math.sin(fi), R * math.sin(
        THETA)  # 直角坐标系--->球面坐标系
    # 计算每个像素代表的球面坐标系的距离
    d_pixel = R * (2 * math.tan(FOV / 2)) / w
    # 转换为array类型
    mtx = np.array(K)
    dist = np.array(D)
    x_d, y_d = (pixel_points[0] - mtx[0, 2]) / mtx[0, 0], (
                pixel_points[1] - mtx[1, 2]) / mtx[1, 1]
    theta_d = np.sqrt(x_d ** 2 + y_d ** 2) + 1e-9

    # 原版,需要改回去,函数最后一个参数要去掉
    status_D = True
    try:
        poly_solve = inverse(
            lambda x: x + dist[0] * x ** 3 + dist[1] * x ** 5 + dist[
                2] * x ** 7 + dist[3] * x ** 9)
        theta = poly_solve(theta_d)
    except RuntimeError:
        status_D = False
        theta = theta_d

    r = np.tan(theta)
    x_c, y_c = r * x_d / theta_d, r * y_d / theta_d
    FI = np.arctan2(x_d, y_d)
    A = np.array([
        [x0, y0, z0],
        [1, 0, -x_c],
        [0, 1, -y_c]
    ])
    # A = A.T
    b = np.array([
        x0 ** 2 + y0 ** 2 + z0 ** 2, 0, 0
    ])
    r = np.linalg.solve(A, b)
    dis = 99999
    X, Y = 0, 0
    for m in range(int(-h / 2), int(h / 2)):
        for n in range(int(-w / 2), int(w / 2)):
            # 计算展开后图像上的点相对图像中心点的偏移量
            d_x = n * d_pixel * math.sin(THETA) * math.cos(fi) + math.sin(
                fi) * m * d_pixel
            d_y = n * d_pixel * math.sin(THETA) * math.sin(fi) - math.cos(
                fi) * m * d_pixel
            d_z = - n * d_pixel * math.cos(THETA)
            x1, y1, z1 = x0 + d_x, y0 + d_y, z0 + d_z
            if (r[0] - x1) ** 2 + (r[1] - y1) ** 2 + (r[2] - z1) ** 2 < dis:
                dis = (r[0] - x1) ** 2 + (r[1] - y1) ** 2 + (r[2] - z1) ** 2
                X, Y = m + int(h / 2), n + int(w / 2)
    if X == 0 or Y == 0 or X == h - 1 or Y == w - 1:
        return None, status_D
    return np.array([w - Y, X]), status_D


if __name__ == '__main__':
    pixel = []
    K = np.zeros((3, 3))
    D = np.zeros((4))
    K[0, 0] = 150 / (math.pi / 2)
    K[0, 2] = 150
    K[1, 1] = 150 / (math.pi / 2)
    K[1, 2] = 150
    theta = 45
    fi = 45
    fov = 50
    w = 200
    h = 200
    # 将所有输入的角度转换为弧度制
    THETA = theta / 180 * math.pi
    FI = fi / 180 * math.pi
    FOV = fov / 180 * math.pi
    R = 150
    # 计算展开图像与投影半球面的切点，也是展开后图像的中心点
    # x0, y0, z0 = R * math.cos(THETA) * math.cos(FI), R * math.cos(THETA) * math.sin(FI), R * math.sin(
    #     THETA)  # 直角坐标系--->球面坐标系
    # # print(x0, y0, z0, THETA, FI, R)
    # # 计算每个像素代表的球面坐标系的距离
    # d_pixel = (2 * math.tan(FOV / 2) * R) / w
    # print(theta, fi, fov)
    a = trans_corrdinate_without_inner(np.array([190, 210]), 200, 200, K, D,
                                       theta, fi, fov)
    print(a)
    """
    n = a[0]-100
    m = a[1]-100
    d_x = n * d_pixel * math.sin(THETA) * math.cos(FI) + math.sin(FI) * m * d_pixel
    d_y = n * d_pixel * math.sin(THETA) * math.sin(FI) - math.cos(FI) * m * d_pixel
    d_z = - n * d_pixel * math.cos(THETA)
    # 计算展开后图像上的点在球面上的坐标
    x1, y1, z1 = x0 + d_x, y0 + d_y, z0 + d_z
    print(x1, y1, z1)
    rvec = np.array([0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 0.0])
    tvec = tvec[None, :]
    obj_points = np.array([x1, y1, z1])
    obj_points = obj_points[None, None, :]
    imagePoints = project_point_withK(obj_points, rvec, tvec, K, D)
    u = int(imagePoints[0])
    v = int(imagePoints[1])
    aa = int(h / 2)
    bb = int(w / 2)
    print(n + bb, m + aa, u, v)
    """
