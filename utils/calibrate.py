"""
标定算法工具包
    保持所有有关标定的相关算法
"""
import cv2
import numpy as np

np.set_printoptions(suppress=True)


# 根据棋盘格规格，生成世界坐标
def get_obj_points(checker_board: tuple):
    """
    生成世界坐标
    根据标定版规格，自动生成世界坐标
    在进行相机内参数标定时，相机的内参数信息为固定值，与世界坐标无关
    所以根据图片的棋盘格规格，只需生成相应的世界坐标，即可进行标定
    :param checker_board: (row, col)
    :return: obj_points: 世界坐标
    """
    obj_points = np.zeros((1, checker_board[0] * checker_board[1], 3), dtype=np.float32)
    obj_points[0, :, :2] = np.mgrid[
                           0:checker_board[0]:1,
                           0:checker_board[1]:1
                           ].T.reshape(-1, 2)
    return obj_points


# 寻找亚像素焦点
def _find_chessboard_corners(gray, checker_board, something=None):
    """
    寻找标定板交点，在此基础上寻找亚像素焦点优化。
    :param gray: 需要求角点的灰度图
    :param checker_board: 角点的横纵数量，格式为元组
    :param something: 不知道是干什么的参数，如果之后有用再修改
    :return: 角点寻找是否成功与像素角点， shape:[num, 2]
    """
    ret, corners = cv2.findChessboardCorners(gray, checker_board, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    if ret:
        temp_corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        if [temp_corners]:
            corners = temp_corners
    return ret, corners


# 寻找棋盘格四个顶点的值
def find_image_points(gray, checker_board):
    """
    获取像素坐标四个顶点的值
    :return:
    """
    _, corners = _find_chessboard_corners(gray, checker_board)
    return np.array([
        corners[0].tolist()[0],
        corners[checker_board[0] - 1].tolist()[0],
        corners[checker_board[0] * (checker_board[1] - 1)].tolist()[0],
        corners[checker_board[0] * checker_board[1] - 1].tolist()[0]
    ])


# 单目相机的内参数标定
def cal_internal_monocular(obj_p, img_list, checker_board):
    """
    单目相机标定
    :param obj_p: 角点的世界坐标，shape:[num, 3]
    :param img_list: 供标定所用的图片列表，里面都是BGR图像
    :param checker_board: 角点的横纵数量，格式为元组
    :return: 标定是否成功，相机内参数矩阵，畸变系数
    """
    try:
        # 存储3D点
        obj_points = []
        # 存储2D点
        img_points = []
        # 储存灰度图规格
        gray_size = None
        # 遍历图片列表
        for img_ in img_list:
            # 将图片转为灰度图
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            # 获取灰度图规格
            gray_size = gray.shape[::-1]
            # 寻找棋盘格上的亚像素角点
            ret, corners = _find_chessboard_corners(gray, checker_board)
            # 寻找成功
            if ret:
                # 将世界坐标添加到世界坐标列表中
                obj_points.append(obj_p)
                # 将寻找到的亚像素角点添加到像素坐标列表中
                img_points.append(corners)
        # 进行相机的内参数标定
        ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, gray_size, None, None,
                                                   flags=cv2.CALIB_THIN_PRISM_MODEL)

        # 返回相关信息
        return ret, mtx, dist
    except cv2.error as e:
        # 标定失败，返回相关信息
        return False, None, None


# 使用棋盘格标定相机外参
def cal_outside_image_monocular(obj_p, img_, checker_board, mtx, dist):
    """
    根据内参，图像，世界坐标，求出对于当前世界坐标系下的外参（旋转、平移）
    :param obj_p: 角点的世界坐标，shape:[]
    :param img_: 用于标定的图片
    :param checker_board: 角点的横纵数量，格式为元组
    :param mtx: 相机内参矩阵
    :param dist: 相机畸变向量，shape:[5, ]
    :return: 相机的外参，包含旋转向量，旋转矩阵，平移向量
    """
    try:
        # 将图片转换为灰度图
        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        # 寻找棋盘格四个顶点的像素坐标
        corners = find_image_points(gray, checker_board)
        # 使用pnp方法进行相机的外参标定
        ret, rvec, tvec = cv2.solvePnP(obj_p, corners.squeeze(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), None)
        # 将旋转向量转换为旋转矩阵
        rmat = cv2.Rodrigues(rvec)[0]
        # 返回相关信息
        return True, rvec, rmat, tvec
    except cv2.error:
        # 标定失败，返回相关信息
        return False, None, None, None


# 使用标点法进行相机的外参标定
def cal_outside_punctuation_monocular(obj_point, pix_point, mtx, dist):
    """
    使用标点法进行外参标定
    :param obj_point: 角点的世界坐标，shape:[]
    :param pix_point: 角点的像素坐标
    :param mtx: 相机内参矩阵
    :param dist: 相机畸变向量，shape:[5, ]
    :return: 是否标定成功，旋转向量，旋转矩阵，平移向量
    """
    try:
        # 使用pnp方法进行相机的外参标定
        ret, rvec, tvec = cv2.solvePnP(obj_point, pix_point, mtx, dist)
        # 将旋转向量转换为旋转矩阵
        rmat = cv2.Rodrigues(rvec)[0]
        # 返回相关信息
        return True, rvec, rmat, tvec
    except cv2.error:
        # 标定失败，返回相关信息
        return False, None, None, None


# 鱼眼相机的内参标定
def cal_internal_fisheye(obj_p, img_list, checker_board):
    """
    鱼眼相机的内参标定
    :param obj_p: 角点的世界坐标，shape:[1, num, 3]
    :param img_list: 供标定所用的图片列表，里面都是BGR图像
    :param checker_board: 角点的横纵数量，格式为元组
    :return: 标定是否成功，相机矩阵，畸变系数，K，D
    """
    try:
        # 存储3D点
        obj_points = []
        # 存储2D点
        img_points = []
        # 储存灰度图规格
        gray_size = None
        # 遍历图片列表
        for img_ in img_list:
            # 将图片转为灰度图
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            # 获取灰度图规格
            gray_size = gray.shape[::-1]
            # 寻找棋盘格上的亚像素角点
            ret, corners = _find_chessboard_corners(gray, checker_board)
            # 寻找成功
            if ret:
                # 将世界坐标添加到世界坐标列表中
                obj_points.append(obj_p)
                # 将寻找到的亚像素角点添加到像素坐标列表中
                img_points.append(corners)
        # 生成空相机固有矩阵
        K = np.zeros((3, 3))
        # 生成空的失真系数的输出向量
        D = np.zeros((4, 1))

        # 进行鱼眼相机的内参数标定
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points,
            img_points,
            gray_size,
            K,
            D,
            None,
            None,
            cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        # 返回相关信息
        return ret, mtx, dist
    except cv2.error:
        # 标定失败，返回相关信息
        return False, None, None


# 使用棋盘格标定鱼眼相机外参
def cal_outside_image_fisheye(obj_p, img_, checker_board, mtx, dist):
    """
    根据内参，图像，世界坐标，求出对于当前世界坐标系下的外参（旋转、平移）
    检测角点、使用内参和矫正系数修正点坐标、solvePnP（不需要矫正系数，相机矩阵为标准矩阵）
    :param obj_p: 角点的世界坐标，shape:[]
    :param img_: 用于标定的图片
    :param checker_board: 角点的横纵数量，格式为元组
    :param mtx: 相机内参矩阵
    :param dist: 相机畸变向量，shape:[4, ]
    :return: 相机的外参，包含旋转向量，旋转矩阵，平移向量
    """
    try:
        # 将图片转换为灰度图
        gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
        # 寻找棋盘格四个顶点的像素坐标
        corners = find_image_points(gray, checker_board)
        img_p = cv2.fisheye.undistortPoints(corners, mtx, dist)
        ret, rvec, tvec = cv2.solvePnP(obj_p.squeeze(), img_p.squeeze(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                       None)
        # 将旋转向量转换为旋转矩阵
        rmat = cv2.Rodrigues(rvec)[0]
        # 返回相关信息
        return True, rvec, rmat, tvec
    except cv2.error:
        # 标定失败，返回相关信息
        return False, None, None, None


# 使用标点法标点鱼眼相机外参
def cal_outside_punctuation_fisheye(obj_point, img_point, mtx, dist):
    """
    使用标点法标点鱼眼相机外参
    :param obj_point: 角点的世界坐标，shape:[]
    :param img_point: 角点的像素坐标
    :param mtx: 相机内参矩阵
    :param dist: 相机畸变向量，shape:[5, ]
    :return: 是否标定成功，旋转向量，旋转矩阵，平移向量
    """
    try:
        img_point = cv2.fisheye.undistortPoints(img_point, mtx, dist)
        # 使用pnp进行鱼眼相机的外参标定，其中第三个参数为单位矩阵
        ret, rvec, tvec = cv2.solvePnP(obj_point, img_point.squeeze(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                       None)
        # 将旋转向量转换为旋转矩阵
        rmat = cv2.Rodrigues(rvec)[0]
        # 返回相关信息
        return True, rvec, rmat, tvec
    except cv2.error:
        # 标定失败，返回相关信息
        return False, None, None, None
