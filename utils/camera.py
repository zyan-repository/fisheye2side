"""
操作相机工具
"""
import sys

import cv2
import mmcv
import os
import timeout_decorator

# from software.components.utils.QMessageBox import show_q_message_box


# 使用rtsp流打开相机
# @timeout_decorator.timeout(1)
def open_camera(username: str, password: str, ip: str):
    """
    使用rtsp流打开相机
    rtsp格式：rtsp://[username]:[password]@[ip]:[port]/[codec]/[channel]/[subtype]/av_stream
        username: 用户名。例如admin。
        password: 密码。例如12345。
        ip: 为设备IP。例如 192.0.0.64。
        port: 端口号默认为554，若为默认可不填写。
        codec：有h264、MPEG-4、mpeg4这几种。
        channel: 通道号，起始为1。例如通道1，则为ch1。
        subtype: 码流类型，主码流为main，辅码流为sub。
    :return:相机是否打开，相机
    """
    try:
        # 使用rtsp流打开相机
        # cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cam = cv2.VideoCapture('rtsp://{}:{}@{}/h264/ch1/main/av_stream'.format(username, password, ip))
        ret, frame = cam.read()
        cnt = 0
        while True:
            if cnt > 10:
                cv2.imwrite("D:/camera_test/" + "6240zm" + '.jpg', frame)
                # cv2.imencode('.jpg', frame)[1].tofile("D:/camera_test/PLZ/" + ip)
                print(cnt)
                break
            else:
                cnt += 1
        return True, cam

    except:
        pass
    # except cv2.error:
    #     # 捕获cv异常
    #     # 打开相机失败
    #     show_q_message_box("连接相机失败", "连接相机失败，请重新设置相机IP地址，打开相机")
    #     sys.exit()
    #
    # except cv2.Error:
    #     show_q_message_box("连接相机失败", "连接相机失败，请重新设置相机IP地址，打开相机")
    #     sys.exit()


# 判断相机是否成功打开
def is_opened(cam: cv2.VideoCapture):
    return cam.isOpened()


# 关闭相机
def close_camera(cam: cv2.VideoCapture):
    """
    销毁相机
    :param cam: 相机
    :return:
    """
    cam.release()


# 读取相机
def read(cam: cv2.VideoCapture):
    """
    读取相机
    :param cam: 相机
    :return:是否读取成功，读取后获得的帧图
    """
    # 读取相机
    ret, img = cam.read()

    if ret:
        # 读取相机成功，返回读取到的BGR图像
        return True, img

    else:
        # 读取相机失败
        return False, None


# 保存图片
def keep_image(cam: cv2.VideoCapture, path: str):
    """
    保存图片
    :param cam: 相机
    :param path: 保存路径，包括文件名及文件后缀
    :return: 是否保存成功
    """
    # 读取相机
    ret, img = read(cam)
    if ret:
        # 读取相机成功，保存图片
        # 使用cv2.imwirte()保存图片，无法保存中文路径
        cv2.imencode('.jpg', img)[1].tofile(path)
        return True

    else:
        # 保存失败
        return False


if __name__ == '__main__':
    # cap = cv2.VideoCapture("rtsp://admin:shuzhi123@10.100.22.35:554/h264/ch1/main/av_stream")  # 普利兹鱼眼
    # for ip in mmcv.track_iter_progress(range(255)):
    # 北京普利兹摄像头
    # open_camera("admin", "shuzhi123", "10.100.22.43:554")
    # 北京测温摄像头
    open_camera("admin", "NXIN1234", "10.100.22.44:554")
