# *coding:utf-8 *
# 在猪舍原图上判断两条射线是否与标签相交
# 用两条夹角90°的射线切割图像，确保切割出的块中猪只完整，从而确定鱼眼展开的水平角fi
# 对每条射线计算和图像边缘线段的交点
# 暂时只支持处理长宽相等的图像
import os
import cv2
import copy
import mmcv
import math
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Polygon
from sympy import symbols, Eq, solve
from utils.unfold import project_unfold_fast

# ann_info = mmcv.load('D:/data/20220606092038/annotations/val.json')
# ann_info = mmcv.load('D:/data/20220617180937/train_via/via_region_data.json')
# img_dir = 'D:/data/20220606092038/val/'
# save_dir = 'D:/data/0609/val_step_1/'
# unfold_dir = 'D:/data/0609/black_unfold/'


# 鱼眼相机内参矩阵
K = np.array([[581.1058307906718, 0.0, 955.5987388116735],
              [0.0, 579.8976865646564, 974.0212406615763], [0.0, 0.0, 1.0]])
# 鱼眼相机畸变系数
D = np.array([-0.015964497003735242, -0.002789473611910958,
              0.005727838947159351, -0.0025185770227346576])

K_1080 = np.array([[326.872, 0.0, 537.524], [0.0, 326.192, 547.887],
                   [0.0, 0.0, 1.0]])


# 求二维平面两个向量顺时针夹角
def clockwise_angle(v1, v2):
    x1, y1 = v1
    x2, y2 = v2
    # dot = |v1| * |v2| * cos(theta)
    dot = x1 * x2 + y1 * y2
    # cross = |v1| * |v2| * sin(theta)
    cross = x1 * y2 - y1 * x2
    # 用arctan是考虑到其值域是[-pi, pi]
    theta = np.arctan2(cross, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    return math.degrees(theta)


def via_item_auto_unfold(
        item,
        img_dir,
        unfold_dir=None,
        cut_dir=None,
        ori_dir=None,
        status_D=True,
        draw_lines=False,
        test_single_image=False
):
    img = cv2.imread(os.path.join(img_dir, item[1]['filename']))
    w = img.shape[1]
    h = img.shape[0]
    r = int(96 / 1920 * w + 0.5)

    # 测试单张图片
    if test_single_image:
        if item[1]['filename'] != test_single_image:
            return None, None

    polygon = []
    for pig in item[1]['regions']:
        if pig['region_attributes']['type'] != 'pig':
            continue
        point_lst = []
        for point in zip(pig['shape_attributes']['all_points_x'],
                         pig['shape_attributes']['all_points_y']):
            point_lst.append(point)
        polygon.append(Polygon(point_lst))

    # 右 下 左 上
    # 对应鱼眼展开的水平角
    # 315-360,0-45 45-135 135-225 225-315
    direct = [h, w, h, w]
    # 记录每条线的选择状态
    line_status = []
    # 对每条线坐标进行记录
    line_record = []
    # 镜头每条边上线的个数
    line_num = []
    # 图像中心点坐标
    center = [int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)]
    for idx, length in enumerate(direct):
        cnt = 0
        for i in range(0, length, 40):
            if idx == 0:
                point = (w, i)
            elif idx == 1:
                point = (length - i, h)
            elif idx == 2:
                point = (0, length - i)
            elif idx == 3:
                point = (i, 0)
            cnt += 1
            x, y, k, b = symbols('x y k b')
            eqs = [
                Eq((x - center[0]) ** 2 + (y - center[1]) ** 2, r ** 2),
                Eq(k * center[0] + b, center[1]),
                Eq(k * point[0] + b, point[1]),
                Eq(k * x + b, y)
            ]
            try:
                s1, s2 = solve(eqs, [x, y, k, b])
                s1 = (s1[0].evalf(), s1[1].evalf())
                s2 = (s2[0].evalf(), s2[1].evalf())
            # 无k值，直线垂直x轴
            except Exception as e:
                s1 = (center[0], center[1] - r)
                s2 = (center[0], center[1] + r)
            if max(point[0], center[0]) >= s1[0] >= min(point[0],
                                                        center[0]) and max(
                point[1], center[1]) >= s1[1] >= min(point[1], center[1]):
                circle_point = [s1[0], s1[1]]
            else:
                circle_point = [s2[0], s2[1]]
            circle_point = [int(circle_point[0] + 0.5),
                            int(circle_point[1] + 0.5)]
            line = LineString([circle_point, point])
            flag = 0
            for pig in polygon:
                if line.intersects(pig):
                    flag = 1
                    break
            if flag:
                if draw_lines:
                    img = cv2.line(img, circle_point, point, (0, 0, 255), 2)
                line_status.append(1)
            else:
                if draw_lines:
                    img = cv2.line(img, circle_point, point, (0, 255, 0), 2)
                line_status.append(0)
            line_record.append([circle_point, point])
        line_num.append(cnt)
    for i in range(1, 4):
        line_num[i] = line_num[i - 1] + line_num[i]
    is_ring = 0
    if line_status[0] == line_status[-1] == 1:
        is_ring = 1

    # 线的总数
    line_sum = len(line_status)
    cluster_info = []
    ori_img = copy.copy(img)

    is_used = {}
    # 暂时只支持处理长宽相等的图像
    cluster_max_num = int((w + 1) / 40) - 1
    # 0°展开点表示的向量
    v0 = [center[0], 0]
    # 避免重复的展开角度
    unfold_angle = {}
    # 角度：展开区域Polygon
    polygon = {}
    # 有环，需要对环特判
    if is_ring:
        ring_start = line_sum - 1
        ring_end = 0
        is_used[0] = 1
        is_used[line_sum - 1] = 1
        for i in range(line_sum - 2, 0, -1):
            if line_status[i] == 1 and i not in is_used:
                is_used[i] = 1
                ring_start = i
            else:
                break
        for i in range(1, line_sum - 1):
            if line_status[i] == 1 and i not in is_used:
                is_used[i] = 1
                ring_end = i
            else:
                break
        len_ring = line_sum - ring_start + ring_end + 1
        success = 0
        # 环可用
        if len_ring <= cluster_max_num:
            start_line = ring_end + 1
            end_line = line_sum - (cluster_max_num - ring_end - 1) - 1
            while end_line != ring_start:
                if line_status[start_line] == line_status[end_line] == 0:
                    success = 1
                    break
                start_line = start_line + 1
                end_line = end_line + 1
        if success:
            # print("ring success:", start_line, end_line, item[1]['filename'])
            color = (np.random.randint(256), np.random.randint(180),
                     np.random.randint(180))
            if draw_lines:
                img = cv2.line(img, line_record[start_line][0],
                               line_record[start_line][1], color, 2)
                img = cv2.line(img, line_record[end_line][0],
                               line_record[end_line][1], color, 2)

            points = [line_record[end_line][1], line_record[end_line][0]]
            idx = (end_line + 1) % line_sum
            for i in range(cluster_max_num):
                points.append(line_record[idx][0])
                idx = (idx + 1) % line_sum
            points.extend(
                [line_record[start_line][0], line_record[start_line][1], [w, h],
                 [0, h], [0, 0]])
            points = np.array(points)
            ring_img = copy.copy(img)
            ring_img = cv2.fillPoly(ring_img, [points], (0, 0, 0))

            v1 = [line_record[end_line][1][0] - center[0],
                  line_record[end_line][1][1] - center[1]]
            angle = (int(clockwise_angle(v0, v1) + 0.5) + 45) % 360
            if w == 1080 and angle not in unfold_angle:
                unfold_angle[angle] = 1
                polygon[angle] = Polygon(
                    [[0, 0], [w, 0], [w, h], [0, h], [0, 0]]).difference(
                    Polygon(points))
                if unfold_dir is not None:
                    cv2.imwrite(
                        os.path.join(
                            unfold_dir,
                            item[1]['filename'].split('.')[0] + '_' +
                            str(angle) + '.' + item[1]['filename'][-3:]
                        ),
                        project_unfold_fast(
                            1080, 1080, K_1080, D, 50, angle, 90, ring_img,
                            100, status_D
                        )
                    )
            elif w == 1920 and angle not in unfold_angle:
                unfold_angle[angle] = 1
                polygon[angle] = Polygon(
                    [[0, 0], [w, 0], [w, h], [0, h], [0, 0]]).difference(
                    Polygon(points))
                if unfold_dir is not None:
                    cv2.imwrite(
                        os.path.join(
                            unfold_dir,
                            item[1]['filename'].split('.')[0] + '_' +
                            str(angle) + '.' + item[1]['filename'][-3:]
                        ),
                        project_unfold_fast(
                            1080, 1080, K, D, 50, angle, 90, ring_img,
                            100, status_D
                        )
                    )
    # 无环，按数组处理
    # flag 0 非簇状态 1 簇状态 用于确定簇的开始结束位置
    flag = 0
    cnt = 0
    for idx, status in enumerate(line_status):
        # 找到簇起始位，重置计数器，状态符变为簇状态
        if status == 1 and flag == 0 and idx not in is_used:
            flag = 1
            cnt = 1
            is_used[idx] = 1
            start = idx
        # 在簇中，计数器加1
        elif status == 1 and flag == 1 and idx not in is_used:
            is_used[idx] = 1
            cnt += 1
        # 簇在上一位结束，状态变为非簇
        elif status == 0 and flag == 1:
            flag = 0
            end = idx - 1
            cluster_info.append((start, end))
    # 对图片中所有簇进行遍历，找出所有满足限制的簇
    # 簇的视场角固定为90°，以1920*1920间隔40像素为例，一簇最大为47根线，共展开49根线内的栏位
    for idx, info in enumerate(cluster_info):
        start = info[0]
        end = info[1]
        cluster_len = end - start + 1
        # 满足要求的簇
        if cluster_len <= cluster_max_num:
            start_line = (end + 1) % line_sum
            end_line = start - (cluster_max_num - cluster_len) - 1
            if end_line < 0:
                end_line = line_sum + end_line
            success = 0
            while end_line != start:
                if line_status[start_line] == line_status[end_line] == 0:
                    success = 1
                    break
                start_line = (start_line + 1) % line_sum
                end_line = (end_line + 1) % line_sum
            if success:
                # print("idx: ", idx, start_line, end_line)
                color = (np.random.randint(256), np.random.randint(180),
                         np.random.randint(180))
                if draw_lines:
                    img = cv2.line(img, line_record[start_line][0],
                                   line_record[start_line][1], color, 2)
                    img = cv2.line(img, line_record[end_line][0],
                                   line_record[end_line][1], color, 2)

                extend_lst = []
                # ring(不会出现，已经单独处理)或者正好在315°-45°
                if start_line <= line_num[0]:
                    extend_lst = [[w, h], [0, h], [0, 0]]
                elif start_line <= line_num[1]:
                    extend_lst = [[0, h], [0, 0], [w, 0]]
                elif start_line <= line_num[2]:
                    extend_lst = [[0, 0], [w, 0], [w, h]]
                elif start_line <= line_num[3]:
                    extend_lst = [[w, 0], [w, h], [0, h]]
                points = [line_record[end_line][1], line_record[end_line][0]]
                idx = (end_line + 1) % line_sum
                for i in range(cluster_max_num):
                    points.append(line_record[idx][0])
                    idx = (idx + 1) % line_sum
                points.extend(
                    [line_record[start_line][0], line_record[start_line][1]])
                points.extend(extend_lst)
                points = np.array(points)
                line_img = copy.copy(img)
                line_img = cv2.fillPoly(line_img, [points], (0, 0, 0))

                v1 = [line_record[end_line][1][0] - center[0],
                      line_record[end_line][1][1] - center[1]]
                angle = (int(clockwise_angle(v0, v1) + 0.5) + 45) % 360
                if w == 1080 and angle not in unfold_angle:
                    unfold_angle[angle] = 1
                    polygon[angle] = Polygon(
                        [[0, 0], [w, 0], [w, h], [0, h], [0, 0]]).difference(
                        Polygon(points))
                    if unfold_dir is not None:
                        cv2.imwrite(
                            os.path.join(
                                unfold_dir,
                                item[1]['filename'].split('.')[0] + '_' +
                                str(angle) + '.' + item[1]['filename'][-3:]),
                            project_unfold_fast(1080, 1080, K_1080, D, 50,
                                                angle, 90, line_img, 100,
                                                status_D)
                        )
                elif w == 1920 and angle not in unfold_angle:
                    unfold_angle[angle] = 1
                    polygon[angle] = Polygon(
                        [[0, 0], [w, 0], [w, h], [0, h], [0, 0]]).difference(
                        Polygon(points))
                    if unfold_dir is not None:
                        cv2.imwrite(
                            os.path.join(
                                unfold_dir,
                                item[1]['filename'].split('.')[0] + '_' +
                                str(angle) + '.' + item[1]['filename'][-3:]
                            ),
                            project_unfold_fast(
                                1080, 1080, K, D, 50, angle, 90, line_img,
                                100, status_D
                            )
                        )
    if ori_dir is not None:
        cv2.imwrite(os.path.join(ori_dir, item[1]['filename']), ori_img)
    if cut_dir is not None:
        cv2.imwrite(os.path.join(cut_dir, item[1]['filename']), img)
    return unfold_angle.keys(), polygon


if __name__ == '__main__':
    for item in mmcv.track_iter_progress(ann_info.items()):
        unfold_angle_key, polygon = \
            via_item_auto_unfold(
                item,
                'D:/data/20220617180937/train/',
                unfold_dir='./',
                draw_lines=True,
                test_single_image='BYZ_1626763099525.jpg'
            )
        print(unfold_angle_key)
    # via_item_auto_unfold(item, 'D:/data/20220606092038/val/',
    # 'D:/data/0609/black_unfold/', draw_lines=True)
