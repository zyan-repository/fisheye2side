# *coding:utf-8 *
import os
import re
import cv2
import copy
import mmcv
import tqdm
import multiprocessing
from multiprocessing import cpu_count
import numpy as np
from functools import partial
from utils.unfold import project_unfold_fast, trans_corrdinate_without_inner
from auto_unfold import via_item_auto_unfold
from black_background2multicolor import black2multicolor
from shapely.geometry import Polygon, Point


# 读取via标注路径
LOKI_ann_file = 'D:/data/20220606092038/annotations/val.json'
# 展开后via标注保存路径
ann_save_dir = 'val.json'
# 原始图片文件夹
dir = 'D:/data/20220606092038/val/'
# 鱼眼展开后图片文件夹
new_dir = 'D:/data/0609/test_tran/'
# 鱼眼相机内参矩阵
K = np.array([[581.1058307906718, 0.0, 955.5987388116735],
              [0.0, 579.8976865646564, 974.0212406615763], [0.0, 0.0, 1.0]])
# 鱼眼相机畸变系数
D = np.array([-0.015964497003735242, -0.002789473611910958,
              0.005727838947159351, -0.0025185770227346576])

K_1080 = np.array([[326.872, 0.0, 537.524], [0.0, 326.192, 547.887],
                   [0.0, 0.0, 1.0]])
# 读取标注文件
# data_infos = mmcv.load(LOKI_ann_file)
# 最终的新json
res_dic = {}


def filter(filename, content, verbose=False):
    lens = re.findall(r'(.*?)%s(.*?)' % content, filename)
    status = True if len(lens) > 0 else False
    if not status and verbose:
        print('\n Skip %s' % filename)
    return status


def new_name(k, filename, fi):
    k = k.split('.')[0] + '_' + str(fi) + '.' + k.split('.')[1]
    filename = filename.split('.')[0] + '_' + str(fi) + '.' + \
               filename.split('.')[1]
    return k, filename


def mask_unfold(item, queue, img_dir, unfold_img_dir):
    k = item[0]
    v = item[1]
    filename = v['filename']
    # print(k.split('.')[0] + '_0.' + k.split('.')[1])
    fi, polygon = via_item_auto_unfold(item, img_dir, draw_lines=False)
    img = cv2.imread(os.path.join(img_dir, filename))
    img_h = img.shape[0]
    for f in fi:
        key, file_name = new_name(k, filename, f)
        status = filter(file_name, content='.', verbose=True)  # Note optional, '.' means leave all filenames
        if not status:
            continue
        dic = {'filename': file_name}  # 新图片的json
        regions = []
        flag = True
        # 遍历老图片的猪
        for pig in v['regions']:
            if pig['region_attributes']['type'] != 'pig':
                continue
            pig_dic = {}  # 新猪字典
            x = pig['shape_attributes']['all_points_x']
            y = pig['shape_attributes']['all_points_y']
            # 遍历猪的点，如果大于两个点则加入新图片
            cnt_points = 0
            new_points_x = []
            new_points_y = []

            for point in zip(x, y):
                if Point(point).intersects(polygon[f]):
                    if img_h == 1080:
                        new_point, status_D = trans_corrdinate_without_inner(
                            np.array(point), 1080, 1080, K_1080, D, 50, f, 90,
                            100)
                    else:
                        new_point, status_D = trans_corrdinate_without_inner(
                            np.array(point), 1080, 1080, K, D, 50, f, 90, 100)
                    if new_point is not None:
                        cnt_points += 1
                        new_points_x.append(new_point[1])
                        new_points_y.append(new_point[0])
                        if not status_D:
                            flag = False
            if cnt_points > 5:
                pig_dic['difficult'] = pig['difficult']
                pig_dic['region_attributes'] = pig['region_attributes']
                pig_dic['truncated'] = pig['truncated']
                pig_dic['shape_attributes'] = {
                    'all_points_x': new_points_x,
                    'all_points_y': new_points_y,
                    'name': pig['shape_attributes']['name']
                }
                regions.append(pig_dic)
        ori_img = copy.copy(img)
        # 外围涂黑
        points = Polygon([[0, 0], [img_h, 0], [img_h, img_h], [0, img_h],
                          [0, 0]]).difference(polygon[f]).exterior.coords[:]
        black_side_img = cv2.fillPoly(ori_img, [np.array(points).astype(int)],
                                      (0, 0, 0))
        # 固定入射角：侧拍高度
        # 改变水平角：侧拍角度
        # 固定视场角：侧拍范围
        if img_h == 1080:
            black_side_img = project_unfold_fast(1080, 1080, K_1080, D, 50, f,
                                                 90, black_side_img, 100, flag)
        elif img_h == 1920:
            black_side_img = project_unfold_fast(1080, 1080, K, D, 50, f, 90,
                                                 black_side_img, 100, flag)
        # 黑变彩色
        multicolor_side_img = black2multicolor(black_side_img)
        cv2.imwrite(
            os.path.join(unfold_img_dir, filename.split('.')[0] + '_' +
                         str(f) + '.' + filename.split('.')[1]),
            multicolor_side_img
        )
        dic['regions'] = regions
        dic['size'] = v['size']
        queue.put((key, dic))


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=cpu_count())
    param = []
    # 多进程需要用multiprocessing.Manager()里的队列，用multiprocessing队列会报错
    q = multiprocessing.Manager().Queue()
    for idx, item in enumerate(data_infos.items()):
        if item[1]['filename'] != 'BYZ_SH_NK_1641266048541.jpg':
            continue
        param.append(item)
    # print("item: ", param)
    with pool as p:
        # widgets = [Bar(), ETA()]
        # pbar = ProgressBar(widgets=widgets, maxval=len(param))
        # r = list(pbar(p.imap(partial(gao, queue=q), param)))
        r = list(tqdm.tqdm(p.imap(
            partial(mask_unfold, queue=q, img_dir=dir, unfold_img_dir=new_dir),
            param), total=len(param)))
    # pool.map(partial(gao, queue=q), param)
    # pool.close()
    # pool.join()
    print(q.qsize())
    while not q.empty():
        a = q.get()
        res_dic[a[0]] = a[1]
    mmcv.dump(res_dic, ann_save_dir)
