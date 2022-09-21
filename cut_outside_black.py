import os
import re
import cv2
import json
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count
from IPython import embed

def filename_filter(filename, content, verbose=False):
    lens = re.findall(r'(.*?)%s(.*?)' % content, filename)
    status = True if len(lens) > 0 else False
    if not status and verbose:
        print('\n Skip %s' % filename)
    return status


def count_file(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for each in files:
            if each[-4:] != 'json':
                count += 1
    return count


class ResourceContent:
    def __init__(self, imp):
        self._imp = imp

    def cut_image(self, file_load_from, img_read_from_dir, img_write_to_dir, remain_folder=True, mask_name=None):
        self._imp.fetch(file_load_from, img_read_from_dir, img_write_to_dir, remain_folder, mask_name)


class ResourceContentFetcher(metaclass=ABCMeta):
    @abstractmethod
    def fetch(self, file_load_from, img_read_from, img_write_to_dir, remain_folder=True, mask_name=None):
        pass


class FileLoadFetcher(ResourceContentFetcher):
    def do_mask(self, dorm_name, img_name, px, py, img_read_from_dir, img_write_to_dir, remain_folder):
        image = cv2.imdecode(
            np.fromfile(img_read_from_dir + '/' + dorm_name + '/' + img_name, dtype=np.uint8), -1)
        mask = ~np.zeros_like(image)
        random_mask = (np.random.random(image.shape) * 255).astype(np.uint8)
        poly = [[x, y] for x, y in zip(px, py)]
        poly = np.array(poly)
        mask = ~mask
        cv2.fillPoly(mask, [poly], (255, 255, 255))
        cv2.fillPoly(random_mask, [poly], (255, 255, 255))
        image = cv2.bitwise_and(image, mask)
        # random_mask = cv2.bitwise_xor(mask, random_mask)
        # image += random_mask
        try:
            if not os.path.exists(img_write_to_dir):
                os.makedirs(img_write_to_dir)
            if not os.path.exists(img_write_to_dir + '/' + dorm_name) and remain_folder:
                os.makedirs(img_write_to_dir + '/' + dorm_name)
        except FileExistsError:
            pass
        if remain_folder:
            cv2.imencode('.jpg', image)[1].tofile(
                img_write_to_dir + '/' + dorm_name + '/' + img_name.split('.')[0] + '.jpg')
        else:
            cv2.imencode('.jpg', image)[1].tofile(img_write_to_dir + '/' + img_name.split('.')[0] + '.jpg')

    def fetch(self, file_load_from, img_read_from_dir, img_write_to_dir, remain_folder=True, mask_name=None):
        pool = multiprocessing.Pool(processes=cpu_count())
        config_lst = os.listdir(file_load_from)
        file_lst = os.listdir(img_read_from_dir)
        file_num = count_file(img_read_from_dir)
        pbar = tqdm(total=file_num)
        pbar.set_description(' Flow ')
        update = lambda *args: pbar.update()
        for dorm_name in file_lst:
            flag = False
            for config_name in config_lst:
                file_name = config_name.split('.')
                if dorm_name == file_name[0]:
                    flag = True
                    break
            if not flag:
                config_file_name = 'default_RegionChoicer'
            else:
                config_file_name = dorm_name + '.mp4_RegionChoicer'
            df = pd.read_csv(file_load_from + '/' + config_file_name, names=['xy'])
            px = []
            py = []
            for xy in df.values:
                c = xy[0].split(' ')
                px.append(int(float(c[0])))
                py.append(int(float(c[1])))
            img_lst = os.listdir(img_read_from_dir + '/' + dorm_name)
            for img_name in img_lst:
                pool.apply_async(FileLoadFetcher().do_mask,
                                 (dorm_name, img_name, px, py, img_read_from_dir, img_write_to_dir, remain_folder),
                                 callback=update)
        pool.close()
        pool.join()


skip = True


class LokiLoadFetcher(ResourceContentFetcher):
    def do_mask(self, v, img_read_from_dir, img_write_to_dir, remain_folder, mask_name):
        filename = img_read_from_dir + '/' + v['filename']
        # Note: filename filter, edit every time you run, '.' means leave all filename
        status = filename_filter(v['filename'], content='.', verbose=False)
        if not status:
            return
        if os.path.exists(filename):
            image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        else:
            print("\n ***Warning %s not found" % filename)
            return
        mask = ~np.zeros_like(image)
        random_mask = (np.random.random(image.shape) * 0).astype(np.uint8)
        found_mask = False
        for region in v['regions']:
            if region['region_attributes']['type'] == mask_name:
                found_mask = True
                obj = region['shape_attributes']
                px = obj['all_points_x']
                py = obj['all_points_y']
                poly = [[x, y] for x, y in zip(px, py)]
                poly = np.array(poly)
                mask = ~mask
                cv2.fillPoly(mask, [poly], (255, 255, 255))
                cv2.fillPoly(random_mask, [poly], (255, 255, 255))
        if not found_mask:
            print("\nNo Mask found %s" % filename)
            if skip:
                print("skip no mask")
                return
        else:
            image = cv2.bitwise_and(image, mask)
            random_mask = cv2.bitwise_xor(mask, random_mask)
            image += random_mask
        try:
            if not os.path.exists(img_write_to_dir):
                os.makedirs(img_write_to_dir)
        except FileExistsError:
            pass
        if remain_folder:
            cv2.imencode('.jpg', image)[1].tofile(
                img_write_to_dir + '/' + v['filename'].split('.')[0] + '.jpg')
        else:
            cv2.imencode('.jpg', image)[1].tofile(img_write_to_dir + '/' + v['filename'].split('.')[0] + '.jpg')

    def fetch(self, file_load_from, img_read_from_dir, img_write_to_dir, remain_folder=True, mask_name=None):
        pool = multiprocessing.Pool(processes=cpu_count())
        file_num = count_file(img_read_from_dir)
        pbar = tqdm(total=file_num)
        pbar.set_description(' Flow ')
        update = lambda *args: pbar.update()
        with open(file_load_from) as loki_json_file:
            train_data_infos = json.load(loki_json_file)
        for idx, v in enumerate(train_data_infos.values()):
            pool.apply_async(LokiLoadFetcher().do_mask,
                             (v, img_read_from_dir, img_write_to_dir, remain_folder, mask_name), callback=update)
        pool.close()
        pool.join()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def main():
    parser = argparse.ArgumentParser(description="Cut image")
    parser.add_argument('-s', '--standard', type=str, choices=['file', 'loki'], help="Cut standard")
    parser.add_argument('-f', '--file_load_from', type=str, help="File direction of cut areas")
    parser.add_argument('-r', '--img_read_from_dir', type=str, help="Original image direction")
    parser.add_argument('-w', '--img_write_to_dir', type=str, help="Mask image direction")
    parser.add_argument('-re', '--remain_folder', type=str2bool, choices=[True, False], default=True,
                        help="Only 'file' effective")
    parser.add_argument('-m', '--mask_name', type=str, default=None, help="Only 'loki' effective")
    args = parser.parse_args()

    standard = args.standard
    file_load_from = args.file_load_from
    img_read_from_dir = args.img_read_from_dir
    img_write_to_dir = args.img_write_to_dir
    remain_folder = args.remain_folder
    mask_name = args.mask_name

    if standard == 'file':
        file_load_fetcher = FileLoadFetcher()
    elif standard == 'loki':
        file_load_fetcher = LokiLoadFetcher()
    image = ResourceContent(file_load_fetcher)
    image.cut_image(file_load_from, img_read_from_dir, img_write_to_dir, remain_folder=remain_folder,
                    mask_name=mask_name)


# cut标准 loki、file
# 文件地址（loki为一个json文件，file是一个存放各栏坐标信息的文件夹(里面文件命名为'栏位'+'.mp4_FourPointChoicer'，对于没有cut信息的栏位统一采用'default_RegionChoicer'里面的数据cut， 如：保育1-1-JK1.mp4_FourPointChoicer)）
# 需要cut的图片地址（一个文件夹,对于file标准图片所在文件夹需要包含栏位信息（如：保育1-6-JK9））
# cut后输出图片的地址
# True 保留原文件夹信息信息，原来有什么文件夹就输出在什么文件夹 False 所有图片输出在同一个文件夹(仅file有效)
# cut坐标在json文件中对应的的关键字（仅loki标准有效）


# Sample
# 猪舍围栏 pigsty_corral
# python cut_outside_black.py  -s loki -f D:/HFSFile/20210727172611/train/via_region_data.json -r D:/HFSFile/20210727172611/train -w D:/HFSFile/czc/train -m center_ROI
# python cut_outside_black.py  -s file -f D:/nxin/regions_BYZ_nankou -r D:/run_on -w D:/run_on_cut -re False
if __name__ == '__main__':
    main()
