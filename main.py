# *coding:utf-8 *
import os
import mmcv
from multiprocessing import cpu_count, Manager
from functools import partial
from mmcv import Config
from mask2fisheye_unfold import mask_unfold
from cut_outside_black import ResourceContent, LokiLoadFetcher
from draw_points import draw


def generate_dir(judge_dir):
    if not os.path.exists(judge_dir):
        os.makedirs(judge_dir)


def main(test_single_image=False):
    cfg = Config.fromfile('configs.py')
    LOKI_ann_file = cfg.LOKI_ann_file
    ori_image_dir = cfg.ori_image_dir
    ann_save_dir = os.path.join(cfg.ann_save_dir, cfg.ann_save_name)
    unfold_image_dir = cfg.unfold_image_dir
    generate_dir(cfg.ann_save_dir)
    generate_dir(unfold_image_dir)
    points_image_dir = None
    for action in cfg.pipeline:
        if action['type'] == 'cut_outside':
            cut_image_dir = cfg.cut_image_dir
            generate_dir(cut_image_dir)
            print("start cut.")
            image = ResourceContent(LokiLoadFetcher())
            image.cut_image(os.path.join(LOKI_ann_file),
                            os.path.join(ori_image_dir),
                            os.path.join(cut_image_dir),
                            remain_folder=False,
                            mask_name=action['mask_name'])
            print("cut finished.")
            ori_image_dir = cut_image_dir
        if action['type'] == 'draw_points':
            points_image_dir = cfg.points_image_dir
            generate_dir(points_image_dir)

    print("start transform task.")
    tasks = []
    q = Manager().Queue()
    data_infos = mmcv.load(LOKI_ann_file)
    for idx, item in enumerate(data_infos.items()):
        if test_single_image:
            if item[1]['filename'] != test_single_image:
                continue
        tasks.append(item)
    mmcv.track_parallel_progress(
        partial(mask_unfold, queue=q, img_dir=ori_image_dir,
                unfold_img_dir=unfold_image_dir), tasks, cpu_count())
    res_dic = {}
    while not q.empty():
        a = q.get()
        res_dic[a[0]] = a[1]
    mmcv.dump(res_dic, ann_save_dir)
    print("transform task finished.")

    if points_image_dir is not None:
        draw(ann_save_dir, unfold_image_dir, points_image_dir)


if __name__ == '__main__':
    main()
