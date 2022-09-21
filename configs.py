# *coding:utf-8 *
# pipeline不用需要注释掉，都不用需要给空列表
# cut_outside用于把栏外未涂黑的情况下涂黑
# mask_name是标注中围栏的key，猪舍围栏pigsty_corral，中心区域center_ROI，需要自己确认
# draw_points用于把鱼眼展开后的标注点画在鱼眼展开后的图上，是用来验证结果的，去掉对结果无影响
pipeline = [
    dict(
        type="cut_outside",
        mask_name="pigsty_corral"
    ),
    dict(type='draw_points')
]
# 读取via标注路径
LOKI_ann_file = 'D:/data/20220706172725/annotations/train.json'
# 原始图片文件夹
ori_image_dir = 'D:/data/20220706172725/train/'
# 展开后via标注保存路径
ann_save_dir = './'
# 展开后via标注保存名称
ann_save_name = 'val.json'
# 鱼眼展开后图片文件夹
unfold_image_dir = 'D:/data/0609/test_tran/'
# cut图片保存地址
cut_image_dir = 'D:/zcut/'
# 画点图片保存地址
points_image_dir = 'D:/zpoints/'
