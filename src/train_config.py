# -*- coding: UTF-8 -*-
"""
训练基本配置
"""
import os
from utils import logger


train_parameters = {
    "current_mode": "train",        # 当前模式，训练模式需要初始化图片数量，其他模式不用
    "data_dir": "../data/",
    "train_data_dir": "../data/trainImageSet/",
    "eval_data_dir": "../data/evalImageSet/",
    "image_count": -1,
    "continue_train": True,  # 是否加载前一次的训练参数，接着训练
    "pretrained": False,
    "filter_by_anchor": True,
    "max_box_num": 35,  # 一幅图上最多有多少个目标
    "num_epochs": 80,
    "train_batch_size": 4,  # 对于完整 yolov3，每一批的训练样本不能太多，内存会炸掉
    "use_gpu": False,
    "yolo_block_cfg": {
        "input_size": [3, 320, 320],    # c, h, w
        "stages": [[8, 3], [16, 3], [16, 4], [32, 4], [64, 2], [128, 1]],   # 每一层的 channel 数和循环次数
        "anchors": [8, 8, 16, 16, 32, 32, 64, 64],      # 每个 anchor 的宽高，w, h 为一组
        "anchor_mask": [[3], [2], [1], [0]],                  # 每一层选哪些 anchor，不同层视野不同，选不同anchor, 只能选1个！
        "save_model_dir": "../resources/models/yolo-block",
        "pretrained_model_dir": "../resources/models/pretrained-yolo-block",
        "freeze_dir": "../resources/models/freeze-yolo-block"
    },
    "ignore_thresh": 0.7,
    "valid_thresh": 0.5,
    "nms_thresh": 0.45,
    "nms_top_k": 400,
    "nms_pos_k": 100,
    "mean_rgb": [127.5, 127.5, 127.5],
    "mode": "train",
    "multi_data_reader_count": 4,
    "box_disturbance_range_x": 6,         # 文字框每个坐标的随机扰动范围，x方向
    "box_disturbance_range_y": 4,         # 文字框每个坐标的随机扰动范围，y方向
    "image_distort_strategy": {
        "expand_prob": 0.5,
        "expand_max_ratio": 4,
        "hue_prob": 0.5,
        "hue_delta": 18,
        "contrast_prob": 0.5,
        "contrast_delta": 0.5,
        "saturation_prob": 0.5,
        "saturation_delta": 0.5,
        "brightness_prob": 0.5,
        "brightness_delta": 0.125,
        "rotate_prob": 0.4,
        "rotate_degree": 20
    },
    "rms_strategy": {
        "learning_rate": 0.01,
        "lr_epochs": [0.3, 0.5, 0.7, 0.9],
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.02]
    }
}


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量
    :return:
    """
    if train_parameters['current_mode'] != "train":
        return
    if train_parameters['image_count'] == -1:
        train_files_list = os.listdir(train_parameters['train_data_dir'])
        train_parameters['image_count'] = int(len(train_files_list) / 2)

    logger.info("train config use_gpu:{}".format(train_parameters['use_gpu']))
    logger.info("train config train_batch_size:{}".format(train_parameters['train_batch_size']))
    logger.info("train config num_epochs:{}".format(train_parameters['num_epochs']))
    logger.info("train config train_data_dir:{}".format(train_parameters['train_data_dir']))
    logger.info("train config image_count:{}".format(train_parameters['image_count']))
    logger.info("train config max_box_num:{}".format(train_parameters['max_box_num']))
    logger.info("train config continue_train:{}".format(train_parameters['continue_train']))
    logger.info("train config pretrained:{}".format(train_parameters['pretrained']))
    logger.info("train config eval_data_dir:{}".format(train_parameters['eval_data_dir']))

