# -*- coding: UTF-8 -*-
"""
按照yolo的网络结构拆分进入的 bbox
"""
import numpy as np
import math
from utils import logger
from utils import distance_of_pp


def get_prior_box(image_size, down_ratio, gt_block):
    """
    根据真实的block获取先验框
    :param image_size: 图片大小，此时是训练时resize之后的大小
    :param down_ratio: 下采样率，也即是现在边长缩小为原来的几分之一了
    :param gt_block: 真实框的位置，block都是近似正方形。坐标为相对比例
    :return:
    """

    block_x, block_y = gt_block[0] * image_size[1], gt_block[1] * image_size[0]
    block_w, block_h = gt_block[2] * image_size[1], gt_block[3] * image_size[0]
    center_x, center_y = round(block_x + block_w / 2), round(block_y + block_h / 2)
    prior_x, prior_y = math.floor(center_x / down_ratio), math.floor(center_y / down_ratio)
    return prior_x, prior_y


def split_by_anchors(gt_box, image_size, down_ratio, yolo_anchors):
    """
    将 ground truth 的外接矩形框分割成一个一个小块，类似 seg-link 中的做法
    :param gt_box: 真实外接矩形框，按照 [lt_x, lt_y, rt_x, rt_y, rd_x, rd_y, ld_x, ld_y] 排布的二维 list，
                    第一维是batch，实际的值都是除以了原始图片尺寸的比例值
    :param image_size: 训练图片的尺寸，[h, w]
    :param down_ratio: int 类型，下采样比例，也暗示现在的特征图被分成多大
    :param yolo_anchors: 当前批次的anchors
    :return:
    """

    gt_box = np.array(gt_box)
    image_size = np.array(image_size)
    down_ratio = np.array(down_ratio)[0]
    yolo_anchors = np.array(yolo_anchors)
    max_tolerant_ratio = 1.95
    min_tolerant_ratio = 0.95
    ret_shift_box = np.zeros(gt_box.shape, gt_box.dtype)
    max_bbox = 0

    for n in range(gt_box.shape[0]):
        current_index = 0
        for i in range(gt_box.shape[1]):
            left_height = distance_of_pp(gt_box[n][i][0:0 + 2], gt_box[n][i][6:6 + 2])
            up_width = distance_of_pp(gt_box[n][i][0:0 + 2], gt_box[n][i][2:2 + 2])
            left_height *= image_size[0]
            up_width *= image_size[1]
            bbox_h = left_height if left_height <= up_width else up_width
            if bbox_h <= 0.0001:
                break
            h_d_s = bbox_h / down_ratio
            s_d_h = down_ratio / bbox_h
            if max_tolerant_ratio >= h_d_s > min_tolerant_ratio:
                ret_shift_box[n, current_index] = gt_box[n, i]
                current_index += 1
                if current_index > max_bbox:
                    max_bbox = current_index

    logger.debug("filter done, down_ratio:{} feature map size:{}".format(down_ratio, image_size / down_ratio))
    return [ret_shift_box]
