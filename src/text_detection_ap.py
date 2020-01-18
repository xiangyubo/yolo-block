# -*- coding: UTF-8 -*-
"""
评判函数，用于计算最后结果的 AP
"""
import os
import shutil
import codecs
import json
import random
from math import ceil
from collections import namedtuple

GtBox = namedtuple('GtBox', ['left', 'top', 'right', 'down'])


def box_iou_xyxy(box1, box2):
    """
    计算两个box的iou，box的格式为[xmin, ymin, xmax, ymax]
    :param box1:
    :param box2:
    :return:
    """
    inter_x1 = max(box1[0], box2[0])
    inter_x2 = min(box1[2], box2[2])
    inter_y1 = max(box1[1], box2[1])
    inter_y2 = min(box1[3], box2[3])

    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1

    if inter_h <= 0 or inter_w <= 0:
        inter_area = 0
        inter_box = [0, 0, 0, 0]
    else:
        inter_area = inter_w * inter_h
        inter_box = [inter_x1, inter_y1, inter_x2, inter_y2]
    b1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    b2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    return inter_box, inter_area


def detection_one_pr(gt_boxes, pred_boxes, iou_threshold=0.7):
    """
    用于计算最后的预测框和真实值之间的精确率和召回率，用于评价检测结果
    :param gt_boxes: 真实框， list 其中的每个元素为[xmin, ymin, xmax, ymax]
    :param pred_boxes: 预测框，需要经过nms，list 其中的每个元素为[xmin, ymin, xmax, ymax]
    :param iou_threshold: 交并比的阈值，用于判定一个 pred_box 是否是真实的正例
    :return: 计算出来的精确率和召回率
    """
    tp = 0
    if len(pred_boxes) == 0:
        return 0.0, 0.0
    gt_box_iou_dict = {}

    for pred_box in pred_boxes:
        pred_box_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        hit =False
        for gt_box in gt_boxes:
            if str(gt_box) not in gt_box_iou_dict:
                gt_box_iou_dict[str(gt_box)] = {'box': GtBox(gt_box[0], gt_box[1], gt_box[2], gt_box[3]), 'list': []}
            inter_box, inter_area = box_iou_xyxy(pred_box, gt_box)
            if inter_area / pred_box_area >= iou_threshold:
                hit = True
                gt_box_iou_dict[str(gt_box)]['list'].append(inter_box)
        if hit:
            tp += 1

    selected_gt_box = 0
    total_area = 0
    cover_area = 0
    for (gt_box, box_list) in gt_box_iou_dict.items():
        # todo...现在先暴力认为就是全部的box的最大外围边界

        min_left, min_top = 1000000, 1000000
        max_right, max_down = 0, 0
        box = box_list['box']
        gt_box_area = (box[2] - box[0]) * (box[3] - box[1])
        total_area += gt_box_area
        inter_box_list = box_list['list']
        for inter_box in inter_box_list:
            if inter_box[0] < min_left:
                min_left = inter_box[0]
            if inter_box[1] < min_top:
                min_top = inter_box[1]
            if inter_box[2] > max_right:
                max_right = inter_box[2]
            if inter_box[3] > max_down:
                max_down = inter_box[3]
        if len(inter_box_list) > 0:
            union_area = (max_right - min_left) * (max_down - min_top)
            cover_area += union_area
            if 1.0 * union_area / gt_box_area >= iou_threshold:
                selected_gt_box += 1

    return 1.0 * tp / len(pred_boxes), 1.0 * selected_gt_box / len(gt_boxes), 1.0 * cover_area / total_area


# def detection_apr(gt_boxes, pred_boxes, iou_threshold=0.5):

