# -*- coding: UTF-8 -*-
"""
一些常用的工具函数，比如日志
"""
import os
import logging
import functools
import math
from collections import namedtuple

import numpy as np
from PIL import Image

MIN_FLOAT = 0.000001
MAX_FLOAT = 20

logger = None


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_path = os.path.join(os.path.dirname(os.getcwd()), 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_name = os.path.join(log_path, 'train.log')
        sh = logging.StreamHandler()
        fh = logging.FileHandler(log_name, mode='a')
        fh.setLevel(logging.INFO)
        sh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.addHandler(fh)


def box_iou_xyxy(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


def rescale_box_in_input_image(boxes, im_shape, input_size):
    """Scale (x1, x2, y1, y2) box of yolo output to input image"""
    h, w = im_shape
    # max_dim = max(h , w)
    # boxes = boxes * max_dim / input_size
    # dim_diff = np.abs(h - w)
    # pad = dim_diff // 2
    # if h <= w:
    #     boxes[:, 1] -= pad
    #     boxes[:, 3] -= pad
    # else:
    #     boxes[:, 0] -= pad
    #     boxes[:, 2] -= pad
    fx = w / input_size
    fy = h / input_size
    boxes[:, 0] *= fx
    boxes[:, 1] *= fy
    boxes[:, 2] *= fx
    boxes[:, 3] *= fy
    boxes[boxes < 0] = 0
    boxes[:, 2][boxes[:, 2] > (w - 1)] = w - 1
    boxes[:, 3][boxes[:, 3] > (h - 1)] = h - 1
    return boxes


def calc_nms_box_new(pred_boxes, nms_thresh=0.4):
    output_boxes = np.empty((0, 5))
    output_scores = np.empty(0)
    output_labels = np.empty(0)

    pred_boxes = np.array(pred_boxes)
    print(pred_boxes.shape)
    print(pred_boxes)
    pred_boxes = pred_boxes.reshape((-1, 5))
    print(pred_boxes.shape)
    pred_boxes = pred_boxes[(-pred_boxes)[:, 0].argsort()]

    detect_boxes = []
    detect_scores = []
    detect_labels = []
    while pred_boxes.shape[0]:
        detect_boxes.append(pred_boxes[0])
        if pred_boxes.shape[0] == 1:
            break
        iou = box_iou_xyxy(np.array([detect_boxes[-1][1:]]), pred_boxes[1:, 1:5])
        pred_boxes = pred_boxes[1:][iou < nms_thresh]

    output_boxes = np.append(output_boxes, detect_boxes, axis=0)
    output_scores = np.append(output_scores, detect_scores)
    output_labels = np.append(output_labels, detect_labels)

    return (output_boxes, output_scores, output_labels)


def distance_of_pp(p0, p1):
    return np.sqrt(np.sum(np.square(p0 - p1)))


def resize_img(img, input_size):
    img = img.resize((input_size[1], input_size[2]), Image.BILINEAR)
    return img


def sigmoid(x):
    if isinstance(x, np.ndarray):
        x[x > MAX_FLOAT] = MAX_FLOAT
        x[x < -MAX_FLOAT] = -MAX_FLOAT
    ret = 1.0 / (1.0 + np.e ** (-x))
    if isinstance(ret, np.ndarray):
        ret[ret < MIN_FLOAT] = MIN_FLOAT
    return ret


def box_xywh_to_xyxy(box):
    shape = box.shape
    assert shape[-1] == 4, "Box shape[-1] should be 4."

    box = box.reshape((-1, 4))
    box[:, 0], box[:, 2] = box[:, 0] - box[:, 2] / 2, box[:, 0] + box[:, 2] / 2
    box[:, 1], box[:, 3] = box[:, 1] - box[:, 3] / 2, box[:, 1] + box[:, 3] / 2
    box = box.reshape(shape)
    return box


def get_neighbour(x, y, link_idx):
    if link_idx == 0:
        return [x - 1, y - 1]
    elif link_idx == 1:
        return [x, y - 1]
    elif link_idx == 2:
        return [x + 1, y - 1]
    elif link_idx == 3:
        return [x + 1, y]
    elif link_idx == 4:
        return [x + 1, y + 1]
    elif link_idx == 5:
        return [x, y + 1]
    elif link_idx == 6:
        return [x - 1, y + 1]
    elif link_idx == 7:
        return [x - 1, y]


def get_all_yolo_pred(outputs, yolo_anchors, target_size, input_shape, valid_thresh=0.5, link_thresh=0.6):
    """
    转化预测结果为预测图上的矩形框和连接
    需要分层的 BFS 构造出每一层的联通格子
    :param outputs: 预测的结果，和 anchor 层数一样的层数，每一层包含了 batch_size 的相同层预测结果
    :param yolo_anchors: [anchor_w0, anchor_h0, anchor_w1, anchor_h1]
    :param target_size:训练图的尺寸大小 [c, h, w]
    :param input_shape: 预测图的尺寸[h, w]
    :param valid_thresh: 根据置信度做过滤
    :return: 返回一组二维的点-链接图
    """
    all_pred = []
    for output, anchors in zip(outputs, yolo_anchors):
        pred = get_yolo_detection(output, anchors, target_size, input_shape, valid_thresh, link_thresh)
        all_pred.extend(pred)

    return all_pred


def get_yolo_detection(preds, anchors, target_size, img_shape, valid_thresh, link_thresh):
    """
    计算 yolo-block 的预测结果，组装成应该有的 bbox 和 link
    :param preds: 一层预测图，选出对应的合格点，有 batch_size 个同层结果
    :param anchors:
    :param target_size: 训练图的 [c, h, w]
    :param img_shape: 预测图的尺寸 [h, w]
    :return:
    """
    preds = np.array(preds)
    n, c, h, w = preds.shape
    print("current n:{} c:{} h:{} w:{} valid_thresh:{}".format(n, c, h, w, valid_thresh))
    width_down_sample_ratio = img_shape[1] / w
    height_down_sample_ratio = img_shape[0] / h
    anchor_num = int(len(anchors) // 2)
    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    preds = preds.transpose((0, 2, 3, 1)).reshape([n, h, w, anchor_num * 5 + 8])
    preds[:, :, :, -8:] = sigmoid(preds[:, :, :, -8:])

    ret_boxes = []
    for i in range(anchor_num):
        preds[:, :, :, i * 5] = sigmoid(preds[:, :, :, i * 5])
        preds[:, :, :, i * 5 + 1] += grid_x
        preds[:, :, :, i * 5 + 1] *= width_down_sample_ratio
        preds[:, :, :, i * 5 + 2] += grid_y
        preds[:, :, :, i * 5 + 2] *= height_down_sample_ratio
        preds[:, :, :, i * 5 + 3] = np.exp(preds[:, :, :, i * 5 + 3]) * anchors[0] * img_shape[1] / target_size[2]
        preds[:, :, :, i * 5 + 4] = np.exp(preds[:, :, :, i * 5 + 4]) * anchors[1] * img_shape[0] / target_size[1]
        preds[:, :, :, 5 * i + 1: 5 * i + 5] = box_xywh_to_xyxy(preds[:, :, :, 5 * i + 1: 5 * i + 5])

        # logger.info("anchors:{} preds box:{}".format(anchors, preds[:, :, :, i * 5:i * 5 + 5]))
        # 抽取合格的点，进行 BFS
        valid_map = preds[:, :, :, i * 5] >= valid_thresh
        # print(preds[:, :, :, i * 5])
        # print(valid_map)
        # ret_box = merge_box(valid_map, preds, link_thresh)
        n, h, w = valid_map.shape
        ret_box = []
        for batch_idx in range(n):
            y_idx_dict = {}
            current_box = []
            circular_list = []
            for y in range(h):
                for x in range(w):
                    valid = valid_map[batch_idx, y, x]
                    if valid:
                        pred = preds[batch_idx, y, x]
                        ret_box.append(pred)

        if len(ret_box) > 0:
            ret_boxes.extend(ret_box)

    return ret_boxes


def merge_box(valid_map, preds, link_thresh):
    """
    将所有的box当作图来进行联通子图合并
    """
    Circular = namedtuple("Circular", ['x', 'y', 'links', 'bbox', 'conf'])

    ret_box = []
    n, h, w = valid_map.shape

    for batch_idx in range(n):
        y_idx_dict = {}
        current_box = []
        circular_list = []
        for y in range(h):
            for x in range(w):
                valid = valid_map[batch_idx, y, x]
                if not valid:
                    continue
                pred = preds[batch_idx, y, x]
                conf = pred[0]
                box = tuple(pred[1:5])
                links = tuple(pred[5:5 + 8])
                c = Circular(x, y, links, box, conf)
                circular_list.append(c)
                if y in y_idx_dict:
                    x_dict = y_idx_dict[y]
                else:
                    x_dict = {}
                    y_idx_dict[y] = x_dict
                x_dict[x] = len(circular_list) - 1

        # logger.info(y_idx_dict)
        # 广度优先遍历，找出所有的联通子图
        circular_count = len(circular_list)
        sub_graph_list = []
        visit_queue = []
        visit_set = set()

        for i in range(circular_count):
            if i not in visit_set:
                current_sub_graph = set()
                visit_queue.append(i)
                while True:
                    if len(visit_queue) == 0:
                        break
                    e = visit_queue.pop(0)
                    visit_set.add(e)
                    current_sub_graph.add(e)
                    circular = circular_list[e]
                    links = circular.links
                    # logger.info("e:{} circle:{}".format(e, circular))
                    for idx, link in enumerate(links):
                        if link >= link_thresh:
                            neighbour_idx = None
                            x, y = get_neighbour(circular.x, circular.y, idx)
                            if y in y_idx_dict:
                                if x in y_idx_dict[y]:
                                    neighbour_idx = y_idx_dict[y][x]
                            if neighbour_idx is not None \
                                    and neighbour_idx not in current_sub_graph \
                                    and neighbour_idx not in visit_set and neighbour_idx not in visit_queue:
                                # logger.info("idx:{} x:{} y:{} neighbour_idx:{} append to visit queue, queue:{}, visit_set:{}"
                                #       .format(idx, x, y, neighbour_idx, visit_queue, visit_set))
                                visit_queue.append(neighbour_idx)
                    if len(visit_queue) == 0:
                        break
                sub_graph_list.append(current_sub_graph)

        for sub_graph in sub_graph_list:
            xmin, ymin = 1000000, 1000000
            xmax, ymax = 0, 0
            mean_conf = 0.0
            for idx in sub_graph:
                c = circular_list[idx]
                mean_conf += c.conf
                if c.bbox[0] < xmin:
                    xmin = c.bbox[0]
                if c.bbox[1] < ymin:
                    ymin = c.bbox[1]
                if c.bbox[2] > xmax:
                    xmax = c.bbox[2]
                if c.bbox[3] > ymax:
                    ymax = c.bbox[3]
            mean_conf /= len(sub_graph)
            current_box.append([mean_conf, xmin, ymin, xmax, ymax])
        if len(current_box) > 0:
            ret_box.extend(current_box)
    return ret_box


init_log_config()
