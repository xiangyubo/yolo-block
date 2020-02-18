# -*- coding: UTF-8 -*-
"""
一些常用的工具函数，比如日志
"""
import os
import logging
import predict_grid
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形

import numpy as np
from PIL import Image

MIN_FLOAT = 0.000001
MAX_FLOAT = 20

WORD_END = 0
WORD_MID = 1

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
        fh = logging.FileHandler(log_name, mode='w')
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


def box_iou_quad(quad1, quad2):
    """
    计算两个四边形的 iou，每个四边形的形式为 x1, y1, x2, y2, x3, y3, x4, y4
    :param quad1: numpy type array
    :param quad2: numpy type array
    :return:
    """
    assert quad1.shape[-1] == 8, "quad1 shape[-1] should be 8."
    assert quad2.shape[-1] == 8, "quad2 shape[-1] should be 8."

    ret_iou = np.zeros(len(quad2))

    quad_line1 = np.array(quad1).reshape((-1, 4, 2))
    quad_line1 = quad_line1[0]
    poly1 = Polygon(quad_line1).convex_hull
    quad_line2 = np.array(quad2).reshape((-1, 4, 2))
    for idx, tmp_quad_line in enumerate(quad_line2):
        poly2 = Polygon(tmp_quad_line).convex_hull
        if not poly1.intersects(poly2):  # 如果两四边形不相交
            ret_iou[idx] = 0.0
        elif poly1.contains(poly2):
            ret_iou[idx] = 1.0
        else:
            try:
                inter_area = poly1.intersection(poly2).area  # 相交面积
                union_poly = np.concatenate((quad_line1, tmp_quad_line))  # 合并两个box坐标，变为8*2
                # union_area = poly1.area + poly2.area - inter_area
                union_area = MultiPoint(union_poly).convex_hull.area
                if union_area == 0:
                    ret_iou[idx] = 0.0
                # iou = float(inter_area) / (union_area-inter_area)  #错了
                ret_iou[idx] = float(inter_area) / union_area
                # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
                # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
                # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
            except shapely.geos.TopologicalError:
                logger.warning('shapely.geos.TopologicalError occured, iou set to 0')
                ret_iou[idx] = 0.0

    return ret_iou


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
    """
    对预测框做非极大值抑制
    :param pred_boxes: 有n个batch，每个batch对应一个数组的预测框，conf, x1, y1, x2, y2, x3, y3, x4, y4
    :param nms_thresh:
    :return:
    """
    ret_pred_box = []

    for pb_pred_boxes in pred_boxes:

        pb_pred_boxes = np.array(pb_pred_boxes)
        pb_pred_boxes = pb_pred_boxes.reshape((-1, 9))
        pb_pred_boxes = pb_pred_boxes[(-pb_pred_boxes)[:, 0].argsort()]

        detect_boxes = []
        while pb_pred_boxes.shape[0]:
            detect_boxes.append(pb_pred_boxes[0])
            if pb_pred_boxes.shape[0] == 1:
                break
            iou = box_iou_quad(np.array([detect_boxes[-1][1:]]), pb_pred_boxes[1:, 1:])
            pb_pred_boxes = pb_pred_boxes[1:][iou < nms_thresh]

        ret_pred_box.append(detect_boxes)

    return ret_pred_box


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
    """
    以当前节点的坐标 x, y 和节点的 link 下标出发，给出这个 link 关联的邻居节点坐标
    :param x:
    :param y:
    :param link_idx:
    :return:
    """
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


def get_link_idx(c_x, c_y, n_x, n_y):
    """
    以 c_x, c_y 节点为中心出发，找到邻居坐标为 x, y 的 link 下标
    :param c_x:
    :param c_y:
    :param n_x:
    :param n_y:
    :return:
    """
    if n_x < c_x and n_y < c_y:
        return 0
    elif n_x == c_x and n_y < c_y:
        return 1
    elif n_x > c_x and n_y < c_y:
        return 2
    elif n_x > c_x and n_y == c_y:
        return 3
    elif n_x > c_x and n_y > c_y:
        return 4
    elif n_x == c_x and n_y > c_y:
        return 5
    elif n_x < c_x and n_y > c_y:
        return 6
    elif n_x < c_x and n_y == c_y:
        return 7


def get_all_yolo_pred(outputs, yolo_anchors, target_size, input_shape, valid_thresh=0.5, link_thresh=0.6):
    """
    转化预测结果为预测图上的矩形框和连接
    需要分层的 BFS 构造出每一层的联通格子
    :param outputs: 预测的结果，和 anchor 层数一样的层数，每一层包含了 batch_size 的相同层预测结果
    :param yolo_anchors: [anchor_w0, anchor_h0, anchor_w1, anchor_h1]
    :param target_size:训练图的尺寸大小 [c, h, w]
    :param input_shape: 预测图的尺寸[h, w]
    :param valid_thresh: 根据置信度做过滤
    :param link_thresh: 对于连接的置信度过滤
    :return: 返回一组二维的点-链接图
    """

    temp = outputs[0]
    preds = np.array(temp)
    n, c, h, w = preds.shape
    all_pred = [[] for i in range(n)]

    for output, anchors in zip(outputs, yolo_anchors):
        pred = get_yolo_detection(output, anchors, target_size, input_shape, valid_thresh, link_thresh)
        all_pred.extend(pred)
        for batch_idx, per_batch_ret in enumerate(pred):
            batch_ret = all_pred[batch_idx]
            batch_ret.extend(per_batch_ret)

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
    logger.debug("current n:{} c:{} h:{} w:{} valid_thresh:{}".format(n, c, h, w, valid_thresh))
    width_down_sample_ratio = img_shape[1] / w
    height_down_sample_ratio = img_shape[0] / h
    anchor_num = int(len(anchors) // 2)
    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    preds = preds.transpose((0, 2, 3, 1)).reshape([n, h, w, anchor_num * 5 + 8])
    preds[:, :, :, -8:] = sigmoid(preds[:, :, :, -8:])

    ret_boxes = [[] for i in range(n)]
    for i in range(anchor_num):
        preds[:, :, :, i * 5] = sigmoid(preds[:, :, :, i * 5])
        preds[:, :, :, i * 5 + 1] += grid_x
        preds[:, :, :, i * 5 + 1] *= width_down_sample_ratio
        preds[:, :, :, i * 5 + 2] += grid_y
        preds[:, :, :, i * 5 + 2] *= height_down_sample_ratio
        preds[:, :, :, i * 5 + 3] = np.exp(preds[:, :, :, i * 5 + 3]) * anchors[0] * img_shape[1] / target_size[2]
        preds[:, :, :, i * 5 + 4] = np.exp(preds[:, :, :, i * 5 + 4]) * anchors[1] * img_shape[0] / target_size[1]
        preds[:, :, :, 5 * i + 1: 5 * i + 5] = box_xywh_to_xyxy(preds[:, :, :, 5 * i + 1: 5 * i + 5])

        # 抽取合格的点，进行 BFS
        valid_map = preds[:, :, :, i * 5] >= valid_thresh
        ret_box = merge_box(valid_map, preds, link_thresh)
        for batch_idx, per_batch_ret in enumerate(ret_box):
            batch_ret = ret_boxes[batch_idx]
            batch_ret.extend(per_batch_ret)

    return ret_boxes


def merge_box(valid_map, preds, link_thresh):
    """
    将所有的box当作图来进行联通子图合并
    """

    ret_box = []
    n, h, w = valid_map.shape

    for batch_idx in range(n):
        y_idx_dict = {}
        current_box = []
        circular_list = []
        # 先选出所有可能的点，组装成list，和可以快速访问的map
        for y in range(h):
            for x in range(w):
                valid = valid_map[batch_idx, y, x]
                if not valid:
                    continue
                pred = preds[batch_idx, y, x]
                conf = pred[0]
                box = pred[1:5]
                links = pred[5:5 + 8]
                logger.debug("a available circle, conf:{:.2f} x:{} y:{} links:{} bbox:{}"
                            .format(conf, x, y, links, box))
                c = predict_grid.Grid(x, y, links, box, conf, WORD_MID)
                circular_list.append(c)
                if y in y_idx_dict:
                    x_dict = y_idx_dict[y]
                else:
                    x_dict = {}
                    y_idx_dict[y] = x_dict
                x_dict[x] = len(circular_list) - 1

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
                    # logger.debug("e:{} circle:{}".format(e, circular))
                    for idx, link in enumerate(links):
                        if link >= link_thresh:
                            neighbour_idx = None
                            x, y = get_neighbour(circular.x, circular.y, idx)
                            if y in y_idx_dict and x in y_idx_dict[y]:
                                neighbour_idx = y_idx_dict[y][x]
                            if neighbour_idx is not None:
                                # 存在有效的邻居节点，此时才能证明这个 link 可能是有效的
                                # 是否真的有效，还要看邻居的意思
                                neighbour_c = circular_list[neighbour_idx]
                                n_l_idx = get_link_idx(neighbour_c.x, neighbour_c.y, circular.x, circular.y)
                                origin_n_l = neighbour_c.links[n_l_idx]
                                # 邻居想跟自己连接，才连接，不强求
                                links[idx] = 0.0 if origin_n_l < link_thresh else link
                                if neighbour_idx not in current_sub_graph and neighbour_idx not in visit_set \
                                        and neighbour_idx not in visit_queue and links[idx] > 0.0000001:
                                    # 此时这个邻居节点还是未访问过的，并且想和自己相连
                                    # 查看一下此时邻居节点是否有边和自己相连，如果不相连，还需要修改邻居节点的 links
                                    visit_queue.append(neighbour_idx)
                            else:
                                # 虽然想和邻居节点相连，但是并不存在有效的邻居节点，所以这个 link 无效
                                links[idx] = 0.0

                    # 此时和所有的邻居都验证完毕了，计算节点是否为端点
                    links[links >= link_thresh] = 1.0
                    links[links < link_thresh] = 0.0
                    has_link = np.where(links >= 0.9, 1, 0)
                    link_count = np.sum(has_link)
                    circular.end_type = WORD_END if link_count <= 1 else WORD_MID
                    if len(visit_queue) == 0:
                        break
                sub_graph_list.append(current_sub_graph)

        # 为组合在一起的 bbox 构建外接矩形框
        for sub_graph in sub_graph_list:
            x1, y1, x2, y2, x3, y3, x4, y4 = 0, 0, 0, 0, 0, 0, 0, 0
            mean_conf = 0.0
            end_c_0 = None
            end_c_1 = None
            logger.debug("start a new sub graph")
            for idx in sub_graph:
                c = circular_list[idx]
                conf = c.conf
                x = c.x
                y = c.y
                box = c.bbox
                links = c.links
                end_type = c.end_type
                logger.debug("in sub graph circle, x:{} y:{} end type:{} links:{}".format(x, y, end_type, links))
                mean_conf += c.conf
                if c.end_type == WORD_END:
                    if end_c_0 is None:
                        end_c_0 = c
                    elif end_c_1 is None:
                        end_c_1 = c
            if end_c_1 is None:
                x1, y1 = end_c_0.bbox[0], end_c_0.bbox[1]
                x2, y2 = end_c_0.bbox[2], end_c_0.bbox[1]
                x3, y3 = end_c_0.bbox[2], end_c_0.bbox[3]
                x4, y4 = end_c_0.bbox[0], end_c_0.bbox[3]
            else:
                if end_c_0.x < end_c_1.x:
                    x1, y1 = end_c_0.bbox[0], end_c_0.bbox[1]
                    x2, y2 = end_c_1.bbox[2], end_c_1.bbox[1]
                    x3, y3 = end_c_1.bbox[2], end_c_1.bbox[3]
                    x4, y4 = end_c_0.bbox[0], end_c_0.bbox[3]
                elif end_c_0.x > end_c_1.x:
                    x1, y1 = end_c_1.bbox[0], end_c_1.bbox[1]
                    x2, y2 = end_c_0.bbox[2], end_c_0.bbox[1]
                    x3, y3 = end_c_0.bbox[2], end_c_0.bbox[3]
                    x4, y4 = end_c_1.bbox[0], end_c_1.bbox[3]
                else:
                    if end_c_0.y < end_c_1.y:
                        x1, y1 = end_c_0.bbox[0], end_c_0.bbox[1]
                        x2, y2 = end_c_0.bbox[2], end_c_0.bbox[1]
                        x3, y3 = end_c_1.bbox[2], end_c_1.bbox[3]
                        x4, y4 = end_c_1.bbox[0], end_c_1.bbox[3]
                    else:
                        x1, y1 = end_c_1.bbox[0], end_c_1.bbox[1]
                        x2, y2 = end_c_1.bbox[2], end_c_1.bbox[1]
                        x3, y3 = end_c_0.bbox[2], end_c_0.bbox[3]
                        x4, y4 = end_c_0.bbox[0], end_c_0.bbox[3]
            mean_conf /= len(sub_graph)
            current_box.append([mean_conf, x1, y1, x2, y2, x3, y3, x4, y4])

        # 按照每个 batch 一个输出
        ret_box.append(current_box)
    return ret_box


init_log_config()
