# -*- coding: UTF-8 -*-
"""
模仿 seg-link 的形式开发的 loss
"""
import numpy as np
from utils import logger
from utils import distance_of_pp

CONF_RATIO = 1.0
NON_CONF_RATIO = 0.2
LOC_RATIO = 1.0
LINK_RATIO = 1.0
MIN_FLOAT = 10 ** -6
MAX_FLOAT = 20
epsilon = 10 ** -7


def sigmoid(x):
    if isinstance(x, np.ndarray):
        x[x > MAX_FLOAT] = MAX_FLOAT
        x[x < -MAX_FLOAT] = -MAX_FLOAT
    ret = 1.0 / (1.0 + np.e ** (-x))
    if isinstance(ret, np.ndarray):
        ret[ret < MIN_FLOAT] = MIN_FLOAT
    return ret


def sigmoid_grad(x):
    sigmoid_value = sigmoid(x)
    return sigmoid_value * (1.0 - sigmoid_value)


def safe_log(x):
    if isinstance(x, np.ndarray):
        if len(x[x < MIN_FLOAT]) > 0:
            logger.warning("in safe log, x <= 0, x: {}".format(x))
        x[x < MIN_FLOAT] = MIN_FLOAT
    elif isinstance(x, int) or isinstance(x, float):
        if x <= 0:
            logger.warning("in safe log, x <= 0, x: {}".format(x))
        if x < MIN_FLOAT:
            x = MIN_FLOAT
    return np.log(x + epsilon)


def smooth_l1_loss(pred, gt):
    """
    smooth L1 损失函数，损失函数公式为：
        loss = |x| - 0.5;  if |x| >= 1
        loss = 0.5*x^2;    if |x| < 1
    :param pred:
    :param gt:
    :return:
    """
    diff = pred - gt
    loss0 = np.abs(diff[np.abs(diff) >= 1.0]) - 0.5
    loss1 = np.square(diff[np.abs(diff) < 1.0]) * 0.5
    return np.sum(loss0) + np.sum(loss1)


def smooth_l1_loss_grad(pred, gt):
    """
    smooth L1 损失函数的梯度
    :param pred:
    :param gt:
    :return:
    """
    grad = np.zeros(pred.shape)
    diff = pred - gt
    grad[diff >= 1.0] = 1.0
    grad[diff <= -1.0] = -1.0
    grad[np.abs(diff) < 1.0] = diff[np.abs(diff) < 1.0]
    return grad


def get_gt_block(gt_box, image_size):
    return [int(gt_box[0] * image_size[0]), int(gt_box[1] * image_size[1]),
            int(gt_box[2] * image_size[0]), int(gt_box[3] * image_size[1])]


def get_point_grid(gt_block, down_ratio):
    """
    获取某个点在那个晶格内
    :param gt_block: [c_x, c_y, w, h] 基于训练图尺寸
    :param down_ratio: 缩小比例，也是晶格的宽度
    :return:
    """
    x = gt_block[0]
    y = gt_block[1]
    return int(x // down_ratio), int(y // down_ratio)


def get_pred_box(gt_block, i, j, down_ratio, anchor):
    """
    从真实框获取在训练图片上的bbox，有一个编码坐标的过程
    :param gt_block: 真实框
    :param i: 在当前特征图上的横坐标
    :param j: 在当前特征图上的纵坐标
    :param down_ratio: 下采样比例，特征图边长相对于训练图边长的比
    :param anchor: 当前点的anchor，[w, h]
    :return: 返回 yolo 的预测框，这个框是 [conf, c_x, c_y, w, h] 训练图中的大小
    """

    c_x = gt_block[0] / down_ratio - i
    c_y = gt_block[1] / down_ratio - j
    w = safe_log(gt_block[2] / anchor[0])
    h = safe_log(gt_block[3] / anchor[1])
    return np.array([c_x, c_y, w, h])


def get_iou(gt_block, pred_box):
    """
    获取两个 bbox 的 iou，两个 box 布局都是 [c_x, c_y, w, h]
    :param gt_block:
    :param pred_box:
    :return:
    """
    inter_left = max(gt_block[0] - gt_block[2] / 2, pred_box[0] - pred_box[2] / 2)
    inter_top = max(gt_block[1] - gt_block[3] / 2, pred_box[1] - pred_box[3] / 2)
    inter_right = min(gt_block[0] + gt_block[2] / 2, pred_box[0] + pred_box[2] / 2)
    inter_down = min(gt_block[1] + gt_block[3] / 2, pred_box[1] + pred_box[3] / 2)

    if inter_left >= inter_right or inter_top >= inter_down:
        return 0.0
    else:
        inter_area = (inter_right - inter_left) * (inter_down - inter_top)
        gt_area = gt_block[2] * gt_block[3]
        pred_area = pred_box[2] * pred_box[3]
        return 1.0 * inter_area / (pred_area + gt_area - inter_area)


def split_filed_to_block(gt_quad, image_size, down_ratio):
    """
    将一个字段的四边形切分成一系列的 block，移动的步长应该是和 down_ratio 一致，目的是采用密集采样
    :param gt_quad: 四边形布局，比例值，[lt_x, lt_y, rt_x, rt_y, rd_x, rd_y, ld_x, ld_y]
    :param image_size: 训练图大小 [h, w]
    :param down_ratio: 下采样比例
    :return: 返回 [c_x, c_y, w, h] 的block数组，此处的值为训练图片上大小的值
    """
    block_list = []
    ratio_limit = 1.8
    # 四个顶点表示的四边形
    quad = np.array([gt_quad[0:2], gt_quad[2:4], gt_quad[4:6], gt_quad[6:8]])
    # 转化成训练图上的大小
    quad[:, 0] = quad[:, 0] * image_size[1]
    quad[:, 1] = quad[:, 1] * image_size[0]
    l_h = distance_of_pp(quad[0], quad[3])
    u_w = distance_of_pp(quad[0], quad[1])

    if u_w >= l_h * ratio_limit:
        # 宽比高长不少，是横条型四边形，block的高需要差值计算
        r_h = distance_of_pp(quad[1], quad[2])
        trend_line_l_p = np.array([(quad[0][0] + quad[3][0]) / 2, (quad[0][1] + quad[3][1]) / 2])
        trend_line_r_p = np.array([(quad[1][0] + quad[2][0]) / 2, (quad[1][1] + quad[2][1]) / 2])
        trend_line_d = distance_of_pp(trend_line_l_p, trend_line_r_p)
        sin = (trend_line_r_p[1] - trend_line_l_p[1]) / trend_line_d
        cos = (trend_line_r_p[0] - trend_line_l_p[0]) / trend_line_d
        split_count = int(trend_line_d / (down_ratio / cos) + 0.5)
        split_count = split_count if split_count > 0 else 1
        step = trend_line_d / split_count
        for i in range(split_count):
            step_ratio = step * (0.5 + i) / trend_line_d
            step_ratio = 1 if step_ratio > 1 else step_ratio
            real_h = round((1 - step_ratio) * l_h + step_ratio * r_h)
            c_x = round(trend_line_l_p[0] + down_ratio * (0.5 + i))
            c_y = round(trend_line_l_p[1] + sin * step * (0.5 + i))
            real_w = real_h if c_x - real_h / 2 > 0 else down_ratio
            real_w = real_w if real_w < down_ratio else down_ratio
            block = [c_x, c_y, down_ratio, real_h]
            # 有时候计算每那么精确，防御型的避免一些多出来 block 的情况
            if block[0] + block[2] > image_size[1]:
                break
            block_list.append(block)
    elif l_h >= u_w * ratio_limit:
        # 高比宽敞不少，是竖条形四边形，block的高需要差值计算
        d_w = distance_of_pp(quad[2], quad[3])
        trend_line_t_p = np.array([(quad[0][0] + quad[1][0]) / 2, (quad[0][1] + quad[1][1]) / 2])
        trend_line_d_p = np.array([(quad[2][0] + quad[3][0]) / 2, (quad[2][1] + quad[3][1]) / 2])
        trend_line_d = distance_of_pp(trend_line_t_p, trend_line_d_p)
        sin = (trend_line_d_p[1] - trend_line_t_p[1]) / trend_line_d
        cos = (trend_line_d_p[0] - trend_line_t_p[0]) / trend_line_d
        if sin == 0.0:
            logger.debug("quad: {} trend_line_t_p:{} trend_line_d_p:{}".format(quad, trend_line_t_p, trend_line_d_p))
        split_count = int(trend_line_d / (down_ratio / sin) + 0.5)
        split_count = split_count if split_count > 0 else 1
        step = trend_line_d / split_count
        for i in range(split_count):
            step_ratio = step * (0.5 + i) / trend_line_d
            step_ratio = 1 if step_ratio > 1 else step_ratio
            real_w = round((1 - step_ratio) * u_w + step_ratio * d_w)
            c_x = round(trend_line_t_p[0] + cos * step * (0.5 + i))
            c_y = round(trend_line_t_p[1] + down_ratio * (0.5 + i))
            real_h = real_w if c_y - real_w / 2 > 0 else down_ratio
            real_h = real_h if real_h < down_ratio else down_ratio
            block = [c_x, c_y, real_w, down_ratio]
            # 有时候计算每那么精确，防御型的避免一些多出来 block 的情况
            if block[1] + block[3] > image_size[0]:
                break
            block_list.append(block)
    else:
        # 高和宽差不多，是正方形式四边形
        block = [
            quad[0][0] if quad[0][0] < quad[3][0] else quad[3][0],
            quad[0][1] if quad[0][1] < quad[1][1] else quad[1][1],
            quad[2][0] if quad[2][0] > quad[1][0] else quad[1][0],
            quad[2][1] if quad[2][1] > quad[3][1] else quad[3][1]
        ]
        w = block[2] - block[0]
        h = block[3] - block[1]
        block[0] += round(w / 2)
        block[1] += round(h / 2)
        block[2] = w
        block[3] = h
        block_list.append(block)

    return np.array(block_list)


def get_link_by_grid_loc(current_grid_loc, next_grid_loc):
    """
    获取当前晶格和下一个晶格的 link 值。link 排布为：
    [left_top, mid_top, right_top, right_mid, right_down, mid_down, left_down, left_mid]
    :param current_grid_loc: 晶格在特征图上的坐标，(x, y)
    :param next_grid_loc: 晶格在特征图上的坐标，(x, y)
    :return: 当前晶格的 link，下一个晶格的 link
    """
    current_link = -1
    if current_grid_loc[0] == next_grid_loc[0]:
        # 上下
        if current_grid_loc[1] + 1 == next_grid_loc[1]:
            current_link = 5
        elif current_grid_loc[1] - 1 == next_grid_loc[1]:
            current_link = 1
    elif current_grid_loc[1] == next_grid_loc[1]:
        # 左右
        if current_grid_loc[0] + 1 == next_grid_loc[0]:
            current_link = 3
        elif current_grid_loc[1] - 1 == next_grid_loc[1]:
            current_link = 7
    elif current_grid_loc[0] + 1 == next_grid_loc[0]:
        # 右侧上下角
        if current_grid_loc[1] - 1 == next_grid_loc[1]:
            current_link = 2
        elif current_grid_loc[1] + 1 == next_grid_loc[1]:
            current_link = 4
    elif current_grid_loc[0] - 1 == next_grid_loc[0]:
        # 左侧上下角
        if current_grid_loc[1] - 1 == next_grid_loc[1]:
            current_link = 0
        elif current_grid_loc[1] + 1 == next_grid_loc[1]:
            current_link = 6
    return current_link, (current_link + 4) % 8


def get_link_loss(pred_link, gt_link):
    """
    计算 link 的损失函数，对每一个方向的link计算交叉熵，0为没有，1为有
    :param pred_link:
    :param gt_link:
    :return:
    """
    # loss = 0.0
    # for pp, gp in zip(pred_link, gt_link):
    #     loss += -((1 - gp) * np.log(1 - pp) + gp * np.log(pp))
    # 需要先经过 sigmoid 一次才能最为概率
    # logger.info(pred_link)
    pred_link = sigmoid(pred_link)
    loss = (1 - gt_link) * safe_log(1 - pred_link) + gt_link * safe_log(pred_link)
    # logger.info("pred link{} gt link:{} loss:{}".format(pred_link, gt_link, -loss))
    return pred_link, np.sum(-loss)


def get_link_loss_grad(pred, gt_link):
    """
    计算 link 的梯度，梯度的公式为 (x - gt)/(x * (1 - x))，此处 x = sigmoid(pred_link)
    :param pred_link:
    :param gt_link:
    :return:
    """
    pred_link_grad = sigmoid_grad(pred)
    pred_link = sigmoid(pred)
    grad = (pred_link - gt_link) / (pred_link * (1 - pred_link)) * pred_link_grad
    # logger.info("pred:{} pred link{} gt link:{} grad:{}".format(pred, pred_link, gt_link, grad))
    return grad


def seg_loss(gt_block, pred_grid, grid_x, grid_y, down_ratio, yolo_anchors):
    """
    计算一个匹配上预测框的晶格的 loss。包含两个部分：
    1、每个 anchor 的坐标损失，坐标损失还包括中心点坐标损失 + 宽高长度损失
    2、每个 anchor 的置信度损失，1是target，0是background
    :param gt_block: [c_x, c_y, w, h] 训练图中的大小
    :param pred_grid: [score, c_x, c_y, w, h]，重复 anchor 个数次
    :param grid_x: 晶格的横坐标
    :param grid_y: 晶格的纵坐标
    :param down_ratio: 下采样率
    :param yolo_anchors: 当前的一组 anchor，排列为 [a1_w, a1_h, a2_w, a2_h, ...]
    :return:
    """
    conf_loss = 0.0
    location_loss = 0.0
    for mask_idx in range(0, len(yolo_anchors), 2):
        # 计算网格和字段的对应关系
        anchor = yolo_anchors[mask_idx:mask_idx + 2]
        anchor_idx = int(mask_idx / 2)
        pred_offset = anchor_idx * 5
        box = pred_grid[pred_offset:pred_offset + 5]
        conf = sigmoid(box[0])
        gt_box = get_pred_box(gt_block, grid_x, grid_y, down_ratio, anchor)
        current_conf_loss = -safe_log(conf)
        logger.debug("anchor:{} down_ratio:{} grid_x:{} grid_y:{} pred conf:{} gt conf:1.0 conf loss:{}"
                     .format(anchor, down_ratio, grid_x, grid_y, conf, current_conf_loss))
        conf_loss = conf_loss + current_conf_loss  # conf loss
        current_location_loss = smooth_l1_loss(box[1:], gt_box)
        logger.debug("anchor:{} down_ratio:{} grid_x:{} grid_y:{} pred loc:{} gt loc:{} loc loss:{}"
                     .format(anchor, down_ratio, grid_x, grid_y, box[1:], gt_box, current_location_loss))
        location_loss = location_loss + current_location_loss  # location loss

    logger.debug("seg loss, conf part:{} conf_ratio:{}, loc_ratio:{} location part:{}"
                 .format(conf_loss, CONF_RATIO, location_loss, LOC_RATIO))
    return CONF_RATIO * conf_loss + LOC_RATIO * location_loss


def seg_loss_grad(gt_block, pred_grid, grid_x, grid_y, down_ratio, yolo_anchors):
    """
    计算 seg 部分的梯度
    :param gt_block: [c_x, c_y, w, h] 训练图中的大小
    :param pred_grid: [score, c_x, c_y, w, h]，重复 anchor 个数次
    :param grid_x: 晶格的横坐标
    :param grid_y: 晶格的纵坐标
    :param down_ratio: 下采样率
    :param yolo_anchors: 当前的一组 anchor，排列为 [a1_w, a1_h, a2_w, a2_h, ...]
    :return:
    """
    grad = np.zeros(pred_grid.shape)
    for mask_idx in range(0, len(yolo_anchors), 2):
        # 计算网格和字段的对应关系
        anchor = yolo_anchors[mask_idx:mask_idx + 2]
        anchor_idx = int(mask_idx / 2)
        offset = int(anchor_idx * 5)
        box = pred_grid[offset:offset + 5]
        conf = sigmoid(box[0])
        gt_box = get_pred_box(gt_block, grid_x, grid_y, down_ratio, anchor)
        # conf loss grad
        conf_grad = CONF_RATIO * sigmoid_grad(box[0]) / (-conf)
        logger.debug("anchor:{} down_ratio:{} grid_x:{} grid_y:{} pred value:{} pred conf:{} grad:{}"
                     .format(anchor, down_ratio, grid_x, grid_y, box[0], conf, conf_grad))
        grad[offset:offset + 1] = conf_grad
        # location loss grad
        grad[offset + 1:offset + 5] = LOC_RATIO * smooth_l1_loss_grad(box[1:], gt_box)
        logger.debug("anchor:{} down_ratio:{} grid_x:{} grid_y:{} pred box:{} gt_box:{} grad:{}"
                     .format(anchor, down_ratio, grid_x, grid_y, box[1:], gt_box, grad[offset + 1:offset + 5]))
    return grad


def detect_loss(pred, gt_quad, image_size, down_ratio, yolo_anchors):
    """
    将 ground truth 的外接四边形框分割成一个一个小块，类似 seg-link 中的做法。
    在计算 loss 的时候，先拿所有的 prior box 去对应一个分割出来的 gt_block，把对应上的全部找到之后，相邻的 prior box 就应该相连，
    此时才知道 link 的 ground truth。对于文字来讲，只可能出现最多两个方向的链接
    :param pred: 预测值，数据排列为：[n, c, h, w]
        每一个像素点的预测值 c 为两部分，第一部分是得分和坐标，形如：[score, c_x, c_y, w, h]，重复 anchor 个数次
        第二部分是否有 link 的概率，8个方向，形如：
        [left_top, mid_top, right_top, right_mid, right_down, mid_down, left_down, left_mid]
        [\, |, /, ->, \, |, /, <-]
        第一维：batch_size大小的训练样本数组
    :param gt_quad: 真实外接四边形框，按照 [lt_x, lt_y, rt_x, rt_y, rd_x, rd_y, ld_x, ld_y] 最为最小元素，值为除以了原始图片尺寸的比例。
        第一维：batch_size大小的训练样本数组
        第二维：每张图片的字段数组，设定不超过80个字段吧，简单起见
    :param image_size: 训练图片的尺寸，[h, w]
    :param down_ratio: int 类型，下采样比例，也暗示现在的特征图被分成多大
    :param yolo_anchors: 当前批次的anchors，[a1_w, a1_h, a2_w, a2_h, ...]
    :return: batch 维度的 loss
    """
    pred = np.array(pred)
    pred = pred.transpose([0, 2, 3, 1])
    gt_quad = np.array(gt_quad)
    image_size = np.array(image_size)
    down_ratio = np.array(down_ratio)[0]
    yolo_anchors = np.array(yolo_anchors)
    loss = np.zeros(gt_quad.shape[0], np.float32)
    logger.debug("receive pred shape:{} gt_quad shape:{}".format(pred.shape, gt_quad.shape))

    grid_loss = 0.0
    link_loss = 0.0
    non_matched_loss = 0.0

    batch_size, field_num, _ = gt_quad.shape
    batch_size, grid_h, grid_w, pred_channel = pred.shape
    grid_matched_mask = np.zeros([batch_size, grid_h, grid_w], np.int32)  # 记录每一个晶格和 ground truth 的匹配情况
    grid_matched_block = np.zeros([batch_size, grid_h, grid_w, 4], np.int32)  # 记录每一个晶格对应框坐标的 ground truth
    grid_matched_link = np.zeros([batch_size, grid_h, grid_w, 8], np.float32)  # 记录每一个晶格对应链接的 ground truth

    for n in range(batch_size):
        for f in range(field_num):
            current_field_quard = gt_quad[n][f]
            # 判断当前是否已经都处理完了，遇到空值跳过
            if current_field_quard[0] == current_field_quard[4] and current_field_quard[1] == current_field_quard[5]:
                break
            # 在此处分割出真正的 gt_block，并建立 link 信息
            block_list = split_filed_to_block(current_field_quard, image_size, down_ratio)
            # logger.info("block_list:{}".format(block_list))
            link_grid_list = []
            for gt_block in block_list:
                grid_x, grid_y = get_point_grid(gt_block, down_ratio)
                link_grid_list.append((grid_x, grid_y))  # 将所有一个字段的晶格存起来，用于后面计算 link loss
                # 计算这个晶格的 loss，两个部分，一个是 conf loss，sigmod 的交叉熵
                # 一个是 location loss，c_x, c_y, w, h 使用 L1-smooth
                pred_grid = pred[n][grid_y][grid_x]
                grid_matched_mask[n][grid_y][grid_x] = 1  # 记录文字框的匹配情况
                grid_matched_block[n][grid_y][grid_x] = gt_block
                grid_loss += seg_loss(gt_block, pred_grid, grid_x, grid_y, down_ratio, yolo_anchors)

            # 计算 link loss，link部分对于 pred_box 是成对 link，这是个强约束；
            # 并且对于第一个和最后一个晶格，最多只有一个 link，如果只有一个 block，那么没有 link
            previous_link = -1
            next_link = -1
            for idx, current_grid_loc in enumerate(link_grid_list):
                gt_link = np.zeros(8, np.float32)
                if idx == len(link_grid_list) - 1:
                    # 对于最后一个，最多是单边，而且是向前的 link，此时已经在计算前一个 grid 的 link 时计算出来了
                    pass
                else:
                    # 需要知道下一个晶格的坐标，才能确定 gt_link，所以最多遍历到倒数第二个
                    next_grid_loc = link_grid_list[idx + 1]
                    current_link, next_link = get_link_by_grid_loc(current_grid_loc, next_grid_loc)
                    gt_link[current_link] = 1.0
                if previous_link != -1:
                    gt_link[previous_link] = 1.0

                # 记录计算出来的真实链接情况
                grid_matched_link[n][current_grid_loc[1]][current_grid_loc[0]] = gt_link
                previous_link = next_link
                grid = pred[n][current_grid_loc[1]][current_grid_loc[0]]
                pred_link, current_link_loss = get_link_loss(grid[-8:], gt_link)
                logger.debug("in detect_loss, gt link:{}, link pred:{} link loss:{}"
                             .format(gt_link, pred_link, current_link_loss))
                link_loss += current_link_loss

        # 负样本计算 conf loss
        non_matched_pred = pred[n][grid_matched_mask[n] == 0]
        for anchor_mask_idx in range(0, len(yolo_anchors), 2):  # 每有一个 anchor 就有一组 conf
            anchor_idx = int(anchor_mask_idx / 2)
            pred_conf = sigmoid(non_matched_pred[:, anchor_idx * 5])
            current_non_matched_loss = np.sum(-safe_log(1 - pred_conf))
            logger.debug("negative conut:{} conf:{} loss:{}"
                         .format(len(non_matched_pred), pred_conf, current_non_matched_loss))
            non_matched_loss += current_non_matched_loss  # 1是target，0是background

        negative_count = len(non_matched_pred)
        negative_count = negative_count if negative_count != 0 else 1
        positive_count = grid_h * grid_w - negative_count
        positive_count = positive_count if positive_count != 0 else 1
        # negative_count = 1
        # positive_count = 1
        logger.debug("negative_count:{} positive_count:{}".format(negative_count, positive_count))
        loss[n] = grid_loss / positive_count + link_loss + NON_CONF_RATIO * non_matched_loss / negative_count
        logger.debug("in detect_loss, current loss:{}, grid loss:{} link loss:{} non_matched_loss:{}"
                     .format(loss[n], grid_loss, link_loss, non_matched_loss))

    return loss, grid_matched_mask, grid_matched_block, grid_matched_link


def detect_loss_grad(pred, gt_quad, image_size, down_ratio, yolo_anchors,
                     loss, grid_matched_mask, grid_matched_block, grid_matched_link,
                     loss_dy, grid_matched_mask_dy, grid_matched_block_dy, grid_matched_link_dy):
    """
    计算loss的梯度，输入参数为所有的前向输入，所有前向输出，以及所有前向输出的梯度
    :param forward_input: 正向计算时候的输入，pred, gt_quad, image_size, down_ratio, yolo_anchors
    :param forward_output: 正向计算时候的输出，loss, grid_matched_mask, gt_grid_link
        loss: [batch, ]
        grid_matched_mask: [batch_size, grid_h, grid_w]
        gt_grid_link: [batch_size, grid_h, grid_w, 8]
    :param dy: 前一步的梯度，因为这是反向的第一步，所以理论上应该是 1，维度应该和batch_size一致
    :return: 返回各个参数的梯度值
    """

    pred = np.array(pred)
    pred = pred.transpose([0, 2, 3, 1])
    gt_quad = np.array(gt_quad)
    image_size = np.array(image_size)
    down_ratio = np.array(down_ratio)
    yolo_anchors = np.array(yolo_anchors)

    loss = np.array(loss)
    grid_matched_mask = np.array(grid_matched_mask)
    grid_matched_block = np.array(grid_matched_block)
    grid_matched_link = np.array(grid_matched_link)
    batch_size, grid_h, grid_w = grid_matched_mask.shape

    pred_grad = np.zeros(shape=pred.shape, dtype=pred.dtype)
    gt_quad_grad = np.ones(shape=gt_quad.shape, dtype=np.float32)
    image_size_grad = np.ones(shape=image_size.shape, dtype=np.float32)
    down_ratio_grad = np.ones(shape=down_ratio.shape, dtype=np.float32)
    yolo_anchors_grad = np.ones(shape=yolo_anchors.shape, dtype=np.float32)

    down_ratio = down_ratio[0]
    logger.debug("in detect_loss_grad")
    for n in range(batch_size):
        for y in range(grid_h):
            for x in range(grid_w):
                if grid_matched_mask[n][y][x] == 1:
                    # 文字框匹配上了 ,下面开始计算各个部分的梯度
                    # seg 部分的梯度
                    pred_grid = pred[n][y][x]
                    gt_block = grid_matched_block[n][y][x]
                    seg_grad = seg_loss_grad(gt_block, pred_grid, x, y, down_ratio, yolo_anchors)
                    pred_grad[n][y][x][0:len(pred_grid)] = seg_grad

                    # link 部分的梯度
                    gt_link = grid_matched_link[n][y][x]
                    pred_grad[n][y][x][-8:] = get_link_loss_grad(pred_grid[-8:], gt_link)
                else:
                    # 文字框没匹配上，此时只有一个 conf loss
                    non_matched_pred = pred[n][y][x]
                    for anchor_mask_idx in range(0, len(yolo_anchors), 2):  # 每有一个 anchor 就有一组 conf
                        anchor_idx = int(anchor_mask_idx / 2)
                        non_conf = sigmoid(non_matched_pred[anchor_idx * 5])
                        non_conf_grad = sigmoid_grad(non_matched_pred[anchor_idx * 5])
                        # e = np.e ** (-non_matched_pred[anchor_idx * 5])
                        non_conf_grad = 1 / (1 - non_conf) * non_conf_grad
                        logger.debug("grid_x:{} grid_y:{} pred value:{} non matched conf:{} grad:{} ratio:{}"
                                     .format(x, y, non_matched_pred[anchor_idx * 5], non_conf, non_conf_grad,
                                             NON_CONF_RATIO))
                        pred_grad[n][y][x][anchor_idx * 5] = NON_CONF_RATIO * non_conf_grad

    logger.debug("in detect_loss_grad, grad shape:{}".format(pred_grad.shape))
    return pred_grad.transpose([0, 3, 1, 2]), gt_quad_grad, image_size_grad, down_ratio_grad, yolo_anchors_grad
