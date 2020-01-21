# -*- coding: UTF-8 -*-
"""
数据读取器，定义读取一张图片的部分，针对不同数据集，需要自定义修改
"""
import traceback

import numpy as np
import random
import os
import codecs

from PIL import ImageEnhance
from PIL import Image
from utils import logger
from utils import resize_img
from train_config import train_parameters


def box_iou_xywh(box1, box2):
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1 + 1
    inter_h = inter_y2 - inter_y1 + 1
    inter_w[inter_w < 0] = 0
    inter_h[inter_h < 0] = 0

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area)


def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes *= np.expand_dims(mask.astype('float32'), axis=1)
    labels *= mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] + boxes[:, 2]) / 2 / w, (boxes[:, 2] - boxes[:, 0]) / w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] + boxes[:, 3]) / 2 / h, (boxes[:, 3] - boxes[:, 1]) / h

    return boxes, labels, mask.sum()


def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    if random.random() > 0.6:
        return img, boxes, labels
    if len(boxes) == 0:
        return img, boxes, labels

    if not constraints:
        constraints = [
            (0.3, 1.0),
            (0.5, 1.0),
            (0.7, 1.0),
            (0.9, 1.0)
        ]

    w, h = img.size
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[
                (crop_x + crop_w / 2.0) / w,
                (crop_y + crop_h / 2.0) / h,
                crop_w / float(w),
                crop_h / float(h)
            ]])

            iou = box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3]))
        return img, crop_boxes, crop_labels
    return img, boxes, labels


def random_brightness(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_distort_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_distort_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_distort_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_distort_strategy']['hue_prob']:
        hue_delta = train_parameters['image_distort_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_image(img):
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob < 0.85:
        func_list = [random_brightness, random_contrast, random_saturation, random_hue]
        loop_count = len(func_list)
        for i in range(loop_count):
            idx = np.random.randint(0, len(func_list))
            img = func_list[idx](img)
            func_list.pop(idx)

    return img


def disturbance_box(box, im_width, im_height):
    """
    轻微扰动box[lt_x, lt_y, rt_x, rt_y, rd_x, rd_y, ld_x, ld_y]
    :param box: 当前图上的 bbox
    :param im_width: 原图的宽
    :param im_height: 原图的高
    :return:
    """
    if np.random.uniform(0, 1) > 0.6:
        return box
    x_range = train_parameters["box_disturbance_range_x"]
    y_range = train_parameters["box_disturbance_range_y"]
    for n in range(box.shape[0]):
        for i in range(4):
            disturbance_x = np.random.randint(-x_range, x_range)
            disturbance_y = np.random.randint(-y_range, y_range)
            box[n, i * 2 + 0] += 0.2 * disturbance_x / im_width
            box[n, i * 2 + 1] += 0.2 * disturbance_y / im_height

    return box


def preprocess(img, bbox_labels, input_size, mode):
    """
    数据预处理，如果是训练模型，还会做一些图像增强和标注的轻微扰动
    :param img:
    :param bbox_labels:
    :param input_size:
    :param mode:
    :return:
    """
    sample_labels = np.array(bbox_labels)
    if mode == 'train':
        img = distort_image(img)
        im_width, im_height = img.size
        sample_labels = disturbance_box(sample_labels, im_width, im_height)

    img = resize_img(img, input_size)
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843
    return img, sample_labels


def rotate_quad(quad):
    """
     旋转四个顶点的四边形，使得第1个点是左上角点
    :param quad:
    :return:
    """
    vertex0 = (quad[0], quad[1])
    vertex1 = (quad[4], quad[5])
    if vertex0[0] < vertex1[0] and vertex0[1] < vertex1[1]:
        return quad
    elif vertex0[0] > vertex1[0] and vertex0[1] < vertex1[1]:
        return [quad[6], quad[7], quad[0], quad[1], quad[2], quad[3], quad[4], quad[5]]
    elif vertex0[0] > vertex1[0] and vertex0[1] > vertex1[1]:
        return [quad[4], quad[5], quad[6], quad[7], quad[0], quad[1], quad[2], quad[3]]
    else:
        return [quad[2], quad[3], quad[4], quad[5], quad[6], quad[7], quad[0], quad[1]]


def custom_reader(image_annotation_list, data_dir, input_size, mode):
    """
    用户自定义的数据读取函数
    :param image_annotation_list: 图片文件名和标注文件名的 list
    :param data_dir: 图片和标注所在的目录
    :param input_size: 需要 resize 的目标图片大小 [c, h, w]
    :param mode: 读取模式，通产是针对是否需要对图像做增强
    :return:
    """
    max_box_num = train_parameters['max_box_num']

    def reader():
        np.random.shuffle(image_annotation_list)
        for image_path, annotation_path in image_annotation_list:
            logger.info("current deal {} {}".format(image_path, annotation_path))
            try:
                ######################  以下可能是需要自定义修改的部分   ############################
                if not os.path.exists(image_path):
                    image_path = os.path.join(data_dir, image_path)
                img = Image.open(image_path)
                if img is None:
                    logger.info("deal {} image, but get None".format(image_path))
                    continue
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                im_width, im_height = img.size
                if os.path.exists(os.path.join(data_dir, annotation_path)):
                    with codecs.open(os.path.join(data_dir, annotation_path), encoding='utf8') as f:
                        annotation_lines = f.readlines()
                else:
                    annotation_lines = []

                if len(annotation_lines) > max_box_num:
                    continue
                # bbox 的列表，每一个元素为这样
                # layout: lt_x, lt_y, rt_x, rt_y, rd_x, rd_y, ld_x, ld_y
                bbox_list = []
                for line in annotation_lines:
                    if len(line) <= 1:
                        continue
                    parts = line.split(',')
                    bbox_sample = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                                   float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                    bbox_sample = rotate_quad(bbox_sample)

                    bbox_sample[0] /= im_width
                    bbox_sample[2] /= im_width
                    bbox_sample[4] /= im_width
                    bbox_sample[6] /= im_width

                    bbox_sample[1] /= im_height
                    bbox_sample[3] /= im_height
                    bbox_sample[5] /= im_height
                    bbox_sample[7] /= im_height

                    bbox_list.append(bbox_sample)
                ######################  可能需要自定义修改部分结束   ############################

                img, bbox_list = preprocess(img, bbox_list, input_size, mode)
                boxes = bbox_list
                ret_boxes = np.zeros((max_box_num, 8), dtype=np.float32)
                cope_size = max_box_num if len(boxes) >= max_box_num else len(boxes)
                ret_boxes[0: cope_size] = boxes[0: cope_size]
                yield img, ret_boxes
            except Exception as e:
                logger.error("deal {} {} exception, {}".format(image_path, annotation_path, traceback.format_exc()))
                pass

    return reader
