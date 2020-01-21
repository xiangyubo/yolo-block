# -*- coding: UTF-8 -*-
"""
使用模型进行推理
"""
import codecs
import sys
import numpy as np
import time
import paddle
import paddle.fluid as fluid
import math
import os
import functools
import utils
from utils import logger
from utils import resize_img
from PIL import Image
from PIL import ImageDraw
from train_config import train_parameters
from collections import namedtuple


yolo_config = train_parameters['yolo_block_cfg']

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']
yolo_anchors = []
for mask_pair in anchor_mask:
    mask_anchors = []
    for mask in mask_pair:
        mask_anchors.append(anchors[2 * mask])
        mask_anchors.append(anchors[2 * mask + 1])
    yolo_anchors.append(mask_anchors)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = yolo_config['freeze_dir']
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)


def draw_bbox_image(img, pred_boxes, save_name):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    draw = ImageDraw.Draw(img)
    for box in pred_boxes:
        # for box in layers:
        conf, xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3], box[4]
        print("pred text box, conf:{} bbox:{}".format(conf, [xmin, ymin, xmax, ymax]))
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red')

    img.save(save_name)


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = Image.open(img_path)
    img = resize_img(origin, target_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img


def infer(image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    origin, tensor_img = read_image(image_path)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    print("image shape high:{}, width:{}".format(input_h, input_w))
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = time.time() - t1
    print("predict cost time:{}".format("%2.2f sec" % period))
    # for layer in batch_outputs:
    #     logger.info(np.array(layer))

    pred_boxes = utils.get_all_yolo_pred(batch_outputs, yolo_anchors,
                                         target_size, image_shape, train_parameters['valid_thresh'])
    print(len(pred_boxes))
    # print(pred_boxes)
    # pred_boxes = utils.calc_nms_box_new(pred_boxes, train_parameters['nms_thresh'])

    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-result.jpg'
    draw_bbox_image(origin, pred_boxes, out_path)


if __name__ == '__main__':
    image_name = sys.argv[1]
    print(os.getcwd())
    image_path = image_name
    infer(image_path)
