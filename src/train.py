# -*- coding: UTF-8 -*-
"""
训练常基于YOLOv3-tiny的网络，文字检测
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import logger
from split_by_anchors import split_by_anchors
from reader import custom_reader
from yolo_block_model import YOLOv3Block
from train_config import init_train_parameters
from train_config import train_parameters
from yolo_block_loss import detect_loss, detect_loss_grad

import os

os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = '0.82'
import numpy as np
import time
import math
import paddle
import paddle.fluid as fluid


def multi_process_custom_reader(data_dir, num_workers, input_size, mode):
    all_file = os.listdir(data_dir)
    images = [file for file in all_file if file.endswith('.jpg') or file.endswith('.png')]
    images.sort()
    annotations = [file for file in all_file if file.endswith('.txt') or file.endswith('.xml')]
    annotations.sort()
    image_annotation_list = list(zip(images, annotations))

    readers = []
    n = int(math.ceil(len(image_annotation_list) // num_workers))
    image_annotation_parts = [image_annotation_list[i: i + n] for i in range(0, len(image_annotation_list), n)]
    for image_annotation_part in image_annotation_parts:
        reader = custom_reader(image_annotation_part, data_dir, input_size, mode)
        reader = paddle.reader.shuffle(reader, train_parameters['train_batch_size'])
        reader = paddle.batch(reader, train_parameters['train_batch_size'])
        readers.append(reader)
    return paddle.reader.multiprocess_reader(readers, False)


def single_custom_reader(data_dir, input_size, mode):
    """
    读取一个目录下面的所有数据，对于目标检测类的任务来说，通常是一张图和一个标注文件对应，文件名除后缀外相同
    :param data_dir:
    :param input_size:
    :param mode:
    :return:
    """
    all_file = os.listdir(data_dir)
    images = [file for file in all_file if file.endswith('.jpg') or file.endswith('.png')]
    images.sort()
    annotations = [file for file in all_file if file.endswith('.txt') or file.endswith('.xml')]
    annotations.sort()
    image_annotation_list = list(zip(images, annotations))
    reader = custom_reader(image_annotation_list, data_dir, input_size, mode)
    reader = paddle.reader.shuffle(reader, train_parameters['train_batch_size'])
    reader = paddle.batch(reader, train_parameters['train_batch_size'])
    return reader


def optimizer_sgd_setting():
    num_epochs = train_parameters["num_epochs"]
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    iters = 1 if iters < 1 else iters
    learning_strategy = train_parameters['sgd_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [int(i * iters * num_epochs) for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    logger.info("origin learning rate: {} boundaries: {}  values: {}".format(lr, boundaries, values))

    optimizer = fluid.optimizer.RMSPropOptimizer(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values),
        regularization=fluid.regularizer.L2Decay(0.00005))

    return optimizer


def build_program_with_async_reader(main_prog, startup_prog):
    max_box_num = train_parameters['max_box_num']
    yolo_config = train_parameters['yolo_block_cfg']
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 8], dtype='float32')
        data_reader = fluid.layers.create_py_reader_by_data(capacity=train_parameters['train_batch_size'],
                                                            feed_list=[img, gt_box],
                                                            name='train')
        multi_reader = multi_process_custom_reader(train_parameters['train_list'],
                                                   train_parameters['data_dir'],
                                                   train_parameters['multi_data_reader_count'],
                                                   yolo_config['input_size'],
                                                   'train')
        data_reader.decorate_paddle_reader(multi_reader)
        img, gt_box = fluid.layers.read_file(data_reader)
        return data_reader, get_loss(img, gt_box, main_prog)


def build_program_with_feeder(main_prog, startup_prog, place):
    max_box_num = train_parameters['max_box_num']
    yolo_config = train_parameters['yolo_block_cfg']
    with fluid.program_guard(main_prog, startup_prog):
        img = fluid.layers.data(name='img', shape=yolo_config['input_size'], dtype='float32')
        gt_box = fluid.layers.data(name='gt_box', shape=[max_box_num, 8], dtype='float32')
        feeder = fluid.DataFeeder(feed_list=[img, gt_box], place=place, program=main_prog)
        reader = single_custom_reader(train_parameters['train_data_dir'],
                                      yolo_config['input_size'], 'train')
        return feeder, reader, get_loss(img, gt_box, main_prog)


def create_tmp_var(program, name, dtype, shape, stop_gradient=False):
    return program.current_block().create_var(name=name, dtype=dtype, shape=shape, stop_gradient=stop_gradient)


def get_loss(img, gt_box, program):
    yolo_config = train_parameters['yolo_block_cfg']
    with fluid.unique_name.guard():
        model = YOLOv3Block(yolo_config['anchors'], yolo_config['anchor_mask'])
        outputs = model.net(img)
        losses = []
        down_sample_ratio = model.get_downsample_ratio()
        with fluid.unique_name.guard('train'):
            train_image_size_tensor = fluid.layers.assign(np.array(yolo_config['input_size'][1:]))
            for i, out in enumerate(outputs):
                down_ratio = fluid.layers.fill_constant(shape=[1], value=down_sample_ratio, dtype="int32")
                yolo_anchors = fluid.layers.assign(np.array(model.get_yolo_anchors()[i]))
                filter_bbox = create_tmp_var(program, None, gt_box.dtype, gt_box.shape, True)
                fluid.layers.py_func(func=split_by_anchors,
                                     x=[gt_box, train_image_size_tensor, down_ratio, yolo_anchors],
                                     out=[filter_bbox])

                n, c, h, w = out.shape[0], out.shape[1], out.shape[2], out.shape[3]
                tmp_loss = create_tmp_var(program, None, np.float32, [n])
                grid_matched_mask = create_tmp_var(program, None, np.int32, [n, h, w], True)
                grid_matched_block = create_tmp_var(program, None, np.int32, [n, h, w, 4], True)
                grid_matched_link = create_tmp_var(program, None, np.float32, [n, h, w, 8], True)
                logger.info("current out shape:{}".format(out.shape))
                tmp_loss, _, _, _ = fluid.layers.py_func(func=detect_loss,
                                                         x=[out, filter_bbox, train_image_size_tensor, down_ratio,
                                                            yolo_anchors],
                                                         out=[tmp_loss, grid_matched_mask, grid_matched_block,
                                                              grid_matched_link],
                                                         backward_func=detect_loss_grad)
                losses.append(fluid.layers.reduce_mean(tmp_loss))
                down_sample_ratio //= 2
            loss = sum(losses)
            optimizer = optimizer_sgd_setting()
            optimizer.minimize(loss)
            return loss


def load_params(exe, program):
    yolo_config = train_parameters['yolo_block_cfg']
    if train_parameters['continue_train'] and os.path.exists(yolo_config['save_model_dir']):
        logger.info('load param from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=yolo_config['save_model_dir'],
                                   main_program=program)
    elif train_parameters['pretrained'] and os.path.exists(yolo_config['pretrained_model_dir']):
        logger.info('load param from pretrained model')

        def if_exist(var):
            return os.path.exists(os.path.join(yolo_config['pretrained_model_dir'], var.name))

        fluid.io.load_vars(exe, yolo_config['pretrained_model_dir'], main_program=program,
                           predicate=if_exist)


def train():
    logger.info("start train YOLOv3-block")
    train_parameters["current_mode"] = "train"
    init_train_parameters()
    yolo_config = train_parameters['yolo_block_cfg']
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()

    logger.info("build network and program")
    train_program = fluid.Program()
    start_program = fluid.Program()
    eval_program = fluid.Program()
    start_program = fluid.Program()
    # train_reader, loss = build_program_with_async_reader(train_program, start_program)
    feeder, reader, loss = build_program_with_feeder(train_program, start_program, place)
    # eval_program = eval_program.clone(for_test=True)

    logger.info("loss name:{}".format(loss.name))
    logger.info("build executor and init params")
    exe = fluid.Executor(place)
    exe.run(start_program)
    train_fetch_list = [loss.name]
    load_params(exe, train_program)

    total_batch_count = 0
    current_best_loss = 10000000.0
    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: %d, start read image", pass_id)
        pass_mean_loss = 0.0
        pass_batch_count = 0
        for batch_id, data in enumerate(reader()):
            try:
                logger.info("start pass {}, batch {}".format(pass_id, batch_id))
                t1 = time.time()
                loss = exe.run(train_program,
                               feed=feeder.feed(data),
                               fetch_list=train_fetch_list)
                logger.info("end pass {}, train batch {} exe.run".format(pass_id, batch_id + 1))
                period = time.time() - t1
                temp_loss = np.array(loss)
                loss = np.mean(temp_loss)

                count = 0
                too_small = np.where(temp_loss < 0.0, 1, 0)
                too_big = np.where(temp_loss > 100000.0, 1, 0)
                count += np.sum(too_big)
                count += np.sum(too_small)
                if count == 0:
                    pass_mean_loss += np.clip(temp_loss, 0.0, None)
                    pass_batch_count += 1
                batch_id += 1
                total_batch_count += 1

                if batch_id % 1 == 0:
                    logger.info("end pass {}, batch {}, loss {} time {}"
                                .format(pass_id, batch_id, loss, "%2.2f sec" % period))
            except Exception as e:
                logger.error("train exception: {}".format(e))

        # 每训练完成一个批次，保存一次
        pass_loss = np.sum(pass_mean_loss) / pass_batch_count
        logger.info("pass {} train result, current pass mean loss: {}".format(pass_id, pass_loss))
        if pass_loss < current_best_loss:
            logger.info("temp save pass {} train result, current best pass mean loss: {}".format(pass_id, pass_loss))
            fluid.io.save_persistables(dirname=yolo_config['save_model_dir'], main_program=train_program,
                                       executor=exe)
            current_best_loss = pass_loss

    logger.info("training till last epoch, end training YOLOv3-block")


if __name__ == '__main__':
    train()
