# -*- coding: UTF-8 -*-
"""
将训练时候的参数固化
"""
import paddle
import paddle.fluid as fluid
import codecs

from train_config import train_parameters
from yolo_block_model import YOLOv3Block


def freeze_model():

    exe = fluid.Executor(fluid.CPUPlace())
    yolo_config = train_parameters['yolo_block_cfg']
    path = yolo_config['save_model_dir']
    model = YOLOv3Block(yolo_config['anchors'], yolo_config['anchor_mask'])
    image = fluid.layers.data(name='image', shape=yolo_config['input_size'], dtype='float32')
    outputs = model.net(image)

    freeze_program = fluid.default_main_program()
    fluid.io.load_persistables(exe, path, freeze_program)
    freeze_program = freeze_program.clone(for_test=True)
    print("freeze from:{} out: {}, pred layout: {}".format(path, yolo_config['freeze_dir'], outputs))
    fluid.io.save_inference_model(yolo_config['freeze_dir'], ['image'], outputs, exe, freeze_program)


if __name__ == '__main__':
    freeze_model()
