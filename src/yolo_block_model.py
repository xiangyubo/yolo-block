# -*- coding: UTF-8 -*-
"""
训练常基于dark-net的YOLOv3网络结构
"""
import paddle
import paddle.fluid as fluid

from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from train_config import train_parameters


class YOLOv3Block(object):
    def __init__(self, anchors, anchor_mask):
        self.outputs = []
        self.downsample_ratio = 1
        self.anchor_mask = anchor_mask
        self.anchors = anchors

        self.yolo_anchors = []
        for mask_pair in self.anchor_mask:
            mask_anchors = []
            for mask in mask_pair:
                mask_anchors.append(self.anchors[2 * mask])
                mask_anchors.append(self.anchors[2 * mask + 1])
            self.yolo_anchors.append(mask_anchors)

    def name(self):
        return 'YOLOv3-block'

    def get_anchors(self):
        return self.anchors

    def get_anchor_mask(self):
        return self.anchor_mask

    def get_downsample_ratio(self):
        return self.downsample_ratio

    def get_yolo_anchors(self):
        return self.yolo_anchors

    def conv_bn(self,
                input,
                num_filters,
                filter_size,
                stride,
                padding,
                act="relu",
                num_groups=1,
                use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            groups=num_groups,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                # initializer=fluid.initializer.Normal(0., 0.02),
                initializer=fluid.initializer.MSRA(),      # 鲁棒性很好的初始化方式，能很快逼近好的训练趋势
                                 gradient_clip=fluid.clip.GradientClipByValue(max=5.0, min=-5.0)),
            bias_attr=False)

        # batch_norm中的参数不需要参与正则化，所以主动使用正则系数为0的正则项屏蔽掉
        out = fluid.layers.batch_norm(
            input=conv, 
            act=act,
            param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02), regularizer=L2Decay(0.)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))

        return out

    def depthwise_conv_bn(self, input, filter_size=3, stride=1, padding=1):
        num_filters = input.shape[1]
        return self.conv_bn(input,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            num_groups=num_filters)

    def downsample(self, input, pool_size=2, pool_stride=2):
        self.downsample_ratio *= 2
        return fluid.layers.pool2d(input=input, pool_type='max', pool_size=pool_size,
                                   pool_stride=pool_stride)

    def basicblock(self, input, num_filters, count):
        conv1 = self.conv_bn(input, num_filters, filter_size=3, stride=1, padding=1)
        for i in range(1, count):
            if i != count - 1:
                conv1 = self.conv_bn(conv1, num_filters, filter_size=3, stride=1, padding=1)
            else:
                conv1 = self.conv_bn(conv1, num_filters, filter_size=3, stride=1, padding=1)
        # conv1 = fluid.layers.concat([conv1, input], axis=1)
        out = self.downsample(conv1)
        return out

    def upsample(self, input, scale=2):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(input)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * scale
        out_shape.stop_gradient = True

        # resize by actual_shape
        out = fluid.layers.resize_nearest(
            input=input,
            scale=scale,
            actual_shape=out_shape)
        return out

    def yolo_detection_block(self, input, num_filters):
        route = self.conv_bn(input, num_filters, filter_size=1, stride=1, padding=0)
        tip = self.conv_bn(route, num_filters * 2, filter_size=3, stride=1, padding=1)
        return route, tip

    def net(self, img):
        # darknet-tiny
        # stages = [16, 32, 64, 128, 256, 512]
        stages = [[8, 2], [16, 2], [32, 4], [64, 4], [128, 2], [256, 1]]
        assert len(self.anchor_mask) <= len(stages), "anchor masks can't bigger than downsample times"
        # 256x256
        tmp = img
        blocks = []
        for i, stage_count in enumerate(stages):
            if i == len(stages) - 1:
                block = self.conv_bn(tmp, stage_count[0], filter_size=3, stride=1, padding=1)
                for j in range(1, stage_count[1]):
                    block = self.conv_bn(block, stage_count[0], filter_size=3, stride=1, padding=1)
                blocks.append(block)
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.depthwise_conv_bn(blocks[-1])
                block = self.conv_bn(blocks[-1], stage_count[0] * 2, filter_size=1, stride=1, padding=0)
                blocks.append(block)
            else:
                tmp = self.basicblock(tmp, stage_count[0], stage_count[1])
                blocks.append(tmp)

        blocks = [blocks[-1], blocks[3], blocks[2], blocks[1]]

        # yolo detector
        for i, block in enumerate(blocks):
            # yolo 中跨视域链接
            if i > 0:
                block = fluid.layers.concat(input=[route, block], axis=1)
            if i == 0:
                route, tip = self.yolo_detection_block(block, num_filters=256 // (2 ** i))
            else:
                tip = self.conv_bn(block, num_filters=256, filter_size=3, stride=1, padding=1)
            # 5 个元素代表 score|c_x|c_y|w|h|，循环anchor次
            # 8 个元素代表 左上，中上，右上，右中，右下，中下，左下，左中
            block_out = fluid.layers.conv2d(
                input=tip,
                num_filters=len(self.anchor_mask[i]) * 5 + 8,
                filter_size=1,
                stride=1,
                padding=0,
                param_attr=ParamAttr(initializer=fluid.initializer.Normal(0., 0.02)),
                bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0), regularizer=L2Decay(0.)))
            self.outputs.append(block_out)
            # 为了跨视域链接，差值方式提升特征图尺寸
            if i < len(blocks) - 1:
                route = self.conv_bn(route, 128 // (2 ** i), filter_size=1, stride=1, padding=0)
                route = self.upsample(route)

        return self.outputs

