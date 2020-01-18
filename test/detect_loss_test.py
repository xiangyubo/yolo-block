# -*- coding: UTF-8 -*-
"""
loss 计算的单元测试
"""
import unittest
import numpy as np
import codecs
from utils import box_xywh_to_xyxy
from utils import distance_of_pp
from PIL import Image
from PIL import ImageDraw
from yolo_block_loss import detect_loss
from yolo_block_loss import detect_loss_grad
from yolo_block_loss import split_filed_to_block


class TestDetectLoss(unittest.TestCase):
    """
    detect_loss 的单元测试
    """

    def prepare_forward_input(self):
        """
        准备向前计算的输入参数
        :return:
        """
        # 构造输入参数
        # 预测结果，[n, c, h, w]
        c_1_1 = [0.78, 0.22, 0.21, 0.03, 0.05, 0.86, 0.2, 0.2, 0.01, 0.02, 0.2, 0.3, 0.4, 0.5, 0.9, 0.8, 0.7, 0.6]
        c_1_2 = [0.32, 0.1, 0.2, 0.001, 0.002, 0.32, 0.1, 0.2, 0.001, 0.002, 0.2, 0.3, 0.4, 0.5, 0.9, 0.8, 0.7, 0.6]
        c_2_1 = [0.02, 0.1, 0.2, 0.52, 0.52, 0.03, 0.1, 0.2, 0.001, 0.002, 0.2, 0.3, 0.4, 0.5, 0.9, 0.8, 0.7, 0.6]
        c_2_2 = [0.78, 0.22, 0.21, 0.04, 0.05, 0.86, 0.1, 0.2, 0.005, 0.008, 0.2, 0.3, 0.4, 0.5, 0.9, 0.8, 0.7, 0.6]
        # [n:2, c:(2*(1+4)+8), h:2, w:2]
        pred = np.array([[[c_1_1, c_1_2], [c_2_1, c_2_2]], [[c_1_1, c_1_2], [c_2_1, c_2_2]]])
        pred = pred.transpose([0, 3, 1, 2])

        # 标注的 ground truth, [n, f, 8]
        empty = [0, 0, 0, 0, 0, 0, 0, 0]
        field1_1 = [5 / 64, 5 / 64, 32 / 64, 3 / 64, 35 / 64, 32 / 64, 4 / 64, 34 / 64]
        field2_1 = [5 / 64, 5 / 64, 58 / 64, 3 / 64, 60 / 64, 32 / 64, 3 / 64, 33 / 64]
        field2_2 = [38 / 64, 30 / 64, 45 / 64, 33 / 64, 46 / 64, 43 / 64, 35 / 64, 41 / 64]
        # [n:2, f:2, c:8]
        gt_quad = np.array([[field1_1, empty], [field2_1, field2_2]])

        image_size = [64, 64]
        down_ratio = [32]
        yolo_anchors = [10, 10, 30, 30]     # 2 组 anchor
        print("pred shape:{}".format(pred.shape))
        return pred, gt_quad, image_size, down_ratio, yolo_anchors

    def test_detect_loss(self):
        """
        测试 loss 计算部分
        :return:
        """
        pred, gt_quad, image_size, down_ratio, yolo_anchors = self.prepare_forward_input()
        loss_result = detect_loss(pred, gt_quad, image_size, down_ratio, yolo_anchors)
        loss = loss_result[0]
        grid_matched_mask = loss_result[1]
        grid_matched_block = loss_result[2]
        grid_matched_link = loss_result[3]
        print("loss: {}".format(loss))
        print("grid_matched_mask: {}".format(grid_matched_mask))
        print("grid_matched_block: {}".format(grid_matched_block))
        print("grid_matched_link: {}".format(grid_matched_link))

    def test_detect_loss_grad(self):
        """
        测试梯度计算部分
        :return:
        """
        # 向前计算的输入
        pred, gt_quad, image_size, down_ratio, yolo_anchors = self.prepare_forward_input()

        # 向前计算的结果
        loss = [59.57348246, 255.91055196]
        grid_matched_mask = [[[1, 0], [0, 0]], 
                             [[1, 1], [0, 1]]]

        grid_matched_block = [[[[20, 19, 31, 31], [0, 0, 0, 0]], 
                               [[0, 0, 0, 0], [0, 0, 0, 0]]],

                              [[[20, 19, 28, 28], [52, 18, 29, 29]],
                               [[0, 0, 0, 0], [41, 36, 11, 13]]]]

        grid_matched_link = [[[[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0.]],
                              [[0, 0, 0, 0, 0, 0, 0, 0.], [0, 0, 0, 0, 0, 0, 0, 0.]]],

                             [[[0, 0, 0, 1, 0, 0, 0, 0.], [0, 0, 0, 0, 0, 0, 0, 1.]],
                              [[0, 0, 0, 0, 0, 0, 0, 0.], [0, 0, 0, 0, 0, 0, 0, 0.]]]]
        grad = detect_loss_grad(pred, gt_quad, image_size, down_ratio, yolo_anchors,
                                loss, grid_matched_mask, grid_matched_block, grid_matched_link,
                                np.ones(pred.shape), 1, 1, 1)
        grad = grad[0]
        print("grad shape: {}".format(grad.shape))
        print("grad: {}".format(grad.transpose([0, 2, 3, 1])))

    def test_split_block(self):
        target_image_size = [640, 640]
        down_ratio_list = [8, 16, 32, 64, 128]
        down_ratio_dict = {}
        tolerant_ratio = 1.5
        # file_name = 'tr_img_01002'
        file_name = 'tr_img_01105'
        # file_name = 'tr_img_03184'
        image = Image.open("resources/" + file_name + '.jpg')
        resize_image = image.resize(target_image_size)
        im_width, im_height = image.size
        x_rate = im_width / target_image_size[1]
        y_rate = im_height / target_image_size[0]

        with codecs.open("resources/" + file_name + '.txt', encoding='utf8') as f:
            for line in f:
                parts = line.split(',')
                bbox_sample = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                               float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                bbox_sample[0] /= im_width
                bbox_sample[2] /= im_width
                bbox_sample[4] /= im_width
                bbox_sample[6] /= im_width

                bbox_sample[1] /= im_height
                bbox_sample[3] /= im_height
                bbox_sample[5] /= im_height
                bbox_sample[7] /= im_height
                bbox_sample = np.array(bbox_sample)
                left_height = distance_of_pp(bbox_sample[0:0 + 2], bbox_sample[6:6 + 2])
                up_width = distance_of_pp(bbox_sample[0:0 + 2], bbox_sample[2:2 + 2])
                left_height *= target_image_size[0]
                up_width *= target_image_size[1]
                bbox_h = left_height if left_height <= up_width else up_width
                for down_ratio in down_ratio_list:
                    h_d_s = bbox_h / down_ratio
                    s_d_h = down_ratio / bbox_h
                    if h_d_s <= 2.0 and h_d_s >= 1.0:
                        block_list = split_filed_to_block(bbox_sample, target_image_size, down_ratio)
                        block_list = box_xywh_to_xyxy(block_list)
                        if down_ratio not in down_ratio_dict:
                            down_ratio_dict[down_ratio] = []
                        down_ratio_block_list = down_ratio_dict[down_ratio]
                        down_ratio_block_list.extend(block_list)
        for (down_ratio, block_list) in down_ratio_dict.items():
            temp_image = image.copy()
            temp_resized_image = resize_image.copy()
            draw = ImageDraw.Draw(temp_image)
            resize_draw = ImageDraw.Draw(temp_resized_image)
            for gt_block in block_list:
                draw.rectangle((gt_block[0] * x_rate, gt_block[1] * y_rate, gt_block[2] * x_rate, gt_block[3] * y_rate), None, 'red')
                resize_draw.rectangle((gt_block[0], gt_block[1], gt_block[2], gt_block[3]), None, 'red')

            temp_image.save(file_name + '-' + str(down_ratio) + '-result.jpg')
            temp_resized_image.save(file_name + '-' + str(down_ratio) + '-resize-result.jpg')


if __name__ == '__main__':
    unittest.main()
