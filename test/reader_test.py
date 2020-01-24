# -*- coding: UTF-8 -*-
"""
reader 计算的单元测试
"""
import unittest
import reader
import codecs
from PIL import Image
from PIL import ImageDraw


class TestReader(unittest.TestCase):

    def test_rotate_image(self):
        target_image_size = [3, 640, 640]
        file_name = 'tr_img_01002'
        # file_name = 'tr_img_03184'
        image = Image.open("resources/" + file_name + '.jpg')
        im_width, im_height = image.size
        bbox_list = []
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
                bbox_list.append(bbox_sample)

        rotate_image, rotate_bbox = reader.ramdom_rotate(image, bbox_list)
        rotate_image = reader.resize_img(rotate_image, target_image_size)
        rotate_img_width, rotate_img_height = rotate_image.size
        draw = ImageDraw.Draw(rotate_image)
        for bbox_sample in rotate_bbox:
            bbox_sample[0] *= rotate_img_width
            bbox_sample[2] *= rotate_img_width
            bbox_sample[4] *= rotate_img_width
            bbox_sample[6] *= rotate_img_width

            bbox_sample[1] *= rotate_img_height
            bbox_sample[3] *= rotate_img_height
            bbox_sample[5] *= rotate_img_height
            bbox_sample[7] *= rotate_img_height

            draw.polygon(bbox_sample)

            rotate_image.save("resources/" + file_name + '-rotate.jpg')


if __name__ == '__main__':
    unittest.main()