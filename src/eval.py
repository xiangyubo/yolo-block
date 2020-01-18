import codecs
import json
import os
import numpy as np
import time
import paddle
import paddle.fluid as fluid
import text_detection_ap

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from train_config import train_parameters
from yolo_block_model import get_yolo
from text_detection_ap import detection_one_pr


ues_tiny = train_parameters['use_tiny']
yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']

label_dict = {}
with codecs.open(train_parameters['data_dir'] + 'label_list.txt') as f:
    for line in f:
        parts = line.strip().split()
        label_dict[int(parts[0])] = parts[1]
print(label_dict)
class_dim = len(label_dict)
print("class dim:{0}".format(class_dim))
place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = yolo_config['freeze_dir']
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)


def draw_bbox_image(img, boxes, labels, save_name):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """
    # font = ImageFont.truetype("font.ttf", 25)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red')
        draw.text((xmin, ymin), label_dict[int(label)], (255, 255, 0))
    img.save(save_name)


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = Image.open(img_path)
    img = resize_img(origin, target_size)
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img


def eval_read(file_list):
    def reader():
        for line in file_list:
            try:
                ######################  以下可能是需要自定义修改的部分   ############################
                parts = line.split('\t')
                image_path = parts[0]
                origin, img, _ = read_image(image_path)
                input_w, input_h = origin.size[0], origin.size[1]
                image_shape = np.array([input_h, input_w], dtype=np.int32)
                # bbox 的列表，每一个元素为这样
                # layout: label | x-center | y-cneter | width | height
                gt_boxes = []
                for object_str in parts[1:]:
                    if len(object_str) <= 1:
                        continue

                    object = json.loads(object_str)
                    bbox = object['coordinate']
                    # 这个地方的 box 是 x,y,w,h 的格式
                    gt_box = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
                    gt_boxes.append(gt_box)
                ######################  可能需要自定义修改部分结束   ############################
                yield img, image_shape[np.newaxis, :], gt_boxes, image_path
            except Exception as e:
                print("deal {0} excaption {1}".format(line, e))
    return reader


def infer(image, image_shape, image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image:
    :param image_shape:
    :return:
    """
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: image,
                                  feed_target_names[1]: image_shape},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = time.time() - t1
    # print("predict cost time:{0}".format("%2.2f sec" % period))
    bboxes = np.array(batch_outputs[0])
    # print(bboxes)

    if bboxes.shape[1] != 6:
        print("No object found in {}".format(image_path))
        return [], [], []
    return bboxes[:, 0].astype('int32'), bboxes[:, 1].astype('float32'), bboxes[:, 2:].astype('float32')


if __name__ == '__main__':
    file_path = os.path.join(train_parameters['data_dir'], train_parameters['eval_list'])
    # file_path = os.path.join(train_parameters['data_dir'], train_parameters['train_list'])
    file_path = os.path.join(train_parameters['data_dir'], 'debug.txt')
    images_list = [line.strip() for line in open(file_path)]
    image_count = len(images_list)
    reader = eval_read(images_list)
    total_precision = 0
    total_recall = 0
    total_cover = 0
    for i, data in enumerate(reader()):
        labels, scores, boxes = infer(data[0], data[1], data[3])
        precision, recall, cover = detection_one_pr(data[2], boxes)
        total_precision += precision
        total_recall += recall
        total_cover += cover
        print("{} precision:{} recall:{} cover:{}".format(data[3], precision, recall, cover))

    print("average precision:{} average recall:{} average cover:{}".
          format(total_precision / image_count, total_recall / image_count, total_cover /image_count))
