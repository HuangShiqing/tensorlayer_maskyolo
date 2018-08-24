import os
import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt  # dealing with plots
from tqdm import tqdm

from varible import *


# <class 'list'>: ['2007_000027.jpg', [486, 500, [['person', 174, 101, 349, 351]]]]
def pascal_voc_clean_xml(ANN, pick, exclusive=False):
    print('Parsing for {} {}'.format(
        pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
    # annotations = glob.glob(str(annotations) + '*.xml')
    size = len(annotations)

    # dumps = list()
    # cur_dir = os.getcwd()
    # os.chdir(ANN)
    # path = '/home/hsq/DeepLearning/data/car/bdd100k/daytime.txt'
    # annotations = []
    # with open(path) as fh:
    #     for line in tqdm(fh):
    #         temp = '/home/hsq/DeepLearning/data/car/bdd100k/labels/100k/train_xml/' + line.strip()[-21:].rstrip(
    #             '.jpg') + '.xml'
    #         annotations.append(temp)
    # size = len(annotations)

    for i, file in enumerate(annotations):
        # progress bar
        sys.stdout.write('\r')
        percentage = 1. * (i + 1) / size
        progress = int(percentage * 20)
        bar_arg = [progress * '=', ' ' * (19 - progress), percentage * 100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()

        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text) + '.jpg'
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):
            # current = list()
            current = dict()
            name = obj.find('name').text
            if name not in pick:
                continue

            xmlbox = obj.find('bndbox')
            xn = int(float(xmlbox.find('xmin').text))
            xx = int(float(xmlbox.find('xmax').text))
            yn = int(float(xmlbox.find('ymin').text))
            yx = int(float(xmlbox.find('ymax').text))
            # current = [name, xn, yn, xx, yx]
            current['name'] = name
            current['xmin'] = xn
            current['xmax'] = xx
            current['ymin'] = yn
            current['ymax'] = yx
            all += [current]

        add = [[jpg, [w, h, all]]]
        if len(all) is not 0:  # skip the image which not include any 'pick'
            dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current['name'] in pick:
                if current['name'] in stat:
                    stat[current['name']] += 1
                else:
                    stat[current['name']] = 1

    print('\nStatistics:')
    for i in stat: print('{}: {}'.format(i, stat[i]))
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps


def _rand_scale(scale):
    scale = np.random.uniform(1, scale)
    return scale if (np.random.randint(2) == 0) else 1. / scale;


def random_flip(image, flip):
    if flip == 1: return cv2.flip(image, 1)
    return image


def _constrain(min_v, max_v, value):
    if value < min_v: return min_v
    if value > max_v: return max_v
    return value


def random_distort_image(image, hue=18, saturation=1.5, exposure=1.5):
    # determine scale factors
    dhue = np.random.uniform(-hue, hue)
    dsat = _rand_scale(saturation)
    dexp = _rand_scale(exposure)

    # convert RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype('float')

    # change satuation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp

    # change hue
    image[:, :, 0] += dhue
    image[:, :, 0] -= (image[:, :, 0] > 180) * 180
    image[:, :, 0] += (image[:, :, 0] < 0) * 180

    # convert back to RGB from HSV
    return cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2RGB)


def apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy):
    im_sized = cv2.resize(image, (new_w, new_h))

    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (dx, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[:, -dx:, :]
    if (new_w + dx) < net_w:
        im_sized = np.pad(im_sized, ((0, 0), (0, net_w - (new_w + dx)), (0, 0)), mode='constant', constant_values=127)

    if dy > 0:
        im_sized = np.pad(im_sized, ((dy, 0), (0, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        im_sized = im_sized[-dy:, :, :]

    if (new_h + dy) < net_h:
        im_sized = np.pad(im_sized, ((0, net_h - (new_h + dy)), (0, 0), (0, 0)), mode='constant', constant_values=127)

    return im_sized[:net_h, :net_w, :]


def correct_bounding_boxes(boxes, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h):
    boxes = copy.deepcopy(boxes)

    # randomize boxes' order
    # np.random.shuffle(boxes)

    # correct sizes and positions
    sx, sy = float(new_w) / image_w, float(new_h) / image_h
    zero_boxes = []

    for i in range(len(boxes)):
        boxes[i]['xmin'] = int(_constrain(0, net_w, boxes[i]['xmin'] * sx + dx))
        boxes[i]['xmax'] = int(_constrain(0, net_w, boxes[i]['xmax'] * sx + dx))
        boxes[i]['ymin'] = int(_constrain(0, net_h, boxes[i]['ymin'] * sy + dy))
        boxes[i]['ymax'] = int(_constrain(0, net_h, boxes[i]['ymax'] * sy + dy))

        # if boxes[i]['xmax'] <= boxes[i]['xmin'] or boxes[i]['ymax'] <= boxes[i]['ymin']:
        #     zero_boxes += [i]
        #     continue

        if flip == 1:
            swap = boxes[i]['xmin']
            boxes[i]['xmin'] = net_w - boxes[i]['xmax']
            boxes[i]['xmax'] = net_w - swap

    # boxes = [boxes[i] for i in range(len(boxes)) if i not in zero_boxes]
    return boxes


# TODO maybe list.pop is better than list.remove
def remove_outbox(boxes):
    temp = copy.deepcopy(boxes)
    for i, obj in enumerate(temp):
        if (obj['xmin'] == 416 and obj['xmax'] == 416) or (obj['ymin'] == 416 and obj['ymax'] == 416):
            boxes.remove(obj)
    return boxes


def remove_smallobj(boxes):
    temp = copy.deepcopy(boxes)
    for i, obj in enumerate(temp):
        if (obj['xmax'] - obj['xmin'] < 30) and (obj['ymax'] - obj['ymin'] < 30) or (
                (obj['xmax'] - obj['xmin']) * (obj['ymax'] - obj['ymin']) < 750):
            boxes.remove(obj)
    return boxes


# images_path = "D:/DeepLearning/data/VOCdevkit/VOC2012/JPEGImages/"
# annotations_path = "D:/DeepLearning/data/VOCdevkit/VOC2012/Annotations/"
# chunks = pascal_voc_clean_xml(annotations_path, "person")


# chunk = a[0]

# chunk = ['2007_000027.jpg', [486, 500, [{'ymax': 351, 'name': 'person', 'xmax': 349, 'ymin': 101, 'xmin': 174}]]]
# chunk = ['2007_000032.jpg', [500, 281, [{'name': 'person', 'ymax': 229, 'ymin': 180, 'xmin': 195, 'xmax': 213},
#                                         {'name': 'person', 'ymax': 238, 'ymin': 189, 'xmin': 26, 'xmax': 44}]]]

def get_data(chunk, images_path):
    net_w = net_h = 416
    jitter = Gb_jitter
    img_abs_path = images_path + chunk[0]
    w, h, allobj_ = chunk[1]

    if allobj_ is None:
        return None, None
    image = cv2.imread(img_abs_path)  # RGB image

    # for obj in allobj_:
    #     cv2.rectangle(image, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),(0,255,0), 1)
    # cv2.imwrite("C:/Users/john/Desktop/0.jpg", image)

    if image is None:
        print('Cannot find ', img_abs_path)
    image = image[:, :, ::-1]  # RGB image

    image_h, image_w, _ = image.shape

    # determine the amount of scaling and cropping
    dw = jitter * image_w
    dh = jitter * image_h

    new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh))
    scale = np.random.uniform(1, 2)

    if new_ar < 1:
        new_h = int(scale * net_h)
        new_w = int(net_h * new_ar)
    else:
        new_w = int(scale * net_w)
        new_h = int(net_w / new_ar)

    dx = int(np.random.uniform(0, net_w - new_w))
    dy = int(np.random.uniform(0, net_h - new_h))

    # apply scaling and cropping
    im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
    # randomly distort hsv space
    # im_sized = random_distort_image(im_sized)
    # randomly flip
    flip = np.random.randint(2)
    im_sized = random_flip(im_sized, flip)
    # correct the size and pos of bounding boxes
    boxes_sized = correct_bounding_boxes(allobj_, new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)
    # remove the box which out of the 416*416 after augmentation
    boxes_sized = remove_outbox(boxes_sized)
    # remove the box which ares is too small to get nan loss
    boxes_sized = remove_smallobj(boxes_sized)
    # temp = copy.deepcopy(boxes_sized)
    # for i, obj in enumerate(temp):
    #     if (obj['xmin'] == 416 and obj['xmax'] == 416) or (obj['ymin'] == 416 and obj['ymax'] == 416):
    #         boxes_sized.pop(i)
    #     cv2.rectangle(im_sized, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (0, 255, 0), 1)
    # im_sized = im_sized[:, :, ::-1]  # BGR image
    # cv2.imwrite("C:/Users/john/Desktop/1.jpg", im_sized)

    # for obj in boxes_sized:
    #     cv2.rectangle(im_sized, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), (0, 255, 0), 1)
    # im_sized = im_sized[:, :, ::-1]  # BGR image
    # cv2.imwrite("C:/Users/john/Desktop/1.jpg", im_sized)
    return im_sized, boxes_sized


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def process_box(boxes):
    batch_size = Gb_batch_size
    base_grid_h = base_grid_w = 13
    net_w = net_h = 416
    anchors = Gb_anchors
    anchors_BoundBox = [BoundBox(0, 0, anchors[2 * i], anchors[2 * i + 1]) for i in range(len(anchors) // 2)]
    labels = Gb_label

    y_true = list()
    # initialize the inputs and the outputs
    # TODO carefully think about the order of the y_true and anchors
    y_true.append(np.zeros((batch_size, 4 * base_grid_h, 4 * base_grid_w, 3,
                            4 + 1 + len(Gb_label))))  # desired network output 3
    y_true.append(np.zeros((batch_size, 2 * base_grid_h, 2 * base_grid_w, 3,
                            4 + 1 + len(Gb_label))))  # desired network output 2
    y_true.append(np.zeros((batch_size, 1 * base_grid_h, 1 * base_grid_w, 3,
                            4 + 1 + len(Gb_label))))  # desired network output 1

    for instance_index in range(batch_size):
        # allobj_sized = [{'xmin': 96, 'name': 'person', 'ymin': 96, 'xmax': 304, 'ymax': 304},
        #                 {'xmin': 329, 'name': 'person', 'ymin': 272, 'xmax': 337, 'ymax': 337}]
        allobj_sized = boxes[instance_index]
        for obj in allobj_sized:
            # find the best anchor box for this object
            max_anchor = None
            max_index = -1
            max_iou = -1

            # TODO replace this
            shifted_box = BoundBox(0,
                                   0,
                                   obj['xmax'] - obj['xmin'],
                                   obj['ymax'] - obj['ymin'])
            for i in range(len(anchors_BoundBox)):
                anchor = anchors_BoundBox[i]
                iou = bbox_iou(shifted_box, anchor)

                if max_iou < iou:
                    max_anchor = anchor
                    max_index = i
                    max_iou = iou

                    # determine the yolo to be responsible for this bounding box
            grid_h, grid_w = y_true[max_index // 3].shape[1:3]

            # determine the position of the bounding box on the grid
            center_x = .5 * (obj['xmin'] + obj['xmax'])
            center_x = center_x / float(net_w)  # * grid_w  # sigma(t_x) + c_x
            center_y = .5 * (obj['ymin'] + obj['ymax'])
            center_y = center_y / float(net_h)  # * grid_h  # sigma(t_y) + c_y

            # determine the sizes of the bounding box
            # w = np.log((obj['xmax'] - obj['xmin']) / float(max_anchor.xmax))  # t_w
            # h = np.log((obj['ymax'] - obj['ymin']) / float(max_anchor.ymax))  # t_h
            w = (obj['xmax'] - obj['xmin']) / float(net_w)  # t_w
            h = (obj['ymax'] - obj['ymin']) / float(net_h)  # t_h

            box = [center_x, center_y, w, h]

            # determine the index of the label
            obj_indx = labels.index(obj['name'])

            # determine the location of the cell responsible for this object
            grid_x = int(np.floor(center_x * grid_w))
            grid_y = int(np.floor(center_y * grid_h))

            # assign ground truth x, y, w, h, confidence and class probs to y_batch
            y_true[max_index // 3][instance_index, grid_y, grid_x, max_index % 3] = 0
            y_true[max_index // 3][instance_index, grid_y, grid_x, max_index % 3, 0:4] = box
            y_true[max_index // 3][instance_index, grid_y, grid_x, max_index % 3, 4] = 1.
            y_true[max_index // 3][instance_index, grid_y, grid_x, max_index % 3, 5 + obj_indx] = 1

    return y_true


""""
# Output shape
        image_data     [Gb_batch_size, 416, 416, 3]
        boxes_labeled  [[Gb_batch_size,52,52,3,85],[Gb_batch_size, 26,26,3,85],[Gb_batch_size,13,13,3,85]]
"""


def data_generator(chunks):
    images_path = Gb_images_path
    # annotations_path = Gb_ann_path
    batch_size = Gb_batch_size
    # pick = Gb_label

    # chunks = pascal_voc_clean_xml(annotations_path, pick)
    n = len(chunks)
    i = 0
    count = 0
    while count < (n / Gb_batch_size):
        image_data = []
        box_data = []
        while len(box_data) < batch_size:
            # for t in range(batch_size):
            i %= n
            imgs_sized, boxes_sized = get_data(chunks[i], images_path)
            i += 1
            # plt.cla()
            # plt.imshow(imgs_sized)
            # for obj in boxes_sized:
            #     x1 = obj['xmin']
            #     x2 = obj['xmax']
            #     y1 = obj['ymin']
            #     y2 = obj['ymax']
            #
            #     plt.hlines(y1, x1, x2, colors='red')
            #     plt.hlines(y2, x1, x2, colors='red')
            #     plt.vlines(x1, y1, y2, colors='red')
            #     plt.vlines(x2, y1, y2, colors='red')
            # plt.show()

            if len(boxes_sized) is 0:  # in case all the box in a batch become empty becase of the augmentation
                continue

            image_data.append(imgs_sized)
            box_data.append(boxes_sized)
        boxes_labeled = process_box(box_data)

        image_data = np.array(image_data)
        image_data = image_data / 255.
        # boxes_labeled = np.array(boxes_labeled)
        yield image_data, boxes_labeled
        count += 1

# annotations_path = Gb_ann_path
# pick = Gb_label
# chunks = pascal_voc_clean_xml(annotations_path, pick)
# a = data_generator(chunks)
# for x in a:
#     print('ok')
#
# exit()
