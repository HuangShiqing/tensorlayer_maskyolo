import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm
from varible import *


def read_xml(ANN, pick):
    print('Parsing for {}'.format(pick))

    chunks = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)
    annotations = os.listdir('.')
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

    for file in tqdm(annotations):
        # actual parsing
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()

        segmented = root.find('segmented').text
        if segmented == '0':
            continue

        jpg = str(root.find('filename').text)
        jpg = jpg.rstrip('.jpg') if 'jpg' in jpg else jpg
        all = list()
        for obj in root.iter('object'):
            name = obj.find('name').text
            if name not in pick:
                continue
            all += [name]
        new_all = list(set(all))
        new_all.sort(key=all.index)
        add = [jpg, new_all]
        if len(all) is not 0:  # skip the image which not include any 'pick'
            chunks.append(add)
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in chunks:
        all = dump[1]
        for current in all:
            if current in pick:
                if current in stat:
                    stat[current] += 1
                else:
                    stat[current] = 1

    print('\nStatistics:')
    for i in stat: print('{}: {}'.format(i, stat[i]))
    print('Dataset size: {}'.format(len(chunks)))

    os.chdir(cur_dir)
    return chunks


def resize_img(img):
    img_w = img.shape[1]
    img_h = img.shape[0]

    ratio = img_w / img_h
    net_w, net_h = 416, 416
    if ratio < 1:
        new_h = int(net_h)
        new_w = int(net_h * ratio)
    else:
        new_w = int(net_w)
        new_h = int(net_w / ratio)
    im_sized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    dx = net_w - new_w
    dy = net_h - new_h

    if dx > 0:
        im_sized = np.pad(im_sized, ((0, 0), (int(dx / 2), 0), (0, 0)), mode='constant', constant_values=0)
        im_sized = np.pad(im_sized, ((0, 0), (0, dx - int(dx / 2)), (0, 0)), mode='constant', constant_values=0)
    else:
        im_sized = im_sized[:, -dx:, :]
    if dy > 0:
        im_sized = np.pad(im_sized, ((int(dy / 2), 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        im_sized = np.pad(im_sized, ((0, dy - int(dy / 2)), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        im_sized = im_sized[-dy:, :, :]
    return im_sized


def select(img, color=None):
    """
    img:
        shape=[416,416,3],RGB
    threshold:
        list. select the color
    new:
        shape=[416,416].
    """
    RGB = img == color
    mask = np.logical_and(RGB[:, :, 0], RGB[:, :, 1])
    mask = np.logical_and(mask, RGB[:, :, 2])
    new = np.zeros([416, 416])
    new[mask] = 255
    return new


def conv(img):
    """

    :param img:
    :return:
    """
    kernel = np.ones([8, 8]) / 255
    result = np.zeros([52, 52])
    for i in range(52):
        for j in range(52):
            result[i][j] = np.sum(np.matmul(img[i * 8:i * 8 + 8, j * 8:j * 8 + 8],
                                            kernel))  # 1 if np.sum(np.matmul(img[i * 8:i * 8 + 8, j * 8:j * 8 + 8], kernel)) > 0 else 0
    return result


def amplifier(segment_data):
    for i in range(52):
        for j in range(52):
            temp2 = np.full([8, 8], segment_data[i][j])
            if j == 0:
                temp = temp2
            else:
                temp = np.hstack((temp, temp2))
        if i == 0:
            result = temp
        else:
            result = np.vstack((result, temp))
    return result


def visualization(origin_img_sized, segment_data):
    origin_img_sized = (origin_img_sized * 255).astype(np.uint8)
    origin_mask = amplifier(segment_data)
    for i in range(3):
        for j in list(np.unique(segment_data)):
            if j == 0:
                continue
            origin_img_sized[:, :, i] = np.where(origin_mask == j,
                                                 origin_img_sized[:, :, i] * 0.5 + 0.5 * Gb_colors[j][i],
                                                 origin_img_sized[:, :, i])
    plt.imshow(origin_img_sized)
    plt.show()
    # origin_img_sized = origin_img_sized[:, :, ::-1]
    # origin_img_sized = origin_img_sized.copy()
    # for y in range(51):
    #     cv2.line(origin_img_sized, (0, 8 * (y + 1)), (416, 8 * (y + 1)), (0, 0, 255), 1)
    # for x in range(51):
    #     cv2.line(origin_img_sized, (8 * (x + 1), 0), (8 * (x + 1), 416), (0, 0, 255), 1)
    # cv2.imwrite('visualization.bmp', origin_img_sized)


def get_data(chunk):
    origin_img_path = Gb_origin_img_path
    segment_img_path = Gb_segment_img_path

    origin_img = cv2.imread(origin_img_path + chunk[0] + '.jpg')
    segment_img = cv2.imread(segment_img_path + chunk[0] + '.png')
    origin_img = origin_img[:, :, :: -1]
    segment_img = segment_img[:, :, :: -1]
    origin_img_sized = resize_img(origin_img)
    segment_img_sized = resize_img(segment_img)

    xml_labels = chunk[1]
    results = list()
    for label in Gb_labels:
        if label not in xml_labels:
            result = np.zeros([52, 52])
        else:
            color = Gb_colors[Gb_labels.index(label)]
            new = select(segment_img_sized, color=color)
            result = conv(new)
        results.append(result)
    results = np.array(results)
    segment_data = np.argmax(results, axis=0)

    return origin_img_sized, segment_data


def data_generator(chunks):
    batch_size = Gb_batch_size
    n = len(chunks)
    i = 0
    count = 0
    while count < (n / batch_size):
        origin_img_sizeds = []
        segment_datas = []
        while len(segment_datas) < batch_size:
            # for t in range(batch_size):
            i %= n
            origin_img_sized, segment_data = get_data(chunks[i])
            i += 1
            # if len(boxes_sized) is 0:  # in case all the box in a batch become empty becase of the augmentation
            #     continue
            origin_img_sizeds.append(origin_img_sized)
            segment_datas.append(segment_data)

        origin_img_sizeds = np.array(origin_img_sizeds)
        origin_img_sizeds = origin_img_sizeds / 255.
        segment_datas = np.array(segment_datas)
        yield origin_img_sizeds, segment_datas
        count += 1


if __name__ == '__main__':
    chunks = read_xml(Gb_ann_path, pick=Gb_labels)
    data_yield = data_generator(chunks)
    for origin_img_sizeds, segment_datas in data_yield:
        for i in range(16):
            visualization(origin_img_sizeds[i], segment_datas[i])
    exit()
