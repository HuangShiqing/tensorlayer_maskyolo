import tensorflow as tf
import tensorlayer as tl
import cv2
import numpy as np
from tensorlayer.layers import *
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from net import Gb_all_layer_out, ResLayer, RouteLayer, upsample, conv2d_unit, detection

# net_out = [tf.zeros(shape=(1, 52, 52, 3, 85)), tf.zeros(shape=(1, 26, 26, 3, 85)), tf.zeros(shape=(1, 13, 13, 3, 85))]
checkpoint_dir = './ckpt/'
ckpt_name = 'ep000-step46000-loss2.157-46000'
label = ['knot']
anchors = tf.constant([125, 311, 127, 192, 212, 378, 273, 178, 324, 490, 362, 865, 404, 292, 513, 505, 639, 727],
                      # [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                      dtype='float', shape=[1, 1, 1, 9, 2])
n_class = len(label)

input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
net = InputLayer(input_pb, name='input')
net = conv2d_unit(net, filters=32, kernels=3, strides=1, bn=True, name='0')
net = conv2d_unit(net, filters=64, kernels=3, strides=2, bn=True, name='1')
net = conv2d_unit(net, filters=32, kernels=1, strides=1, bn=True, name='2')
net = conv2d_unit(net, filters=64, kernels=3, strides=1, bn=True, name='3')
net = ResLayer(net, res=1, name='4')
net = conv2d_unit(net, filters=128, kernels=3, strides=2, bn=True, name='5')
net = conv2d_unit(net, filters=64, kernels=1, strides=1, bn=True, name='6')
net = conv2d_unit(net, filters=128, kernels=3, strides=1, bn=True, name='7')
net = ResLayer(net, res=5, name='8')
net = conv2d_unit(net, filters=64, kernels=1, strides=1, bn=True, name='9')
net = conv2d_unit(net, filters=128, kernels=3, strides=1, bn=True, name='10')
net = ResLayer(net, res=8, name='11')
net = conv2d_unit(net, filters=256, kernels=3, strides=2, bn=True, name='12')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='13')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='14')
net = ResLayer(net, res=12, name='15')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='16')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='17')
net = ResLayer(net, res=15, name='18')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='19')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='20')
net = ResLayer(net, res=18, name='21')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='22')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='23')
net = ResLayer(net, res=21, name='24')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='25')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='26')
net = ResLayer(net, res=24, name='27')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='28')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='29')
net = ResLayer(net, res=27, name='30')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='31')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='32')
net = ResLayer(net, res=30, name='33')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='34')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='35')
net = ResLayer(net, res=33, name='36')
net = conv2d_unit(net, filters=512, kernels=3, strides=2, bn=True, name='37')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='38')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='39')
net = ResLayer(net, res=37, name='40')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='41')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='42')
net = ResLayer(net, res=40, name='43')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='44')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='45')
net = ResLayer(net, res=43, name='46')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='47')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='48')
net = ResLayer(net, res=46, name='49')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='50')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='51')
net = ResLayer(net, res=49, name='52')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='53')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='54')
net = ResLayer(net, res=52, name='55')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='56')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='57')
net = ResLayer(net, res=55, name='58')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='59')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='60')
net = ResLayer(net, res=58, name='61')
net = conv2d_unit(net, filters=1024, kernels=3, strides=2, bn=True, name='62')
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, name='63')
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, name='64')
net = ResLayer(net, res=62, name='65')
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, name='66')
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, name='67')
net = ResLayer(net, res=65, name='68')
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, name='69')
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, name='70')
net = ResLayer(net, res=68, name='71')
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, name='72')
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, name='73')
net = ResLayer(net, res=71, name='74')
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, name='75')
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, name='76')
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, name='77')
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, name='78')
net = conv2d_unit(net, filters=512, kernels=1, strides=1, bn=True, name='79')
net = conv2d_unit(net, filters=1024, kernels=3, strides=1, bn=True, name='80')
net = conv2d_unit(net, filters=3 * (5 + n_class), kernels=1, strides=1, act='liner', bn=False, name='81')
detection(net, '82')
net = RouteLayer(net, [79], name='83')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='84')
net = upsample(net, scale=2, name='85')
net = RouteLayer(net, [85, 61], name='86')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='87')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='88')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='89')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='90')
net = conv2d_unit(net, filters=256, kernels=1, strides=1, bn=True, name='91')
net = conv2d_unit(net, filters=512, kernels=3, strides=1, bn=True, name='92')
net = conv2d_unit(net, filters=3 * (5 + n_class), kernels=1, strides=1, act='liner', bn=False, name='93')
detection(net, '94')
net = RouteLayer(net, [91], name='95')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='96')
net = upsample(net, scale=2, name='97')
net = RouteLayer(net, [97, 36], name='98')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='99')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='100')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='101')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='102')
net = conv2d_unit(net, filters=128, kernels=1, strides=1, bn=True, name='103')
net = conv2d_unit(net, filters=256, kernels=3, strides=1, bn=True, name='104')
net = conv2d_unit(net, filters=3 * (5 + n_class), kernels=1, strides=1, act='liner', bn=False, name='105')
detection(net, '106')
net_out = [Gb_all_layer_out[106], Gb_all_layer_out[94], Gb_all_layer_out[82]]

# 读取ckpt里保存的参数
sess = tf.InteractiveSession()
saver = tf.train.Saver()
# 如果有checkpoint这个文件可以加下面这句话，如果只有一个ckpt文件就不需要这个if
if tf.train.get_checkpoint_state(checkpoint_dir):  # 确认是否存在
    saver.restore(sess, checkpoint_dir + ckpt_name)
    print("load ok!")
else:
    print("ckpt文件不存在")

# tensor = tf.global_variables('layer_0_conv')
# b = sess.run(tensor)
# c = sess.run(net_out, feed_dict={input_pb: image_data})
if not os.path.exists('out'):
    os.mkdir('out')
file_name = input('Input image filedir:')
img_path = os.listdir(file_name)
for path in tqdm(img_path):
    abs_path = file_name + path
    img = cv2.imread(abs_path)
# while True:
#     file_name = input('Input image filename:')
#     img = cv2.imread(file_name)
    img = img[:, :, ::-1]  # RGB image
    img_shape = img.shape[0:2][::-1]

    _scale = min(416 / img_shape[0], 416 / img_shape[1])
    _new_shape = (int(img_shape[0] * _scale), int(img_shape[1] * _scale))
    im_sized = cv2.resize(img, _new_shape)
    im_sized = np.pad(im_sized,
                      (
                          (int((416 - _new_shape[1]) / 2), 416 - _new_shape[1] - int((416 - _new_shape[1]) / 2)),
                          (int((416 - _new_shape[0]) / 2), 416 - _new_shape[0] - int((416 - _new_shape[0]) / 2)),
                          (0, 0)
                      ),
                      mode='constant')
    image_data = np.array(im_sized, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    input_shape = tf.cast(tf.shape(net_out[2])[1:3] * 32, dtype='float32')[::-1]  # hw
    image_shape = tf.cast(img_shape, dtype='float32')[::-1]  # hw
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    # with tf.Session() as sess:
    #     a = sess.run(scale)

    boxes = list()
    box_scores = list()

    cellbase_x = tf.to_float(tf.reshape(tf.tile(tf.range(52), [52]), (1, 52, 52, 1, 1)))
    cellbase_y = tf.transpose(cellbase_x, (0, 2, 1, 3, 4))
    cellbase_grid = tf.tile(tf.concat([cellbase_x, cellbase_y], -1), [1, 1, 1, 3, 1])
    # classes = list()
    for i in range(3):  # 52 26 13
        anchor = anchors[..., 3 * i:3 * (i + 1), :]
        # feats = model.output[i]
        feats = net_out[i]

        grid_w = tf.shape(feats)[1]  # 13
        grid_h = tf.shape(feats)[2]  # 13
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1, 1, 1, 1, 2])

        feats = tf.reshape(feats, [-1, grid_w, grid_h, 3, n_class + 5])

        # Adjust preditions to each spatial grid point and anchor size.
        box_xy = (tf.sigmoid(feats[..., :2]) + cellbase_grid[:, :grid_w, :grid_h, :, :]) / tf.cast(grid_factor[::-1],
                                                                                                   'float32')
        box_wh = tf.exp(feats[..., 2:4]) * anchor / tf.cast(input_shape[::-1], 'float32')
        box_confidence = tf.sigmoid(feats[..., 4:5])
        box_class_probs = tf.sigmoid(feats[..., 5:])

        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        _boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)

        # Scale boxes back to original image shape.
        _boxes *= tf.concat([tf.cast(image_shape, 'float32'), tf.cast(image_shape, 'float32')], axis=-1)
        _boxes = tf.reshape(_boxes, [-1, 4])

        _box_scores = box_confidence * box_class_probs
        _box_scores = tf.reshape(_box_scores, [-1, n_class])
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)

    mask = box_scores >= 0.3
    max_num_boxes = tf.constant(20, dtype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(n_class):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_num_boxes, iou_threshold=0.5)
        class_boxes = tf.gather(class_boxes, nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)

    b, s, c = sess.run([boxes_, scores_, classes_], feed_dict={input_pb: image_data})

    # plt.cla()
    # plt.imshow(img)
    # for i, obj in enumerate(b):
    #     x1 = obj[1]
    #     x2 = obj[3]
    #     y1 = obj[0]
    #     y2 = obj[2]
    #
    #     # TODO: change the color of text
    #     plt.text(x1, y1 - 10, round(s[i], 2))
    #     plt.text(x2 - 30, y1 - 10, label[c[i]])
    #     plt.hlines(y1, x1, x2, colors='red')
    #     plt.hlines(y2, x1, x2, colors='red')
    #     plt.vlines(x1, y1, y2, colors='red')
    #     plt.vlines(x2, y1, y2, colors='red')
    # plt.show()
    img = img[:, :, ::-1]
    file = open("./out/" + path.rstrip('.jpg') + '.txt', 'w')
    for i, obj in enumerate(b):
        cv2.rectangle(img, (obj[1], obj[0]), (obj[3], obj[2]), (0, 0, 255), 3)
        cv2.putText(img, str(round(s[i], 2)), (int(obj[1]), int(obj[0]) - 10), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255),
                    3)
        cv2.putText(img, str(label[c[i]]), (int(obj[3]) - 100, int(obj[0]) - 10), cv2.FONT_HERSHEY_COMPLEX, 2,
                    (0, 0, 255), 3)

        file.write('{0} {1} '.format(label[c[i]], s[i]))
        file.write('{0} {1} {2} {3}'.format(obj[1], obj[0], obj[3], obj[2]))
        file.write('\n')
    file.close()
    cv2.imwrite("./out/" + path, img)
    # cv2.imwrite("1.jpg", img)
