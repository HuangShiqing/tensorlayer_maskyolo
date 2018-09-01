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
ckpt_name = 'ep373-step66866-loss1389.557'
# label = ['knot']

n = 2
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
net = conv2d_unit(net, filters=n, kernels=1, strides=1, act='liner', bn=False, name='81')
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
net = conv2d_unit(net, filters=n, kernels=1, strides=1, act='liner', bn=False, name='93')
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
net = conv2d_unit(net, filters=n, kernels=1, strides=1, act='liner', bn=False, name='105')
detection(net, '106')
net_out = Gb_all_layer_out[106]

# 读取ckpt里保存的参数
sess = tf.InteractiveSession()
saver = tf.train.Saver()
# 如果有checkpoint这个文件可以加下面这句话，如果只有一个ckpt文件就不需要这个if
if tf.train.get_checkpoint_state(checkpoint_dir):  # 确认是否存在
    saver.restore(sess, checkpoint_dir + ckpt_name)
    print("load ok!")
else:
    print("ckpt文件不存在")

from data import resize_img, visualization

img_path = 'C:/Users/john/Desktop/timg.jpg'#'D:/DeepLearning/data2/VOCdevkit/VOC2012/JPEGImages/2007_000170.jpg'
origin_img = cv2.imread(img_path)
origin_img = origin_img[:, :, :: -1]
origin_img = resize_img(origin_img)
img = origin_img / 255.
img = np.expand_dims(img, axis=0)

adjusted_out = tf.sigmoid(net_out)
out = sess.run(adjusted_out, feed_dict={input_pb: img})

out = out[0, :, :, 1]
out = out * 21
out = np.floor(out)
out = out.astype(np.uint8)
cv2.imwrite('out1.bmp',out)
visualization(origin_img, out)

exit()
