import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import *

from tqdm import tqdm
from net import Gb_all_layer_out, ResLayer, RouteLayer, upsample, conv2d_unit, detection

checkpoint_dir = './ckpt3/'
model_name = 'yolov3.ckpt'
n_class = 80
weights_path = 'yolov3.weights'

# Load weights and config.
print('Loading weights.')
weights_file = open(weights_path, 'rb')
major, minor, revision = np.ndarray(
    shape=(3,), dtype='int32', buffer=weights_file.read(12))
if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
    seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
else:
    seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
print('Weights Header: ', major, minor, revision, seen)

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

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in tqdm(range(106)):
        if i in [4, 8, 11, 15, 18, 21, 24, 27, 30, 33, 36, 40, 43, 46, 49, 52, 55, 58, 61, 65, 68, 71, 74]:  # res
            pass
        elif i in [83, 86, 95, 98]:  # route
            pass
        elif i in [82, 94, 106]:  # detction
            pass
        elif i in [85, 97]:  # upsample
            pass
        else:  # conv
            tensor_conv_w = tf.global_variables('layer_' + str(i) + '_conv')[0]

            in_ch = tensor_conv_w.get_shape().as_list()[-2]
            filter_num = tensor_conv_w.get_shape().as_list()[-1]
            kernel = tensor_conv_w.get_shape().as_list()[0]

            conv_bias = np.ndarray(
                shape=(filter_num,),  # (32,),
                dtype='float32',
                buffer=weights_file.read(filter_num * 4))

            if i not in [81, 93, 105]:  # no bn
                tensor_bn_beta = tf.global_variables('layer_' + str(i) + '_bn')[0]
                tensor_bn_gamma = tf.global_variables('layer_' + str(i) + '_bn')[1]
                tensor_bn_mean = tf.global_variables('layer_' + str(i) + '_bn')[2]
                tensor_bn_variance = tf.global_variables('layer_' + str(i) + '_bn')[3]

                bn_weights = np.ndarray(
                    shape=(3, filter_num),  # (3, 32),
                    dtype='float32',
                    buffer=weights_file.read(filter_num * 12))

                tf.assign(tensor_bn_beta, conv_bias).eval()
                tf.assign(tensor_bn_gamma, bn_weights[0]).eval()
                tf.assign(tensor_bn_mean, bn_weights[1]).eval()
                tf.assign(tensor_bn_variance, bn_weights[2]).eval()

                weights_size = np.product([kernel, kernel, in_ch, filter_num])
                conv_weights = np.ndarray(
                    shape=[filter_num, in_ch, kernel, kernel],  # [32, 3, 3, 3],
                    dtype='float32',
                    buffer=weights_file.read(weights_size * 4))
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                tf.assign(tensor_conv_w, conv_weights).eval()
            else:  # in [81, 93, 105] only when there is no bn, the conv_b is used
                weights_size = np.product([kernel, kernel, in_ch, filter_num])
                conv_weights = np.ndarray(
                    shape=[filter_num, in_ch, kernel, kernel],  # [32, 3, 3, 3],
                    dtype='float32',
                    buffer=weights_file.read(weights_size * 4))
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])

                if n_class is 80:  # if not coco,ignore the last layer
                    tensor_conv_b = tf.global_variables('layer_' + str(i) + '_conv')[1]
                    tf.assign(tensor_conv_b, conv_bias).eval()
                    tf.assign(tensor_conv_w, conv_weights).eval()
                else:
                    print("ignore layer", i)

    weights_file.close()
    print('writeing into ckpt...')
    saver.save(sess, checkpoint_dir + model_name)
    print('done')
