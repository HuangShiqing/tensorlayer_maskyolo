import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import *
from tensorflow.contrib import layers

# from tensorlayer.deprecation import deprecated_alias
# import tensorlayer._logging as logging
import logging as log

Gb_all_layer_out = list()

logger = log.getLogger('tensorlayer')
logger.setLevel(level=log.ERROR)


def conv2d_unit(prev_layer, filters, kernels, act='leaky', strides=1, name='0', bn=True):
    input_size = prev_layer.outputs.get_shape().as_list()[1]
    input_ch = prev_layer.outputs.get_shape().as_list()[3]

    if act is 'liner':
        act = tf.identity
    elif act is 'leaky':
        act = lambda x: tl.act.lrelu(x, 0.1)

    if strides > 1:
        network = ZeroPad2d(prev_layer, padding=((1, 0), (1, 0)), name='layer_' + name + '_pad')

    network = Conv2dLayer(
        prev_layer=prev_layer if strides is 1 else network,
        act=tf.identity if bn is True else act,  # tf.nn.leaky_relu,
        shape=(kernels, kernels, input_ch, filters),
        strides=(1, strides, strides, 1),
        padding='SAME' if strides is 1 else 'VALID',
        b_init=None if bn is True else tf.constant_initializer(value=0.0),
        name='layer_' + name + '_conv',
        W_init_args={'regularizer': layers.l2_regularizer(5e-4)})
    if bn is True:
        network = BatchNormLayer(network, epsilon=1e-3, act=lambda x: tl.act.lrelu(x, 0.1), is_train=False,
                                 name='layer_' + name + '_bn')

    out_size = network.outputs.get_shape().as_list()[1]
    out_ch = network.outputs.get_shape().as_list()[3]

    Gb_all_layer_out.append(network.outputs)
    print('   {:3} conv     {:4}  {} x {} / {}   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(name, filters,
                                                                                                      kernels, kernels,
                                                                                                      strides,
                                                                                                      input_size,
                                                                                                      input_size,
                                                                                                      input_ch,
                                                                                                      out_size,
                                                                                                      out_size,
                                                                                                      out_ch))
    return network


class ResLayer(Layer):
    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)
    def __init__(self, prev_layer=None, name='0', res=None):
        super(ResLayer, self).__init__(prev_layer=prev_layer, name='layer_' + name + '_res')

        input_size = prev_layer.outputs.get_shape().as_list()[1]
        input_ch = prev_layer.outputs.get_shape().as_list()[3]

        self.inputs = prev_layer.outputs

        out = Gb_all_layer_out[res]

        with tf.variable_scope('res' + name):
            # self.outputs = tf.concat([o for o in out], 3)
            self.outputs = out + self.inputs

        self.all_layers.append(self.outputs)
        Gb_all_layer_out.append(self.outputs)
        print('   {:3} res   {:2}                   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(name, str(res),
                                                                                                         input_size,
                                                                                                         input_size,
                                                                                                         input_ch,
                                                                                                         input_size,
                                                                                                         input_size,
                                                                                                         input_ch))


def detection(prev_layer, name):
    Gb_all_layer_out.append(prev_layer.outputs)
    print(
        '   {:3} detection'.format(name))


class RouteLayer(Layer):
    # @deprecated_alias(layer='prev_layer', end_support_version=1.9)
    def __init__(self, prev_layer=None, routes=None, name='0'):
        super(RouteLayer, self).__init__(prev_layer=prev_layer, name='layer_' + name + '_route')

        layers = [Gb_all_layer_out[i] for i in routes]

        with tf.variable_scope('layer_' + name + '_route'):
            if len(layers) > 1:
                self.outputs = tf.concat([layers[0], layers[1]], axis=-1)
            else:
                self.outputs = layers[0]  # only one layer to route

        self.all_layers.append(self.outputs)

        Gb_all_layer_out.append(self.outputs)
        print('   {:3} route   {}'.format(name, routes))


def upsample(prev_layer, scale, name='0'):
    input_size = prev_layer.outputs.get_shape().as_list()[1]
    input_ch = prev_layer.outputs.get_shape().as_list()[3]

    out_net = UpSampling2dLayer(prev_layer, (scale, scale), method=1, is_scale=True, name='layer_' + name + '_upsample')

    out_size = out_net.outputs.get_shape().as_list()[1]
    out_ch = out_net.outputs.get_shape().as_list()[3]

    Gb_all_layer_out.append(out_net.outputs)
    print('   {:3} upsample           {:4}x   {:3} x {:3} x {:4}   ->   {:3} x {:3} x {:4}'.format(name, scale,
                                                                                                   input_size,
                                                                                                   input_size,
                                                                                                   input_ch,
                                                                                                   out_size,
                                                                                                   out_size,
                                                                                                   out_ch))
    return out_net

# def residual_block(inputs, filters, name='0'):
#     x = conv2d_unit(inputs, filters, 1, name=name)
#     x = conv2d_unit(x, 2 * filters, 3, name=str(int(name) + 1))
#     x = ResLayer(x, name=str(int(name) + 2))
#
#     return x
#
#
# def stack_residual_block(inputs, filters, n, name='0'):
#     """Stacked residual Block
#     """
#     x = residual_block(inputs, filters, name=name)
#
#     for i in range(n - 1):
#         x = residual_block(x, filters, name=str(int(name) + 3 * (i + 1)))
#
#     return x
