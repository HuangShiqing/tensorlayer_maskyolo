import tensorflow as tf
import tensorlayer as tl
import numpy as np

from varible import *
from data import data_generator, read_xml
from model import infenence
import time
import os


# y_true [16,52,52]
def yolo_loss(y_pred, y_true):
    # out_c = tf.sigmoid(y_pred[..., 0])  # tf.expand_dims(tf.sigmoid(y_pred[..., 0]), axis=-1)
    # out_class = tf.sigmoid(y_pred[..., 1])  # * len(Gb_labels)  # tf.expand_dims(tf.sigmoid(y_pred[..., 1]))

    object_mask = tf.where(y_true != 0, tf.ones_like(y_true), tf.zeros_like(y_true))
    noobject_mask = 1 - object_mask

    true_c = object_mask
    # true_class = y_true / (len(Gb_labels)-1)
    y_true = tf.cast(y_true, tf.uint8)
    true_class = tf.one_hot(y_true, 21)

    loss_c = tf.reduce_sum(
        object_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred[..., 0], labels=true_c)) / Gb_batch_size
    loss_class = tf.reduce_sum(object_mask * tf.nn.softmax_cross_entropy_with_logits(logits=y_pred[..., 1:],
                                                                                     labels=true_class)) / Gb_batch_size
    loss_sum = loss_c + loss_class

    tf.summary.scalar('/loss', loss_sum)
    tf.summary.scalar('/loss_c', loss_c)
    tf.summary.scalar('/loss_class', loss_class)

    loss_sum = tf.Print(loss_sum, [loss_c, loss_class])

    # loss_c = tf.reduce_sum(
    #     5 * object_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred[..., 0], labels=true_c)) / Gb_batch_size
    # loss_class_obj = tf.reduce_sum(
    #     5 * object_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred[..., 1],
    #                                                               labels=true_class)) / Gb_batch_size
    # loss_class_noobj = tf.reduce_sum(
    #     noobject_mask * tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred[..., 1],
    #                                                             labels=true_class)) / Gb_batch_size
    # loss_sum = loss_c + loss_class_obj + loss_class_noobj
    # tf.summary.scalar('/loss', loss_sum)
    # tf.summary.scalar('/loss_c', loss_c)
    # tf.summary.scalar('/loss_class_obj', loss_class_obj)
    # tf.summary.scalar('/loss_class_noobj', loss_class_noobj)
    # loss_sum = tf.Print(loss_sum, [loss_c, loss_class_obj, loss_class_noobj])
    return loss_sum


def training(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def main():
    # n_class = len(Gb_label)
    # model_name = Gb_model_name
    log_dir = Gb_ckpt_dir
    final_dir = Gb_ckpt_dir
    save_frequency = Gb_save_frequency
    batch_size = Gb_batch_size
    learning_rate = Gb_learning_rate

    annotations_path = Gb_ann_path
    pick = Gb_labels
    chunks = read_xml(annotations_path, pick)

    n_epoch = Gb_epoch
    n_step_epoch = int(len(chunks) / batch_size)
    # n_step = n_epoch * n_step_epoch

    input_pb = tf.placeholder(tf.float32, [None, 416, 416, 3])
    y_true_pb = tf.placeholder(tf.float32, [None, 52, 52])
    net_out = infenence(input_pb)
    loss_op = yolo_loss(net_out, y_true_pb)
    train_op = training(loss_op, learning_rate)

    # varis = tf.global_variables()
    # var_to_restore = [val for val in varis if 'Adam' not in val.name and 'optimizer' not in val.name]
    # saver = tf.train.Saver(var_to_restore)
    saver = tf.train.Saver(max_to_keep=100)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # if tf.train.get_checkpoint_state('./ckpt2/'):  # 确认是否存在
        #     saver.restore(sess, './ckpt2/' + "ep094-step17000-loss61286.484")
        #     print("load ok!")
        # else:
        #     print("ckpt文件不存在")

        # tensor = tf.global_variables('layer_0_conv')
        # b = sess.run(tensor)

        train_writer = tf.summary.FileWriter(log_dir, sess.graph)
        step = 0
        min_loss = 10000000
        for epoch in range(n_epoch):
            step_epoch = 0
            # TODO shuffle chunks
            data_yield = data_generator(chunks)

            # train_loss, n_batch = 0, 0
            for origin_img_sizeds, segment_datas in data_yield:
                step += 1
                step_epoch += 1
                start_time = time.time()

                summary_str, loss, _ = sess.run([summary_op, loss_op, train_op],
                                                feed_dict={input_pb: origin_img_sizeds, y_true_pb: segment_datas})
                train_writer.add_summary(summary_str, step)

                # 每step打印一次该step的loss
                print("Loss %fs  : Epoch %d  %d/%d: Step %d  took %fs" % (
                    loss, epoch, step_epoch, n_step_epoch, step, time.time() - start_time))

                if step % 1000 == 0 and loss < min_loss:
                    print("Save model " + "!" * 10)
                    save_path = saver.save(sess,
                                           final_dir + 'ep{0:03d}-step{1:d}-loss{2:.3f}'.format(epoch, step, loss))
                    min_loss = loss

                if step % save_frequency == 0:
                    if step != save_frequency:
                        os.remove(final_dir + temp + '.data-00000-of-00001')
                        os.remove(final_dir + temp + '.index')
                        os.remove(final_dir + temp + '.meta')

                    print("Save model " + "!" * 10)
                    save_path = saver.save(sess,
                                           final_dir + 'ep{0:03d}-step{1:d}-loss{2:.3f}'.format(epoch, step, loss))
                    temp = 'ep{0:03d}-step{1:d}-loss{2:.3f}'.format(epoch, step, loss)


if __name__ == '__main__':
    main()
