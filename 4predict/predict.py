# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict
   Description :
   Author :       DrZ
   date：          2019/1/3
-------------------------------------------------
   Change Activity:
                   2019/1/3:
-------------------------------------------------
"""
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import os
import numpy as np


def read_and_decode_tfrecord(filename):
    filename_deque = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_deque)
    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.string),
        'img_raw': tf.FixedLenFeature([], tf.string)})
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [6 * 26])
    label = tf.cast(label, tf.float32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) / 255.0      # 归一化
    return img, label


def vec2name(vec):
    name = []
    for i in vec:
        a = chr(i + 97)
        name.append(a)
    return "".join(name)


model_dir = r'F:\resnet_for_captcha\3train\model\train.model-140000'
tfrecord_path = r'F:\resnet_for_captcha\2to_tfrecord\tfrecord'

train_list = []
for file in os.listdir(tfrecord_path):
    train_list.append(os.path.join(tfrecord_path, file))

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, [None, 6 * 26])
batch_size_ = 1
min_after_dequeue = 1000

# 顺序读取
img, label = read_and_decode_tfrecord(train_list)
img_batch, label_batch = tf.train.batch([img, label], num_threads=2, batch_size=batch_size_,
                                                capacity=min_after_dequeue + 3 * batch_size_)

pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=6 * 26, is_training=True)
predict = tf.reshape(pred, [-1, 6, 26])
max_idx_p = tf.argmax(predict, 2)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_dir)
    coord = tf.train.Coordinator()
    # 启动QueueRunner,此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    nn = 0
    count_true_num = 0
    count_false_num = 0
    while True:
        nn += 1
        b_image, b_label = sess.run([img_batch, label_batch])
        pre_index = sess.run(max_idx_p, feed_dict={x: b_image})
        vec = pre_index[0].tolist()
        predict_text = vec2name(vec)
        max_idx_l = np.argmax(np.reshape(b_label, [-1, 6, 26]), 2)
        vec1 = max_idx_l[0].tolist()
        true_text = vec2name(vec1)
        print('{}  真实值：{}   预测值：{}'.format(nn, true_text, predict_text))
        if true_text == predict_text:
            count_true_num += 1
        else:
            count_false_num += 1

        if nn == 3430:
            break
    print('正确：{}  错误：{} 准确率：{}'.format(count_true_num, count_false_num,
                                       count_true_num / (count_true_num + count_false_num)))
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
