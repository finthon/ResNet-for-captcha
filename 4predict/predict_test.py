# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict_test
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
from PIL import Image
import os
import numpy as np


# 向量转成标签名字
def vec2name(vec):
    name = []
    for i in vec:
        a = chr(i + 97)
        name.append(a)
    return "".join(name)


model_dir = r'F:\resnet_for_captcha\3train\model\train.model-140000'
x = tf.placeholder(tf.float32, [None, 224, 224, 3])

pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=6 * 26, is_training=True)
predict = tf.reshape(pred, [-1, 6, 26])
max_idx_p = tf.argmax(predict, 2)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_dir)
    test_dir = r'F:\resnet_for_captcha\test'
    for pic in os.listdir(test_dir):
        pic_path = os.path.join(test_dir, pic)
        img = Image.open(pic_path)
        arr = np.array(img) * 255
        im = Image.fromarray(arr)
        im = im.resize((224, 224))
        arr = np.array(im)
        xx = np.zeros([224, 224, 3])
        for ii in range(224):
            for jj in range(224):
                xx[ii, jj, :] = arr[ii, jj]
        img1 = Image.fromarray(xx.astype('uint8'))
        img2 = tf.reshape(img1, [1, 224, 224, 3])
        img3 = tf.cast(img2, tf.float32) / 255.0

        name = os.path.splitext(pic)[0]

        b_image = sess.run(img3)
        t_label = sess.run(max_idx_p, feed_dict={x: b_image})
        vec = t_label[0].tolist()
        predict_text = vec2name(vec)
        print('真实值：{}   预测值：{}'.format(name, predict_text))