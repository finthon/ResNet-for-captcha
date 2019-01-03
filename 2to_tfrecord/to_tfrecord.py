# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     to_tfrecord
   Description :
   Author :       DrZ
   date：          2019/1/3
-------------------------------------------------
   Change Activity:
                   2019/1/3:
-------------------------------------------------
"""
import os
import tensorflow as tf
from PIL import Image
import numpy as np


# 将验证码的名字转换成数组，one hot编码
def name2vec(name):
    vector = np.zeros(6 * 26)
    for i, c in enumerate(name):
        idx = i * 26 + ord(c) - 97
        vector[idx] = 1
    return vector


# 图片路径
cwd = r'F:\resnet_for_captcha\1resize\resize_path'

# 文件路径
file_path = r'F:\resnet_for_captcha\2to_tfrecord\tfrecord'

# 存放图片个数
bestnum = 1000

# 第几个图片
num = 0

# 第几个TFRecord文件
recordfilenum = 0

# tfrecords格式文件名
ftrecordfilename = ("train.tfrecords-%.3d" % recordfilenum)
writer = tf.python_io.TFRecordWriter(os.path.join(file_path, ftrecordfilename))

for i in os.listdir(cwd):
    num += 1
    print(num)
    if num > bestnum:
        num = 1
        recordfilenum += 1
        ftrecordfilename = ("train.tfrecords-%.3d" % recordfilenum)
        writer = tf.python_io.TFRecordWriter(os.path.join(file_path, ftrecordfilename))

    name = os.path.splitext(i)[0]
    name_vec = name2vec(name).tobytes()         # 转成二进制格式
    img = Image.open(os.path.join(cwd, i), 'r')
    img_raw = img.tobytes()
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name_vec])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
    writer.write(example.SerializeToString())
writer.close()
