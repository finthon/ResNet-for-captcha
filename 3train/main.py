# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
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
import tensorflow.contrib.slim.nets as nets
import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


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
    img = tf.cast(img, tf.float32) / 255.0    # 归一化
    return img, label


def main():
    save_dir = r"F:\resnet_for_captcha\3train\model\train.model"
    batch_size_ = 2
    lr = tf.Variable(0.0001, dtype=tf.float32)
    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y_ = tf.placeholder(tf.float32, [None, 6 * 26])

    tfrecord_path = r'F:\resnet_for_captcha\2to_tfrecord\tfrecord'
    train_list = []
    for file in os.listdir(tfrecord_path):
        train_list.append(os.path.join(tfrecord_path, file))

    min_after_dequeue = 1000
    # 随机打乱顺序
    img, label = read_and_decode_tfrecord(train_list)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], num_threads=2, batch_size=batch_size_,
                                                    capacity=min_after_dequeue + 3 * batch_size_,
                                                    min_after_dequeue=min_after_dequeue)

    pred, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=6 * 26, is_training=True)
    pred = tf.reshape(pred, shape=[-1, 6 * 26])
    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    # 准确度
    predict = tf.reshape(pred, [-1, 6, 26])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y_, [-1, 6, 26]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, r'F:\resnet_for_captcha\3train\model\train.model-60000')
        # 创建一个协调器，管理线程
        coord = tf.train.Coordinator()
        # 启动QueueRunner,此时文件名队列已经进队
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        i = 60000
        cycle_num = 0
        while True:
            i += 1
            b_image, b_label = sess.run([img_batch, label_batch])
            _, loss_ = sess.run([optimizer, loss], feed_dict={x: b_image, y_: b_label})
            if i % 20 == 0:
                print('step: {}, loss: {}'.format(i, loss_))
            if i % 100 == 0:
                _loss, acc_train = sess.run([loss, accuracy], feed_dict={x: b_image, y_: b_label})
                print('--------------------------------------------------------')
                print('step: {}  train_acc: {}  loss: {}'.format(i, acc_train, _loss))
                print('--------------------------------------------------------')
            if i % 20000 == 0:
                saver.save(sess, save_dir, global_step=i)
                cycle_num += 1
            if cycle_num == 10:
                break
        coord.request_stop()
        # 其他所有线程关闭之后，这一函数才能返回
        coord.join(threads)


if __name__ == '__main__':
    # 运行时间
    starttime = datetime.datetime.now().timestamp()
    main()
    endtime = datetime.datetime.now().timestamp()
    print(starttime)
    print(endtime)
    run_hour = (endtime - starttime) / 3600
    print('共运行{}小时！'.format(run_hour))