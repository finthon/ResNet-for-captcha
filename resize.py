# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     resize
   Description :
   Author :       DrZ
   date：          2019/1/3
-------------------------------------------------
   Change Activity:
                   2019/1/3:
-------------------------------------------------
"""
import os
import numpy as np
from PIL import Image


# 验证码路径
captcha_path = r'F:\resnet_for_captcha\captcha4'
# 修改后图片存放路径
save_path = r'F:\resnet_for_captcha\1resize\resize_path'
for i in os.listdir(captcha_path):
    img = Image.open(os.path.join(captcha_path, i))
    arr = np.array(img) * 255       # 注意这里的np.array(img)是布尔值，之前二值化遗留下来的问题
    im = Image.fromarray(arr)
    im = im.resize((224, 224))
    arr = np.array(im)
    x = np.zeros([224, 224, 3])    # 创建一个224*224*3的矩阵
    for ii in range(224):
        for jj in range(224):
            x[ii, jj, :] = arr[ii, jj]
    im = Image.fromarray(x.astype('uint8'))     # 图片矩阵使用该格式
    im.save(os.path.join(save_path, i))