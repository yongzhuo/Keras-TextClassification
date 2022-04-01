# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/13 16:13
# @author   :Mo
# @function :


import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)
from distutils.util import strtobool


# gpu/tf日志的环境, 默认CPU
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")  # "0,1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# 自动化(默认AUTO), 即定义是不是高自由度, 如"CUSTOM"可以高度自定义, 网络架构embedding/graph/loss等均可高度自定义
# 默认使用keras_textclassification.keras

tf_keras = os.environ.get("TF_KERAS", "0")
print(tf_keras)
is_tf_keras = strtobool(tf_keras)


if is_tf_keras:
    import tensorflow as tf
    # Python Import机制备忘-模块搜索路径(sys.path)、嵌套Import、package Import
    sys.modules["keras"] = tf.keras


__version__ = "0.2.0"

