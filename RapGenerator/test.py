# -*- coding: utf-8 -*-
# @Time    : 18-8-4上午11:20
# @Author  : 石头人m
# @File    : test.py


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

a1 = tf.get_variable(name='a1', shape=[2, 3])
a2 = tf.get_variable(name='a2', shape=[2, 3], initializer=tf.random_normal_initializer(mean=0, stddev=1))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(a1))
    print(sess.run(a2))
