# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from data_helpers import *
from model import Seq2SeqModel
import math

if __name__ == '__main__':

    # 超参数
    rnn_size = 1024  # 隐层的hidden size大小
    num_layers = 2  # 隐层的层数
    embedding_size = 1024  # embedding的维度，一般和rnn_size一样，大多设置50（可以捕捉10万词信息）～200。
    batch_size = 128  # batch size
    learning_rate = 0.0001  # 学习率
    epochs = 5000  # 训练循环次数
    sources_txt = 'data/sources.txt'  # 输入文本
    targets_txt = 'data/targets.txt'  # 输出，和sources一行一行对应
    model_dir = 'model/'  # 模型输出路径

    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, _ = create_dic_and_map(sources, targets)

    # 在tensorflow中数据流图中的Op（operation，节点）在得到执行之前,必须先创建Session对象,Session对象负责着图中所有Op的执行.
    # Session 对象创建时有三个可选参数:
    # 1.target：在不是分布式中使用Session对象时,该参数默认为空
    # 2.graph：指定了Session对象中加载的Graph对象,如果不指定的话默认加载当前默认的数据流图,但是如果有多个图,就需要传入加载的图对象
    # 3.config：Session对象的一些配置信息,CPU/GPU使用上的一些限制,或者一些优化设置
    with tf.Session() as sess:
        # 构建Seq2Seq模型
        model = Seq2SeqModel(rnn_size, num_layers, embedding_size, learning_rate, word_to_id, mode='train',
                             use_attention=True, beam_search=False, beam_size=5, cell_type='LSTM',
                             max_gradient_norm=5.0)
        # tf全局变量初始化
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            batches = getBatches(sources_data, targets_data, batch_size)
            for nextBatch in batches:
                loss, summary = model.train(sess, nextBatch)
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print("----- Loss %.2f -- Perplexity %.2f" % (loss, perplexity))
            model.saver.save(sess, model_dir + 'seq2seq.ckpt')
