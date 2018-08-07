# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
from RapGenerator.data_helpers import *
from RapGenerator.model import Seq2SeqModel
import sys


def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    # 对每句话进行输出
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))


if __name__ == '__main__':

    # 超参数
    rnn_size = 1024
    num_layers = 2
    embedding_size = 1024
    batch_size = 128
    learning_rate = 0.0001
    epochs = 100
    steps_per_checkpoint = 5
    sources_txt = 'data/sources.txt'
    targets_txt = 'data/targets.txt'
    model_dir = 'model/'

    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, id_to_word = create_dic_and_map(sources, targets)

    with tf.Session() as sess:
        model = Seq2SeqModel(rnn_size, num_layers, embedding_size, learning_rate, word_to_id, mode='decode',
                             use_attention=True, beam_search=True, beam_size=5, cell_type='LSTM', max_gradient_norm=5.0)
        # 加载训练好的模型
        ckpt = tf.train.get_checkpoint_state(model_dir)
        # 如果模型存在就加载训练好的模型
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:  # 模型不存在报错
            raise ValueError('No such file:[{}]'.format(model_dir))
        # model.saver.restore(sess, model_dir)
        # 输入第一句
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        # 用上一句的输出作为下一句的输入,循环预测输出
        while sentence:
            # 将输入的句子转换为可以输入模型的batch
            batch = sentence2enco(sentence, word_to_id)
            # 模型预测得到句子每个词的id
            predicted_ids = model.infer(sess, batch)
            # 将每个词的id转换为对应的词语并输出
            predict_ids_to_seq(predicted_ids, id_to_word, 5)
            print("> ", "")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
