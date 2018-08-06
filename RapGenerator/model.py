# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import tile_batch
from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell
from tensorflow.contrib.rnn import LSTMCell, GRUCell


class Seq2SeqModel(object):
    # 模型初始化构建
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, mode, use_attention,
                 beam_search, beam_size, cell_type='LSTM', max_gradient_norm=5.0):
        # 初始化参数
        self.learing_rate = learning_rate  # 学习率
        self.embedding_size = embedding_size  # embedding的维度
        self.rnn_size = rnn_size  # 隐层的hidden size大小
        self.num_layers = num_layers  # 隐层层数
        self.word_to_idx = word_to_idx  # {词语：id}的映射字典
        self.vocab_size = len(self.word_to_idx)  # 词库大小
        self.mode = mode  # 'train' or 'decode'
        self.use_attention = use_attention  # 是否使用attention机制
        self.beam_search = beam_search  # 是否使用beam search方法
        self.beam_size = beam_size  # beam search的size,选择最大概率的前size条词语组合。
        self.cell_type = cell_type  # RNN的cell类型，‘LSTM’ or ‘GRU’
        self.max_gradient_norm = max_gradient_norm  # 梯度截断，若梯度过大，将截断到设置的这个最大梯度值，防止梯度爆炸

        # placeholder：此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值. 参数有三：
        # 1.dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
        # 2.shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
        # 3.name：名称。绘制计算图的时候可以显示设置的名称。

        # 定义encoder的输入类型和维度，[None,None]代表还未确定，需根据数据的维度最终确定
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        # 定义encoder的每句输入长度
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        # 定义decoder的输出类型和维度
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        # 定义decoder的每句输出长度
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        # batch size
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        # dropout比例：(1-keep_prob)
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # tf.reduce_max():求最大值。设置输出句子的最大长度。
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        # tf.sequence_mask()张量变换函数：返回一个表示每个单元的前N个位置的mask张量。
        # https://www.w3cschool.cn/tensorflow_python/tensorflow_python-2elp2jns.html
        # 函数参数：
        # 1.lengths：整数张量，其所有值小于等于maxlen。
        # 2.maxlen：标量整数张量，返回张量的最后维度的大小；默认值是lengths中的最大值。
        # 3.dtype：结果张量的输出类型。
        # 4.name：操作的名字。
        # 函数返回值：形状为lengths.shape + (maxlen,)的mask张量，投射到指定的dtype。
        # 函数中可能存在的异常：ValueError：如果maxlen不是标量。
        # 这里用来得到targets填充后每句每个位置是否有词的mask，有为True，无为False。
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32,
                                     name='masks')

        # embedding矩阵,encoder和decoder共用该词向量矩阵
        # tf.get_variable(name,  shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式,不写代表随机初始化。
        self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])

        # 网络图结构
        self.__graph__()
        # 模型保存
        self.saver = tf.train.Saver()

    def __graph__(self):

        # encoder
        encoder_outputs, encoder_state = self.encoder()

        # decoder
        with tf.variable_scope('decoder'):
            encoder_inputs_length = self.encoder_inputs_length
            if self.beam_search:
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                print("use beamsearch decoding..")
                # 将encoder的输出复制beam_size份。
                encoder_outputs = tile_batch(encoder_outputs, multiplier=self.beam_size)
                # 将隐藏层状态复制beam_size份，隐层状态包括h和c两个，所以应用lambda表达式。
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size),
                                                   encoder_state)
                # 将encoder的输入长度复制bea_size份。
                encoder_inputs_length = tile_batch(encoder_inputs_length, multiplier=self.beam_size)

            # 定义要使用的attention机制。
            # 使用Bahdanau Attention
            attention_mechanism = BahdanauAttention(num_units=self.rnn_size,  # 隐层的维度
                                                    memory=encoder_outputs,  # encoder的输出
                                                    # memory的mask，通过句子长度判断结尾。
                                                    memory_sequence_length=encoder_inputs_length)
            # 定义decoder阶段要是用的RNNCell，然后为其封装attention wrapper
            decoder_cell = self.create_rnn_cell()
            # AttentionWrapper()用于封装带attention机制的RNN网络
            decoder_cell = AttentionWrapper(cell=decoder_cell,  # decoder的网络
                                            attention_mechanism=attention_mechanism,  # attention实例
                                            attention_layer_size=self.rnn_size,  # TODO：哪个维度
                                            name='Attention_Wrapper'  # 该AttentionWrapper名字
                                            )
            # 如果使用beam_seach则batch_size = self.batch_size * self.beam_size
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            # 定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            # zero_state()先全部初始化为0,再clone()将encoder的最后一个隐层状态初始化为当前decoder的隐层状态
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size,
                                                            dtype=tf.float32).clone(cell_state=encoder_state)
            # 一个全连接层作为输出层，softmax输出为vocab_size，相当于多分类。
            # tf.truncated_normal_initializer()生成截断的正太分布。mean参数指明均值，stddev参数指明方差。
            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(
                mean=0.0,
                stddev=0.1))

            # 如果是训练截断
            if self.mode == 'train':
                #
                self.decoder_outputs = self.decoder_train(decoder_cell, decoder_initial_state, output_layer)
                # loss
                self.loss = sequence_loss(logits=self.decoder_outputs, targets=self.decoder_targets, weights=self.mask)

                # summary
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()

                # optimizer
                optimizer = tf.train.AdamOptimizer(self.learing_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            elif self.mode == 'decode':
                self.decoder_predict_decode = self.decoder_decode(decoder_cell, decoder_initial_state, output_layer)

    def encoder(self):
        '''
        创建模型的encoder部分
        :return: encoder_outputs: 用于attention，batch_size*encoder_inputs_length*rnn_size
                 encoder_state: 用于decoder的初始化状态，batch_size*rnn_size
        '''
        # 实验命名空间‘encoder’，TODO：在这里有什么用？
        with tf.variable_scope('encoder'):
            # 创建网络结构
            encoder_cell = self.create_rnn_cell()
            # tf.nn.embedding_lookup(params,ids):在参数params中查找ids所对应的表示.
            # 把encoder的输入的每个词id对应到embedding的词向量表示。
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            # tf.nn.dynamic_rnn()：跳过padding部分的计算，减少计算量。
            # TODO：返回的是什么东西？输出和最后的状态?
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, sequence_length=
            self.encoder_inputs_length, dtype=tf.float32)
            #
            return encoder_outputs, encoder_state

    def decoder_train(self, decoder_cell, decoder_initial_state, output_layer):
        '''
        创建train的decoder部分
        :param encoder_outputs: encoder的输出
        :param encoder_state: encoder的state
        :return: decoder_logits_train: decoder的predict
        '''
        ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<GO>']), ending], 1)
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)

        training_helper = TrainingHelper(inputs=decoder_inputs_embedded,
                                         sequence_length=self.decoder_targets_length,
                                         time_major=False, name='training_helper')
        training_decoder = BasicDecoder(cell=decoder_cell,
                                        helper=training_helper,
                                        initial_state=decoder_initial_state,
                                        output_layer=output_layer)
        decoder_outputs, _, _ = dynamic_decode(decoder=training_decoder,
                                               impute_finished=True,
                                               maximum_iterations=self.max_target_sequence_length)
        decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
        return decoder_logits_train

    def decoder_decode(self, decoder_cell, decoder_initial_state, output_layer):
        start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<GO>']
        end_token = self.word_to_idx['<EOS>']

        if self.beam_search:
            inference_decoder = BeamSearchDecoder(cell=decoder_cell,
                                                  embedding=self.embedding,
                                                  start_tokens=start_tokens,
                                                  end_token=end_token,
                                                  initial_state=decoder_initial_state,
                                                  beam_width=self.beam_size,
                                                  output_layer=output_layer)
        else:
            decoding_helper = GreedyEmbeddingHelper(embedding=self.embedding,
                                                    start_tokens=start_tokens,
                                                    end_token=end_token)
            inference_decoder = BasicDecoder(cell=decoder_cell,
                                             helper=decoding_helper,
                                             initial_state=decoder_initial_state,
                                             output_layer=output_layer)

        decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder, maximum_iterations=50)
        if self.beam_search:
            decoder_predict_decode = decoder_outputs.predicted_ids
        else:
            decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
        return decoder_predict_decode

    def create_rnn_cell(self):
        '''
        创建网络结构
        :return: cell: 一个多层RNN网络
        '''

        def single_rnn_cell():
            # 根据参数选择的cell_type创建一层GRU或者LSTM。
            single_cell = GRUCell(self.rnn_size) if self.cell_type == 'GRU' else LSTMCell(self.rnn_size)
            # Dropout,每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值.
            # 参数input_keep_prob,output_keep_prob等，分别控制输入和输出的dropout概率。
            # Dropout只能是层与层之间（输入层与LSTM1层、LSTM1层与LSTM2层）的Dropout；同一个层里面，T时刻与T+1时刻是不会Dropout的。
            basiccell = DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            return basiccell

        # 叠加多层
        cell = MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def train(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 0.5,
                     self.batch_size: len(batch.encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def eval(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict
