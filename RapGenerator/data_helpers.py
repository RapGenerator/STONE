# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def load_and_cut_data(filepath):
    '''
    加载数据并分词
    :param filepath: 路径
    :return: data: 分词后的数据
    '''
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = []  # 返回数据集list
        lines = f.readlines()  # 逐行读取文本，返回文本list
        for line in lines:  # 对每一行进行处理
            # line.strip()删除所有开始和结束的空格、制表符、换行和回车。
            # jieba.cut进行分词，cut_all：是否为全模式分词，
            # cut_all=True：全模式，所有可能结果都分出来，可能重复，如‘我来到北京清华大学’=》‘我/来到/北京/清华/清华大学/华大/大学’
            # cut_all=False：默认模式,只生成一种最大可能的分词结果，如‘我来到北京清华大学’=》‘我/来到/北京/清华大学’
            # 所以我们使用默认模式，只返回一种分词结果
            # 返回object Tokenizer.cut对象
            seg_list = jieba.cut(line.strip(), cut_all=False)
            cutted_line = [e for e in seg_list]  # 分词后的结果转每个词list
            data.append(cutted_line)  # 把每一行的分词结果append到data中
    return data


def create_dic_and_map(sources, targets):
    '''
    得到输入和输出的字符映射表
    :param sources:分词后的输入数据列表
           targets:分词后的对应输出数据列表
    :return: sources_data:将词语映射id数字后的sources
             targets_data:将词语映射id数字后的targets
             word_to_id: 字典，汉字到数字的映射
             id_to_word: 字典，数字到汉字的映射
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']  # 特殊字符：填充字符，未知字符，每一行开头，结尾符

    # 得到每个词语的使用频率
    word_dic = {}  # 想要得到的{词语：计数}字典
    for line in (sources + targets):  # 对输入输出所有数据逐行读取
        for character in line:  # 逐词读取每一行进行计数
            # 字典dict的get方法：dict.get(key, default=None)，
            # 参数：key -- 字典中要查找的键。default -- 如果指定键的值不存在时，返回该默认值值。
            # 这里实现的逻辑是从构建的计数词典中找到已经统计的词，对其计数加1,如果还没有，对其数量计数为默认0加1。
            word_dic[character] = word_dic.get(character, 0) + 1

    # 去掉使用频率为1的词
    word_dic_new = []
    # dict.items()方法：返回可遍历的(键, 值) 元组数组。
    for key, value in word_dic.items():  # 对字典的键值对循环，判断每个词的数量。
        if value > 1:
            word_dic_new.append(key)  # 把词的计数大于1的加入新的字典中保存

    # 将字典中的汉字/英文单词映射为数字
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

    # 对特殊字符和计数大于1的词语进行{数字：词语}映射
    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    # 对特殊字符和计数大于1的词语进行{词语：数字}映射
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    # 将sources和targets中的汉字/英文单词映射为数字
    # 字典get方法：dict.get(key, default=None)，参数：key -- 字典中要查找的键。default -- 如果指定键的值不存在时，返回该默认值值。
    # 对sources和targets里的每一行里的每一个词语进行映射，如果没有就映射为<UNK>未知字符。
    sources_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in sources]
    targets_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in targets]

    return sources_data, targets_data, word_to_id, id_to_word


def createBatch(sources, targets):
    # 创建Batch对象
    batch = Batch()
    # 得到每个batch的encoder的输入长度
    batch.encoder_inputs_length = [len(source) for source in sources]
    # 得到每个batch的decoder的输出长度
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    # 每个batch的最大的输入长度
    max_source_length = max(batch.encoder_inputs_length)
    # 每个batch的最大的输出长度
    max_target_length = max(batch.decoder_targets_length)

    for source in sources:
        # 将source进行反序并PAD
        # 反序
        source = list(reversed(source))
        # 填充,对小于最大长度的进行填充
        pad = [padToken] * (max_source_length - len(source))
        # 填充后的当作输入
        batch.encoder_inputs.append(pad + source)

    for target in targets:
        # 将target进行PAD，并添加EOS符号
        # 填充的pad,最大长度减去当前长度再减去1.
        pad = [padToken] * (max_target_length - len(target) - 1)
        # 没一句加一个<EOS>符
        eos = [eosToken] * 1
        # 每一句加一个<EOS>符再加入填充.
        batch.decoder_targets.append(target + eos + pad)

    return batch


def getBatches(sources_data, targets_data, batch_size):
    # 所有训练数据的长度
    data_len = len(sources_data)

    # 根据batch_size循环划分数据
    def genNextSamples():
        for i in range(0, len(sources_data), batch_size):
            yield sources_data[i:min(i + batch_size, data_len)], targets_data[i:min(i + batch_size, data_len)]
    # 将数据划分为多个batch,存在列表中
    batches = []
    # 对数据进行划分
    for sources, targets in genNextSamples():
        # 得到具体的某个batch
        batch = createBatch(sources, targets)
        batches.append(batch)

    return batches


def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    # 分词
    seg_list = jieba.cut(sentence.strip(), cut_all=False)
    cutted_line = [e for e in seg_list]

    # 将每个单词转化为id
    wordIds = []
    for word in cutted_line:
        wordIds.append(word2id.get(word, unknownToken))
    print(wordIds)
    # 调用createBatch构造batch
    batch = createBatch([wordIds], [[]])
    return batch


if __name__ == '__main__':

    sources_txt = 'data/sources.txt'
    targets_txt = 'data/targets.txt'
    keep_rate = 0.6
    batch_size = 128

    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, id_to_word = create_dic_and_map(sources, targets)
    batches = getBatches(sources_data, targets_data, batch_size)

    temp = 0
    for nexBatch in batches:
        if temp == 0:
            print(len(nexBatch.encoder_inputs))
            print(len(nexBatch.encoder_inputs_length))
            print(nexBatch.decoder_targets)
            print(nexBatch.decoder_targets_length)
        temp += 1
