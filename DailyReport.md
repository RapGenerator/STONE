# 2018-08-14
穿插课程:终端深度学习基础,挑战和工程实践--张弥

看SeqGAN源码.

互联网趋势下的消费升级--郎春晖

# 2018-08-13
初步了解SeqGAN.

# 2018-08-10
押韵strict各韵母数统计.
参观美团.

# 2018-08-09
数据处理,得到各种押韵的歌词.

# 2018-08-08
押韵统计,包括排韵 隔行韵 交韵 抱韵.

# 2018-08-07
### Chinese Poetry Generation with Planning based Neural Network:

user's intent -> 4 keyword -> line1 ->line2--line4

#### topic --> lyrics:
user's intent:one keyword or a set of keywords;one sectence or one document.

to several keywords:expansion and extraction:

#### expansion:
    co-occurrence
    RNNLM
    knowledge
#### extraction:
    TextRank
    TF-IDF

### Skip-Thought Vectors:
One encoder generates two decoders


# 2018-08-06
代码注解基本完成

# 2018-08-03
注解了部分代码

# 2018-08-02
### 用基于注意力机制的seq2seq神经网络进行翻译:
http://pytorch.apachecn.org/cn/tutorials/intermediate/seq2seq_translation_tutorial.html

#### 编码器
seq2seq网络的编码器是一个RNN,它为输入句子中的每个单词输出一些值.
对于每个输入字,编码器输出一个向量和一个隐藏状态,并将隐藏状态用于下一个输入字.

![avatar](http://pytorch.apachecn.org/cn/tutorials/_images/encoder-network.png)

#### 简单的解码器
在解码的每一步,解码器都被赋予一个输入指令和隐藏状态. 初始输入指令字符串开始的``<SOS>``指令,第一个隐藏状态是上下文向量(编码器的最后隐藏状态).

![avatar](http://pytorch.apachecn.org/cn/tutorials/_images/decoder-network.png)

#### 注意力解码器
注意力允许解码器网络针对解码器自身输出的每一步”聚焦”编码器输出的不同部分.
首先我们计算一组注意力权重. 这些将被乘以编码器输出矢量获得加权的组合.
结果应该包含关于输入序列的特定部分的信息, 从而帮助解码器选择正确的输出单词.

使用解码器的输入和隐藏状态作为输入,利用另一个前馈层 ``attn``计算注意力权重,
由于训练数据中有各种大小的句子,为了实际创建和训练此层,
我们必须选择最大长度的句子(输入长度,用于编码器输出),以适用于此层.
最大长度的句子将使用所有注意力权重,而较短的句子只使用前几个.

![avatar](http://pytorch.apachecn.org/cn/tutorials/_images/attention-decoder-network.png)


# 2018-08-01
### Word2Vec:

one-hot问题：稀疏，任何两个词之间的余弦相似度为0。

word2vec：
* 跳字模型：skip-gram
* 连续词袋模型：CBOW（continuous bag of words）

两种高效的训练方法：
* 负采样
* 层序softmax

#### skip-gram:中心词，背景词，时间窗

最大化联合概率：中心词周围的背景词的概率

计算量大：每一次的时间复杂度为O(|V|),V:词典

#### CBOW：用一个中心词周围的词来预测该中心词

#### 负采样：O(|V|) --> O(k)
以skip-gram为例：
softmax考虑了背景词可能是词典中的任一词==>

假设中心词Wc生成背景词Wo由一下两个相互独立事件联合组成的近似：

* 中心词 wc 和背景词 wo 同时出现时间窗口。
* 中心词 wc 和第 1 个噪声词 w1 不同时出现在该时间窗口（噪声词 w1 按噪声词分布 ℙ(w) 随机生成，且假设一定和 wc 不同时出现在该时间窗口）。
* …
* 中心词 wc 和第 K 个噪声词 wK 不同时出现在该时间窗口（噪声词 wK 按噪声词分布 ℙ(w) 随机生成，且假设一定和 wc 不同时出现在该时间窗口）。

#### 层序softmax：O(|V|) --> O(log(|V|))
二叉树:树的每个叶子节点代表着词典V中的每个词。

### GloVe
GloVe 使用了词与词之间的共现（co-occurrence）信息。

我们定义 X 为共现词频矩阵，其中元素 xij 为词 j 出现在词 i 的背景的次数。

共现概率比值

用词向量表达共现概率比值

### fastText


# 2018-07-31

1、RNN梯度推导

https://ilewseu.github.io/2017/12/30/RNN%E7%AE%80%E5%8D%95%E6%8E%A8%E5%AF%BC/

2、RNN存在什么样的问题？

长期依赖问题：虽然简单循环网络可从理论上可以建立长时间间隔的状态之间的依赖关系，但是由于梯度爆炸或消失的存在，实际上只能学习到短期的依赖关系。本质原因就是因为矩阵高次幂导致的。

3、RNN的每一个time step的hidden state和最后一个hidden state分别可以用在什么场景？

多对多、多对一

4、LSTM如何解决RNN的问题？

图片: https://ws1.sinaimg.cn/large/005BVyzmly1fotnatxsm7j30jg07bjsi.jpg
图片: https://www.zhihu.com/equation?tex=c%5Et+%3D+f%5Et+%5Codot+c%5E%7Bt-1%7D+%2B+i%5Et+%5Codot+g%5Et
图片: https://pic4.zhimg.com/80/v2-8eb676e7c1bac3eb131d8e0bf2f7db5b_hd.png
公式里其余的项不重要，这里就用省略号代替了。可以看出当 图片: https://www.zhihu.com/equation?tex=f%5Et+%3D+1 时，就算其余项很小，梯度仍然可以很好导到上一个时刻，此时即使层数较深也不会发生 Gradient Vanish 的问题；当 图片: https://www.zhihu.com/equation?tex=f%5Et+%3D+0 时，即上一时刻的信号不影响到当前时刻，则梯度也不会回传回去； 图片: https://www.zhihu.com/equation?tex=f%5Et 在这里也控制着梯度传导的衰减程度，与它 Forget Gate 的功能一致。

5、LSTM三个门有什么作用，如何计算，如何梯度回传？

https://www.jianshu.com/p/dcec3f07d3b5
遗忘门：决定我们会从细胞状态中丢弃什么信息，该门会读取ht−1和xt，输出一个在0到1之间的数字给Ct−1，0表示完全放弃，1表示完全保留。
输入门：决定让多少新的信息加入到cell状态中来。
输出门：我们需要确定输出什么样的值。

梯度回传：
一个是沿时间的反向传播，即从当前 t 时刻开始，计算每个时刻的误差项；
一个是将误差项向上一层传播。
图片: https://upload-images.jianshu.io/upload_images/1667471-9da34d2b2b475e7a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700

6、LSTM有什么变种（2-3个），GRU有什么改变和提升？

https://www.cnblogs.com/wuxiangli/p/7096392.html
peephole connection：让 门层 也会接受细胞状态的输入
使用 coupled 忘记和输入门：不同于lstm是分开确定什么忘记和需要添加什么新的信息，这里是一同做出决定。我们仅仅会当我们将要输入在当前位置时忘记。我们仅仅输入新的值到那些我们已经忘记旧的信息的那些状态 。
GRU：
图片: https://ws1.sinaimg.cn/large/005BVyzmly1fotomifeglj31eq0fo40v.jpg
GRU 少一个门，同时少了细胞状态Ct;
在 LSTM 中，通过遗忘门和传入门控制信息的保留和传入；GRU 则通过重置门来控制是否要保留原来隐藏状态的信息，但是不再限制当前信息的传入;
在 LSTM 中，虽然得到了新的细胞状态Ct，但是还不能直接输出，而是需要经过一个过滤的处理：ht=ot×tanh（Ct）。同样，在 GRU 中, 虽然我们也得到了新的隐藏状态hthat， 但是还不能直接输出，而是通过更新门来控制最后的输出：ht=(1−zt)∗ht−1+zt∗ĥ t

7、LSTM中都用了哪些激活函数？为什么用这些激活函数？可以换成其他的么？

遗忘门：sigmod函数（0到1之间的值）。
输入门：sigmod函数和tanh函数。
输出门：sigmod函数和tanh函数。

8、Seq2Seq中的encoder输入输出是什么？decoder的输入输出是什么？

encoder:输入的序列，输出指定长度的向量。
decoder:输入encoder编码后输出的向量，输出一个新的序列。

9、Seq2Seq的loss function是什么？有其他可用的loss function么？为什么？

最大似然条件概率为损失函数。

10、Seq2Seq在文本生成任务中，有什么常用的方法？

11、Seq2Seq在解码时有哪些常用的方法？

<bos>、<eos>
s0的初始化：tanh(Wh1)

12、BeamSearch可以用在我们的任务中么?会在哪些问题中适用，或者特别适合哪些任务？

https://www.zhihu.com/question/54356960

13、Bleu score是最适合我们的任务的评价方式么？有什么更好的评价指标？Seq2Seq的评价指标都有什么？

BLEU的全名为：bilingual evaluation understudy，即：双语互译质量评估辅助工具。它是用来评估机器翻译质量的工具。BLEU实质是对两个句子的共现词频率计算，容易陷入常用词和短译句的陷阱中，而给出较高的评分值。
优点很明显：方便、快速、结果有参考价值 
缺点也不少，主要有： 
1).不考虑语言表达（语法）上的准确性； 
2).测评精度会受常用词的干扰； 
3).短译句的测评精度有时会较高； 
4).没有考虑同义词或相似表达的情况，可能会导致合理翻译被否定；

14、Attention都有哪几种？哪种适用于哪些情况？

https://blog.csdn.net/m0epNwstYk4/article/details/81073986
BahdanauAttention、LuongAttention

15、Attention适合我们的任务吗?帮助我们解决了哪些问题？

主题生成：不同的注意力？

16、Attention是如何计算的？在文本生成的任务中是如何计算的？

对各个时刻ht不同的注意力，加权平均。

