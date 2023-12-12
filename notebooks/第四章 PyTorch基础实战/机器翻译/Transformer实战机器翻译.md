# Transformer 实战机器翻译

## 引言

自然语言处理与图像具有显著差异，自然语言处理任务也具有其独特性。相对于 CNN（卷积神经网络）在 CV 中的霸主地位，在很长一段时间里，RNN（循环神经网络）、LSTM（长短期记忆递归神经网络）占据了 NLP 的主流地位。作为针对序列建模的模型，RNN、LSTM 在以序列为主要呈现形式的 NLP 任务上展现出远超 CNN 的卓越性能。​但是 RNN、LSTM 虽然在处理自然语言处理的序列建模任务中得天独厚，却也有着难以忽视的缺陷：

​1. RNN 为单向依序计算，序列需要依次输入、串行计算，限制了计算机的并行计算能力，导致时间成本过高。

​2. RNN 难以捕捉长期依赖问题，即对于极长序列，RNN 难以捕捉远距离输入之间的关系。虽然 LSTM 通过门机制对此进行了一定优化，但 RNN 对长期依赖问题的捕捉能力依旧是不如人意的。

​针对上述两个问题，2017年，Vaswani 等人发表了论文《Attention Is All You Need》，抛弃了传统的 CNN、RNN 架构，提出了一种全新的完全基于 attention 机制的模型——Transformer，解决了上述问题，在较小的时间成本下取得了多个任务的 the-state-of-art 效果，并为自然语言处理任务提供了新的思路。自此，attention 机制进入自然语言处理任务的主流架构，在 Transformer 的基础上，诞生了预训练-微调范式的多种经典模型如 Bert、GPT、T5 等。当然，同样是在 Transformer 的肩膀上，引入了 RLHF 机制、实现了大量参数建模的 ChatGPT 则带领 NLP 进入了全新的大模型时代。但不管是预训练-微调范式的主流模型 Bert，还是大模型时代的主流模型 ChatGPT、LLaMA，Transformer 都是其最坚实的基座。

本文将从该论文出发，结合 Transformer 的具体代码实现，基于经典 NLP 任务——机器翻译，来深入讲解如何使用 Pytorch 搭建 Transformer 模型并解决机器翻译任务。

本文的参考论文为：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

本文参考的代码实现仓库包括：[NanoGPT](https://github.com/karpathy/nanoGPT)、[ChineseNMT](https://github.com/hemingkx/ChineseNMT)、[transformer-translator-pytorch](https://github.com/devjwsong/transformer-translator-pytorch)

## Transformer 模型架构

### Seq2Seq 模型

Seq2Seq，即序列到序列，是一种经典 NLP 任务。具体而言，是指模型输入的是一个自然语言序列 $input = (x_1, x_2, x_3...x_n)$，输出的是一个可能不等长的自然语言序列 $output = (y_1, y_2, y_3...y_m)$。事实上，Seq2Seq 是 NLP 最经典的任务，几乎所有的 NLP 任务都可以视为 Seq2Seq 任务。例如文本分类任务，可以视为输出长度为 1 的目标序列（如在上式中 $m$ = 1）；词性标注任务，可以视为输出与输入序列等长的目标序列（如在上式中 $m$ = $n$）。

机器翻译任务即是一个经典的 Seq2Seq 任务，例如，我们的输入可能是“今天天气真好”，输出是“Today is a good day.”。

Transformer 是一个经典的 Seq2Seq 模型，即模型的输入为文本序列，输出为另一个文本序列。Transformer 的整体模型结构如下图：

<div align=center><img src="./figures/transformer_architecture.png" alt="image-20230127193646262" style="zoom:50%;"/></div>

如图，​Transformer 整体由一个 Encoder，一个 Decoder 外加一个 Softmax 分类器与两层编码层构成。上图中左侧方框为 Encoder，右侧方框为 Decoder。

​作为一个 Seq2Seq 模型，在训练时，Transformer 的训练语料为若干个句对，具体子任务可以是机器翻译、阅读理解、机器对话等。此处我们以德语到英语的机器翻译任务为例。在训练时，句对会被划分为输入语料和输出语料，输入语料将从左侧通过编码层进入 Encoder，输出语料将从右侧通过编码层进入 Decoder。Encoder 的主要任务是对输入语料进行编码再输出给 Decoder，Decoder 再根据输出语料的历史信息与 Encoder 的输出进行计算，输出结果再经过一个线性层和 Softmax 分类器即可输出预测的结果概率，整体逻辑如下图：

<div align=center><img src="./figures/transformer_datalink.png" alt="image-20230127193646262" style="zoom:50%;"/></div>

接下来我们将从 Attention 机制出发，自底向上逐层解析 Transformer 的模型原理，并基于 Pytorch 框架进行模型实现，并最后搭建一个 Transformer 模型。

### Attention 机制

​Attention 机制是 Transformer 的核心之一，此处我们简要概述 attention 机制的思想和大致计算方法，如想要探究更多细节请大家具体查阅相关资料，例如：[Understanding Attention In Deep Learning (NLP)](https://towardsdatascience.com/attaining-attention-in-deep-learning-a712f93bdb1e)、[Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)等。在下文中，我们将从何为 Attention、self-attention 和 Multi-Head Attention 三个方面逐步介绍 Transformer 中使用的 Attention 机制，并手动实现 Transformer 中 Multi-Head Attention 层。

#### 何为 Attention

​Attention 机制最先源于计算机视觉领域，其核心思想为当我们关注一张图片，我们往往无需看清楚全部内容而仅将注意力集中在重点部分即可。而在自然语言处理领域，我们往往也可以通过将重点注意力集中在一个或几个 token，从而取得更高效高质的计算效果。

Attention 机制有三个核心变量：**Query**（查询值）、**Key**（键值）和 **Value**（真值）。我们可以通过一个案例来理解每一个变量所代表的含义。例如，当我们有一篇新闻报道，我们想要找到这个报道的时间，那么，我们的 Query 可以是类似于“时间”、“日期”一类的向量（为了便于理解，此处使用文本来表示，但其实际是稠密的向量），Key 和 Value 会是整个文本，通过对 Query 和 Key 进行运算我们可以得到一个权重，这个权重其实反映了从 Query 出发，对文本每一个 token 应该分布的注意力相对大小。通过把权重和 Value 进行运算，得到的最后结果就是从 Query 出发计算整个文本注意力得到的结果。

​具体而言，Attention 机制的特点是通过计算 **Query** 与**Key**的相关性为真值加权求和，从而拟合序列中每个词同其他词的相关关系。
其大致计算过程为：

<div align=center><img src="figures/transformer_attention.png" alt="image-20230129185638102" style="zoom:50%;"/></div>

1. 通过输入与参数矩阵，得到查询值$q$，键值$k$，真值$v$。可以理解为，$q$ 是计算注意力的另一个句子（或词组），$v$ 为待计算句子，$k$ 为待计算句子中每个词（即 $v$ 的每个词）的对应键。其中，$v$ 与 $k$ 的矩阵维度是相同的，$q$的矩阵维度则可以不同，只需要满足 $q$ 能和$k^T$满足矩阵相乘的条件即可。
2. 对 $q$ 的每个元素 $q_i$ ,对 $q_i$ 与 $k$ 做点积并进行 softmax，得到一组向量，该向量揭示了 $q_i$ 对整个句子每一个位置的注意力大小。
3. 以上一步输出向量作为权重，对 $v$ 进行加权求和，将 $q$ 的所有元素得到的加权求和结果拼接得到最后输出。

​其中，q，k，v 分别是由输入与三个参数矩阵做积得到的：

<div align=center><img src="figures/transformer_attention_compute.png" alt="image-20230129185638102" style="zoom:50%;"/></div>

​在实际训练过程中，为提高并行计算速度，直接使用 $q$、$k$、$v$ 拼接而成的矩阵进行一次计算即可。

​具体到 Transformer 模型中，计算公式如下：
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
​其中，$d_k$ 为键向量$k$的维度，除以根号$d_k$主要作用是在训练过程中获得一个稳定的梯度。计算示例如下图：

<div align=center><img src="figures/transformer_attention_compute_2.png" alt="image-20230129185638102" style="zoom:50%;"/></div>

​	Attention 的基本计算过程代码实现如下：

```python
'''注意力计算函数'''
from torch.nn import functional as F

def attention(q, k, v):
    # 此处我们假设 q、k、v 维度都为 (B, T, n_embed)，分别为 batch_size、序列长度、隐藏层维度
    # 计算 QK^T / sqrt(d_k)，维度为 (B, T, n_embed) x (B, n_embed, T) -> (B, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # 计算 softmax，维度为 (B, T, T)
    att = F.softmax(att, dim=-1)
    # V * Score，维度为(B, T, T) x (B, T, n_embed) -> (B, T, n_embed)
    y = att @ v 
    return y
```

#### Mask

由于​Transformer 是一个自回归模型，类似于语言模型，其将利用历史信息依序对输出进行预测。例如，如果语料的句对为：
    
    input： \<BOS> 我爱你 \<EOS>；
    output： \<BOS> I like you \<EOS>。

则 Encoder 获取的输入将会是句 input 整体，并输出句 input 的编码信息，但 Decoder 的输入并不一开始就是句 output 整体，而是先输入起始符\<BOS>，Decoder 根据\<BOS> 与 Encoder 的输出预测 I，再输入\<BOS> I，Decoder 根据输入和 Encoder 的输出预测 like。因此，自回归模型需要对输入进行 mask（遮蔽），以保证模型不会使用未来信息预测当下。

​关于自回归模型与自编码模型的细节，感兴趣的读者可以下来查阅更多资料，在此提供部分链接供读者参考。博客：[自回归语言模型 VS 自编码语言模型](https://zhuanlan.zhihu.com/p/163455527)、[预训练语言模型整理](https://www.cnblogs.com/sandwichnlp/p/11947627.html#预训练任务简介)；论文：[基于语言模型的预训练技术研究综述](http://jcip.cipsc.org.cn/CN/abstract/abstract3187.shtml)等。

因此，我们后续会分别对 Encoder 的 Attention 和 Decoder 的 Attention 进行不同的计算，具体而言，对 Decoder 的 Mask Attention，我们会传入一个 Mask 矩阵，这个 Mask 矩阵一般为和输入同等长度的下三角矩阵，例如，对于上文的 output 句，生成的掩码效果可能是：

    <BOS>
    <BOS> I
    <BOS> I like
    <BOS> I like you
    <BoS> I like you </EOS>

即当输入维度为 （B, T, n_embed），我们的 Mask 矩阵维度一般为 (1, T, T)（通过广播实现同一个 batch 中不同样本的计算），通过下三角掩码来遮蔽历史信息。在 Encoder 的 Attention 中，我们不会生成掩码，因此 Attention 是全部可见的；在 Decoder 的 Attention 中，我们会生成如上所述的掩码，从而保证每一个 token 只能使用自己之前 token 的注意力信息。

在具体实现中，我们通过以下代码生成 Mask 矩阵：

```python
# 此处使用 register_buffer 注册一个 bias 属性
# bias 是一个上三角矩阵，维度为 1 x block_size x block_size，block_size 为序列最大长度
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, config.block_size, config.block_size))
```

#### Self-Attention

​从上对 Attention 机制原理的叙述中我们可以发现，Attention 机制的本质是对两段序列的元素依次进行相似度计算，寻找出一个序列的每个元素对另一个序列的每个元素的相关度，然后基于相关度进行加权，即分配注意力。而这两段序列即是我们计算过程中 $Q$、$K$、$V$ 的来源。

​在经典的 Attention 机制中，$Q$ 往往来自于一个序列，$K$ 与 $V$ 来自于另一个序列，都通过参数矩阵计算得到，从而可以拟合这两个序列之间的关系。例如在 Transformer 的 Decoder 结构中，$Q$ 来自于 Encoder 的输出，$K$ 与 $V$ 来自于 Decoder 的输入，从而拟合了编码信息与历史信息之间的关系，便于综合这两种信息实现未来的预测。

​但在 Transformer 的 Encoder 结构中，使用的是 Attention 机制的变种 —— self-attention （自注意力）机制。所谓自注意力，即是计算本身序列中每个元素都其他元素的注意力分布，即在计算过程中，$Q$、$K$、$V$ 都由同一个输入通过不同的参数矩阵计算得到。在 Encoder 中，$Q$、$K$、$V$ 分别是输入对参数矩阵 $W_q$、$W_k$、$W_v$ 做积得到，从而拟合输入语句中每一个 token 对其他所有 token 的关系。

​例如，通过 Encoder 中的 self-attention 层，可以拟合下面语句中 it 对其他 token 的注意力分布如图：

<div align=center><img src="figures/transformer_selfattention.jpg" alt="image-20230129190819407" style="zoom:50%;"/></div>

​在代码中的实现，self-attention 机制其实是通过给 $Q$、$K$、$V$ 的输入传入同一个参数实现的：

```python
# attn 为一个注意力计算层
self.attn(x, x, x)
```

​上述代码是 Encoder 层的部分实现，self.attn 即是注意力层，传入的三个参数都是 $x$，分别是 $Q$、$K$、$V$ 的计算输入，从而 $Q$、$K$、$$ 均来源于同一个输入，则实现了自注意力的拟合。

#### Multi-Head Attention

​Attention 机制可以实现并行化与长期依赖关系拟合，但一次注意力计算只能拟合一种相关关系，单一的 Attention 机制很难全面拟合语句序列里的相关关系。因此 Transformer 使用了 Multi-Head attention 机制，即同时对一个语料进行多次注意力计算，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息。

<div align=center><img src="figures/transformer_Multi-Head attention.png" alt="image-20230129190819407" style="zoom:50%;"/></div>

在原论文中，作者也通过实验证实，多头注意力计算中，每个不同的注意力头能够拟合语句中的不同信息，如下图：

<div align=center><img src="figures\transformer_Multi-Head visual.jpg" alt="image-20230207203454545" style="zoom:50%;" />	</div>

​上层与下层分别是两个注意力头对同一段语句序列进行自注意力计算的结果，可以看到，对于不同的注意力头，能够拟合不同层次的相关信息。通过多个注意力头同时计算，能够更全面地拟合语句关系。

​Multi-Head attention 的整体计算流程如下：

<div align=center><img src="figures/transformer_Multi-Head attention_compute.png" alt="image-20230129190819407" style="zoom:50%;"/></div>

所谓的多头注意力机制其实就是将原始的输入序列进行多组的自注意力处理；然后再将每一组得到的自注意力结果拼接起来，再通过一个线性层进行处理，得到最终的输出。我们用公式可以表示为：

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ...,
\mathrm{head_h})W^O    \\
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

其最直观的代码实现并不复杂，即 n 个头就有 n 组3个参数矩阵，每一组进行同样的注意力计算，但由于是不同的参数矩阵从而通过反向传播实现了不同的注意力结果，然后将 n 个结果拼接起来输出即可。

但上述实现复杂度较高，我们可以通过矩阵运算巧妙地实现并行的多头计算，其核心逻辑在于使用三个组合矩阵来代替了n个参数矩阵的组合。此处我们手动实现一个 MultiHeadAttention 层（注意，我们加入了 Mask 矩阵和 Dropout 操作）：

```python
import torch.nn as nn
import torch

'''多头注意力计算模块'''
class MultiHeadAttention(nn.Module):

    def __init__(self, config, is_causal=False):
        # 构造函数
        # config: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert config.n_embd % config.n_head == 0
        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x n_embd
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.c_attns = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd, bias=config.bias) for _ in range(3)])
        # 输出的线性层，维度为 n_embd x n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        # 头数
        self.n_head = config.n_head
        # 隐藏层维度
        self.n_embd = config.n_embd
        # Dropout 概率
        self.dropout = config.dropout
        # 是否是解码器的 Casual LM
        self.is_causal = is_causal
        # 判断是否使用 Flash Attention，Pytorch 2.0 支持，即判断 torch.nn.functional.scaled_dot_product_attention 是否存在
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # 如果不使用 Flash Attention，打印一个警告
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 如果自己实现 MHSA，需要一个 causal mask，确保 attention 只能作用在输入序列的左边
            # 此处使用 register_buffer 注册一个 bias 属性
            # bias 是一个上三角矩阵，维度为 1 x 1 x block_size x block_size，block_size 为序列最大长度
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, query, key, value):
        # 输入为 query、key、value，维度为 (B, T, n_embed)
        # print("query",query.size())
        B, T, C = query.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # 计算 Q、K、V，输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, n_embed) -> (B, T, n_embed)
        q, k, v  = [self.c_attns[i](x) for i, x in zip(range(3), (query, key, value))]
        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, C // n_head)，然后交换维度，变成 (B, n_head, T, C // n_head)，因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 注意力计算 
        if self.flash:
            # 直接使用 Flash Attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal)
        else:
            # 手动实现注意力计算
            # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # 如果是解码器的 Casual LM，需要 mask 掉右上角的元素
            if self.is_causal:
                # 这里截取到序列长度，因为有些序列可能比 block_size 短
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # 计算 softmax，维度为 (B, nh, T, T)
            att = F.softmax(att, dim=-1)
            # Attention Dropout
            att = self.attn_dropout(att)
            # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v 
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, C // n_head)，再拼接成 (B, T, n_head * C // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 经过输出层计算，维度为 (B, T, C)，再经过线性层残差连接
        y = self.resid_dropout(self.c_proj(y))
        return y
```
