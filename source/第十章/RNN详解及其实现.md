# 文章结构
提及 RNN，绝大部分人都知道他是一个用于序列任务的神经网络，会提及他保存了时序信息，但是，为什么需要考虑时序的信息？为什么说 RNN 保存了时序的信息？RNN又存在哪些问题？ 本篇内容将按照以下顺序逐步带你摸清 RNN 的细节之处，并使用 torch 来完成一个自己的文本分类模型。

 1. 为什么需要 RNN？
 2. RNN 理解及其简单实现。
 3. 用 RNN 完成文本分类任务。
 4. RNN 存在的问题。

# 为什么需要 RNN？
在现实生活的世界中，有很多的内容是有着前后关系的，比如你所阅读的这段文字，他并不是毫无理由的随机组合，而是我构思之后按顺序写下的一个个文字。除了文字之外，例如人的发音、物品的价格的曲线、温度变化等等，都是有着前后顺序存在的。

很明显，当知道了前面的信息，就可以对后面的信息进行合理的预测。比如，前十天温度都只有20度，明天的温度无论如何不可能零下；这个商品一年来价格都在30左右浮动，明天我去买他的时候，准备40就足够了；老师很好的表扬了你，紧跟着说了一个但是，你就知道他的内容要开始转折了。这就是隐藏在日常生活中的序列信息，因为已经知道了前面发生的内容，所以才可以推理后面的内容。

那么，可以用传统的多层感知机来处理序列问题吗？按照基本的多层感知机模型方案来实现，应该是这样的：将序列输入固定成一个 $d$ 维向量，就可以送入多层感知机进行学习，形如公式：
$$
 \mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_{h})
$$
 公式中， $\phi$ 表示激活函数，$\mathbf{X} \in \mathbb{R}^{n \times d}$ 表示一组小批量样本，其中 $n$ 是样本大小， $d$ 表示输入的特征维度。：$\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$表示模型权重参数，$d \in \mathbb{R}^{1 \times h}$表示模型偏置。最后可以得到隐藏层输入：$\mathbf{H} \in \mathbb{R}^{n \times h}$，其中 $h$ 表示隐藏层大小。

紧接着，模型可以使用下面的公式进行计算，得到输出：
$$
\mathbf{O} = \mathbf{H}\mathbf{W}_{hq} + \mathbf{b}_q
$$
其中，$\mathbf{O} \in \mathbb{R}^{n \times q}$ 为模型输出变量，$q$ 表示输出层向量，由于本次的任务是一个文本分类任务，那这里 $q$ 就表示文本类别，可以使用 $\mathbf{Softmax(O)}$ 来进行概率预测。

但是，上面的流程有一个很明显的前置条件：**固定成 $d$ 维向量**，也就是说，传统的多层感知机，是不能支持向量长度进行变化的。**但是**，在序列任务中，序列长短很明显是并不相同的，不仅需要用一天的数据预测明天的结果，也可能需要拿一年的数据预测明天的结果。在这样的情况下，如果还想要使用传统的多层感知机，就会面临着一个巨大的问题：如何将一天的内容与一年的内容变化成相同的 $d$ 维向量？

除此之外，序列信息可能还有另外一个情况：某些信息可能出现在序列的不同位置。虽然信息出现在不同的位置，但是他可能表达出了相同的含义。

举例来说：当我们和老师谈话时，，如果他表扬了我们半小时，然后说："但是..."，我们往往是不担心的，因为他可能只是为了指出一些小问题。如果他刚刚表扬了一句话，紧接着就说“但是”，那我们就必须做好面对半小时的狂风暴雨。还有另外一种可能，老师可能连续批评你很久，然后使用“但是”转折，你就会在这时候如释重负，因为你知道这场谈话就快要结束了。
这就是我们根据前文(表扬的内容和时间)，在老师说出"但是"的时候，所作出的判断。

上面提到的两个问题，使用多层感知机本身似乎难以解决，但是所幸，RNN 从一个更常规的思路出发来解决这个问题：**记住之前看到的内容，并结合当前看到的内容，来预测之后可能的内容。**

# RNN 理解及其简单实现
根据开篇的内容，相信你已经可以简单的理解为什么传统的多层感知机无法很好的解决序列信息，接下来我们开始理解，RNN 如何记忆之前的内容的。

在这里，我先放出 RNN 的公式，请将其与多层感知机公式进行对比：
$$
\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).
$$
可以看到，与上一个公式相比，这里最明显的一点是多了一个 $\mathbf{H}_{t-1} \mathbf{W}_{hh}$ ，从公式上似乎很好理解，$\mathbf{H}_{t-1}$ 表示着前一时刻的隐藏状态，表示的是**之前看到的内容**，然后加上当前时刻的输入 $\mathbf{X}_t$，就可以输出当前时刻的隐藏结果 $\mathbf{H}_{t}$。在得到隐藏结果后，它就可以被用于下一步的计算。


当然，这个迭代过程也可能随时终止，如果将得到的隐藏结果用于输出，便可以直接得到输出结果，公式表达为：
$$
\mathbf{O} = \phi(\mathbf{H}_{t} \mathbf{W}_{hq}  + \mathbf{b}_q).
$$

可以看到，公式四与公式二极其相似，仅有隐藏状态 $\mathbf{H}$ 略有不同。

此时，根据以上公式及其理解，已经可以构建一个简单的 RNN 模型了：
```python
class RNNDemo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNNClassify, self).__init__()

        self.hidden_size = hidden_size
        # 计算隐藏状态 H
        # 因为要用以下一次计算，所以输出维度应该是 hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 输出结果 O
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input, hidden):
        # 将 X 和 Ht-1 合并
        combined = torch.cat((input, hidden), 1)
        # 计算 Ht
        hidden = self.i2h(combined)
        # 计算当前情况下的输出
        output = self.i2o(combined)
        # 分类任务使用 softmax 进行概率预测
        output = self.softmax(output)
        # 返回预测结果 和 当前的隐藏状态
        return output, hidden
    def initHidden(self):
    	# 避免随机生成的 H0 干扰后续结果
        return torch.zeros(1, self.hidden_size)
```
辅助代码理解：根据公式三可知，$\mathbf{W}_{xh}$ 和 $\mathbf{W}_{hh}$ 在输入阶段时两者互不影响，所以在 `self.i2h = nn.Linear(input_size + hidden_size, hidden_size)` 中对输入的维度进行扩容，前 `input_size` 与公式 $\mathbf{W}_{xh}$ 对应，而后面的 `hidden_size` 则是和 $\mathbf{W}_{hh}$ 对应。

阅读代码之后，请根据代码和公式，来回忆第一部分提出的两个问题，通过回答这两个问题，就可以进一步的分析 RNN。

第一个问题是对于不同的序列长度，如何进行处理其向量表示：

从公式中可以看到，RNN 并不要求不同的序列表示成相同的维度，而是要求序列中的每一个值，表示成为相同的维度，这样，我们可以将在 $t$ 时刻输入的值视为的 $\mathbf{X}_t$，并且结合之前时刻输入并计算得来的隐藏状态 $\mathbf{H}_{t-1}$，得到当前时刻的结果，这样无论序列实际长度如何，我们随时可以在想要中断的时候将隐藏状态转变成输出的结果，甚至我们可以在输入的同时，得到输出的结果。
【图片，待补充】

第二个问题，某些信息可能出现在序列的不同位置，但是其表达的含义是相同的：

对于这个问题，单独查看公式与代码可能不太好理解，但是可以从卷积神经网络中得到一定的灵感。

卷积神经网络具有平移等变性，也就是说输入的 $\mathbf{X}$ 不会因为位置的变化而导致输出的不同，这得益于卷积核使用了参数共享，无论图片哪个位置进行输入，只要卷积核的参数不变，输入值就不变，其结果就不会发生变化。

扭回头来看 RNN 中，其 $\mathbf{X}$  与 $\mathbf{H}$ 所使用的权重矩阵一直是一个，也就是说 $\mathbf{W}_{xh}$ 和 $\mathbf{W}_{hh}$ 是参数共享的，那么无论从序列的哪个位置进行输入，只要输入内容完全一样，其输出结果也就是完全一样的的。

在理解了 RNN 来龙去脉之后，接下来开始从 RNN 的在实际文本分类中进行更深入的分析。(注：该样例源自 [Torch 官方教程](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html))。

# RNN 完成文本分类任务
完成一个基本的算法任务，有以下流程：数据分析、数据转换、构建模型、定义训练函数、执行训练、保存模型、评估模型。

这里摘取官方教程中部分关键代码进行讲解，可以直接[点击这里](https://pytorch.org/tutorials/_downloads/13b143c2380f4768d9432d808ad50799/char_rnn_classification_tutorial.ipynb)直接下载官方 notebook进行训练。训练所用数据位于[这里](https://download.pytorch.org/tutorial/data.zip)。

在 notebook 第一个可执行 cell 中，首先定义了可用字符 `all_letters` 和 可用字符数量 `n_letters` 。同时，将下载的数据转为 ASCII 编码，使用  Dict l类型进行存储。其保存格式为：``{language: [names ...]}``。

在第三个 cell 中，定义了三个方法，主要目的是将由 $n$ 个字符组成的句子变成一个 $n \times d$ 的向量，其中 $d$ 表示字符特征，在这里使用 One-Hot 编码。由于 One-Hot 编码形为 $1 \times n\_letters$，则最终形状为 $n \times 1 \times n\_letters$。

第四个 Cell 中，定义了基本的 RNN 模型的代码与上方代码一致，并设置隐藏层大小为128。接下来的第五个和第六个可执行 Cell中，对于 RNN 进行简单的测试。

在这里，简单的讲解第六个 Cell，第六个 Cell 代码如下：
```python
# 将 Albert 转为 6 * 1 * n_letters 的 Tensor
input = lineToTensor('Albert')
# 设置 h0 全零的原因在上面提到过
hidden = torch.zeros(1, n_hidden)
# 获取 output 和 h1
output, next_hidden = rnn(input[0], hidden)
print(output)
```
第三行中 ` rnn(input[0], hidden)`，输入了首字母 也就是 'A' 的 One-Hot 编码，输出 `output` 是下一个字符可能是什么的概率，而 `next-hidden` 则是 用于搭配 `input[1]` 进行下一步输入训练的模型。

跳过七八九三个 Cell 后，我们再对 train 所在的 Cell 进行分析，下面是相关代码:
```python
# 设置学习率
learning_rate = 0.005 
# 输入参数中， categor_tensor 表示类别，用以计算 loss
# line_tensor 是由一句话所转变的 tensor, shape: n * 1 * n_letter
def train(category_tensor, line_tensor):
    # 设置 H0
    hidden = rnn.initHidden()
    # 梯度清零
    rnn.zero_grad()
    # 将字符挨个输入
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 计算损失
    loss = criterion(output, category_tensor)
    # 梯度回传
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    # 这里其实是一个手动的优化器 optimizer
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()
```
结合注释，进一步观察代码，可以看到，对于一个变长序列，在输入最后一个字符之前，都使用 `hidden` 作为输出用于下一步的计算，也就是将历史信息带入下一轮训练中去，而在最后一个字符输入结束后，使用 `outpt` 作为输出，进行文本分类的预测。

在 `train` 中的代码进行了对一句话进行了单独的训练，而实际过程中，我们要对多个句子进行训练，在示例代码中，采用随机采样法，从全部数据中随机提取一句话进行训练，并得到最终结果：
```python
import time
import math
# 迭代次数
n_iters = 100000
# 输出频率
print_every = 5000
# loss计算频率
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
# 开始时间
start = time.time()

for iter in range(1, n_iters + 1):
    # 使用随机采样提取一句话及其标签
    category, line, category_tensor, line_tensor = randomTrainingExample()
    # 训练
    output, loss = train(category_tensor, line_tensor)
    # 计算loss
    current_loss += loss

    # Print iter number, loss, name and guess
    # 输出阶段结果
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    # 保存阶段性 loss
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
```
到这里，RNN 训练的代码可以说讲解结束了，保存以及推理在理解了训练的过程上并不算难事，所以不再进行讲解。

接下来再根据文本分类任务对 RNN 进行一次分析。可以看到，在本次任务中，一个单独的词作为一个序列，每个词的长短不一并不会影响 RNN 的训练过程，而序列中的值，则是字符，每个字符都构成了相同的向量： $1 \times d$ ，这使得训练的过程也比较的统一。

再简单的举一反三，可以结合之前所学的 word2vec、Glovec 等模型将词语转为向量，将一句话转为一个序列，每个词转为序列中的一个值，这样的话，就可以对一句话进行文本分类了。

# RNN 存在的问题

前面讲解了 RNN 是如何解决简单神经网络无法处理序列问题的，但是 RNN 是否就完美无缺？能应用于全部的序列任务了呢？答案当然是否定的。

这是由于 RNN 存在一个巨大的缺陷：梯度爆炸与梯度消失。

重新审查代码与公式，可以很轻松的发现，在序列达到末尾时，我们才需要计算损失与进行梯度回传，此时将 $\mathbf{H}_t$ 展开，其内部存在 $\mathbf{W}_{hh} \times \mathbf{H}_{t-1}$。而将 $\mathbf{H}_{t-1}$ 展开，也存在一个 $\mathbf{W}_{hh}$，那么很明显 如果 $\mathbf{W}_{hh}$ 大于 1，在经过 $t$ 次连乘之后会产生梯度爆炸，如果  $\mathbf{W}_{hh}$ 小于 1，在经过 $t$ 次连乘之后又会产生梯度消失。同理，在 $\mathbf{W}_{xh}$ 上，也存在这样的依赖关系，也会导致梯度爆炸或者消失。


为了解决这个问题，我们可以考虑使用 梯度剪裁的方式保证梯度不会爆炸，当然，也可以查看下一篇，针对 RNN 进行优化的 LSTM、GRU 等算法











 
 

