# GAT解读

解读文章：@article{velivckovic2017graph,
  title={Graph attention networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Lio, Pietro and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1710.10903},
  year={2017}
}
本文作者：卢文涛@luspyder

## 前言

图神经网络（graph neural network，GNN）是以图数据为载体的一类神经网络研究方向。与研究序列化数据为主的自然语言处理、网格化数据为主的计算机视觉不同，图神经的网络的图数据是在非欧空间上的数据，拥有更大的可变性，也因此具有更强的代表性：化学分子、流体力学中的粒子等实际存在的对象和社交网络、城市交通节点等抽象的对象都可以用图数据代表，在这些数据上使用深度学习模型进行分类、回归等工作，对实际生活生产有极大的潜在价值。

前面的章节已经详细介绍了transformer中的注意力机制，同样的，一直有工作在研究注意力机制与图神经网络这两个都具有强大表达能力和理解能力的方法的结合。图注意力网络（GRAPH ATTENTION NETWORKS）就是早期奠基工作之一，它也是从空域角度出发研究图神经网络的起步学习内容之一。可以说，对于潜心研究GNN的同学，直接用GAT项目作为自己深度学习的“hellow world”项目比用手写数字识别更有价值。

本文简略介绍了GAT模型的核心算法逻辑，以GAT模型的基本代码为主体，详细介绍了GAT的核心公式是如何实现的，并在cora数据集上进行了实验，此后还提供了改进了注意力机制中的权重的GAT模型的实现代码，一并做了注释讲解。

## 模型介绍

GAT与新时代图神经网络开山之作GCN一样，是一个很简单的模型，因此论文中没有复杂的模型架构图，大家都在数据结构中了解过图这种数据类型，因此只需要简单的描述就可以给读者一个感性的理解：“**根据注意力机制，有选择的将目标结点的邻居节点所蕴含的信息传递给目标节点，由此更新目标节点自身的信息。**”

感性的理解了GAT模型之后，GAT的数学表达式也变得很直观：

GAT中邻居节点对中心节点的贡献度是learnable的，利用注意力机制学习两个节点之间的相对权重：

$$
h_v^{(k)}=\sigma(\sum_{u\in N(v))\cup v}\alpha_{uv}^{(k)}W^{(k)}h_u^{(k-1)})
$$

权重$\alpha_{(uv)}^{(k)}$衡量节点与邻居节点之间的相关性：

$$
\alpha_{uv}^{(k)}=softmax(g(a^T[W^{(k)}h_v^{(k-1)}||W^{(k)}h_u^{(k-1)}))
$$

其中 g(⋅) 是LeakyReLU激活函数，a 是可学习参数的向量。GAT进一步执行多头注意力以增加模型的表达能力。门控注意力网络 ( **GAAN** )引入了一种自注意力机制，该机制为每个注意力头计算额外的注意力分数。除了在空间上应用图注意力，GeniePath还提出了一种类似LSTM的门控机制来控制跨图卷积层的信息流。还有其他可能感兴趣的图注意模型。但是，它们不属于卷积图神经网络的框架。

其中 g(⋅) 是LeakyReLU 激活函数，$a$是可学习参数的向量。

## 代码实现

### 固定注意力权重的GAT

#### build GAT model

```python
import torch
from torch import nn

from labml_helpers.module import Module


class GraphAttentionLayer(Module):
    """定义一个GAT层
    """
    def __init__(self, in_features: int, 
                 out_features: int,
                 n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2):#输入特征维度、输出特征维度、注意力头数、每个头的注意力是拼接还是均值合并（好像均值合并的多？）、dropout概率、leaky relu的某个参数
   
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

   
        if is_concat:#计算每个注意力头输出的维度
            assert out_features % n_heads == 0#不符合直接报错
  
            self.n_hidden = out_features // n_heads#如果是把每个注意力头的拼接起来作为最终输出，那么输出维数均分给各个注意力头
        else:

            self.n_hidden = out_features#均值求和，每个注意力头的输出维数一致

   
        self.linear = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)#转换前的线性层，把输入的维数变为够给每个注意力头所需的唯一神经元数目（每个注意力头的每个输入都是唯一的）
   
        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)#计算注意力分数，因为一个Q一个K，所以输入维数是n_hidden*2

        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)#激活函数

        self.softmax = nn.Softmax(dim=1)#计算注意力结果

        self.dropout = nn.Dropout(dropout)#来点dropout

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):#h是输入节点的embedding的维度：（n_nodes,in_features）
        #后者是邻接矩阵，定义中的shape是(n_nodes,n_nodes,n_heads)，对每个注意力头是一样的，所以实际上是(n_nodes,n_nodes,1)

        n_nodes = h.shape[0]#节点数目
        g = self.linear(h).view(n_nodes, self.n_heads, self.n_hidden)#初始transformation，对每个节点做线性变换并将其独立分给各个注意力头
        #view和reshape是一样的 https://blog.csdn.net/qq_26400705/article/details/109816853
        #g_i^k=W^k*h_i
        #接下来计算注意力得分
        #e_{ij}=LeakyReLU(a^T(g_i,g_j))
        #a是注意力机制，在GAT中是将两个节点的embbeding线性变化之后(g_i)串联起来再套一个线性变换层、一个relu完事（好简单粗暴啊……）
        g_repeat = g.repeat(n_nodes, 1, 1)#计算所有节点对之间的(g_i,g_j)，此处g_repeat得到n_nodes次循环的[g_1,……g_N]一维序列
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)#此处g_repeat_interleave得到的是[g_1,…,g_1,g_2,…,g_2,g_3…]，每个元素循环n_nodes次
        #again,n_nodes是注意力头数

        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)#把上面两个cat一下，就得到{g_1||g_1,g_1||g_2,…,g_1||g_n,g_2||g_1,g_2||g_2,…,g_2||g_n,…}了
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.n_hidden)#把数据格式改一下，让g_concat[i,j]=g_i||g_j

        e = self.activation(self.attn(g_concat))#计算LeakyReLU(a^T(g_i,g_j))，得到的e的shape是[n_nodes,n_nodes,n_heades,1]
        e = e.squeeze(-1)#移除数组中维度为1的维度

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads#检验邻接矩阵的shape为[n_nodes, n_nodes, n_heads]或[n_nodes, n_nodes, 1]
        e = e.masked_fill(adj_mat == 0, float('-inf'))#将不连通节点对的e_ij设置为负无穷

        a = self.softmax(e)#正则化注意力分数
        #a_ij=softmax(e_ij)=\frac{exp(e_ij)}{\sum_{k\in N_i}exp(e_ik)}

        a = self.dropout(a)#dropout一下，别过拟合了

        attn_res = torch.einsum('ijh,jhf->ihf', a, g)#计算每个注意力头的最终输出
        #矩阵乘积求和，后面字符串参数指定的是求和的shape https://zhuanlan.zhihu.com/p/434232512
        #h_i^`k=\sum_{j\in N_i}a_{ij}^k g_j^k
        #原文在这里有激活函数的，但是代码没把它写在GAT layer里面，写在GAT model里面

        if self.is_concat:#按照参数串联或者均值求和各个注意力头的输出
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)
```

#### run GAT on cora dataset

```python
from typing import Dict

import numpy as np
import torch
from torch import nn

from labml import lab, monit, tracker, experiment
from labml.configs import BaseConfigs, option, calculate
from labml.utils import download
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_nn.graphs.gat import GraphAttentionLayer
from labml_nn.optimizers.configs import OptimizerConfigs


class CoraDataset:
    """Cora是个引文数据集，包含7类节点（文章），引用作为边
    """


    labels: torch.Tensor#每个节点的标签
    classes: Dict[str, int]#类名和整数索引
    features: torch.Tensor#每个节点的向量特征
    adj_mat: torch.Tensor#邻接矩阵(true if connect)

    @staticmethod
    def _download():#下载数据集
        if not (lab.get_data_path() / 'cora').exists():
            download.download_file('https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz',
                                   lab.get_data_path() / 'cora.tgz')
            download.extract_tar(lab.get_data_path() / 'cora.tgz', lab.get_data_path())

    def __init__(self, include_edges: bool = True):#加载数据集
        self.include_edges = include_edges

        self._download()#下载数据集

        with monit.section('Read content file'):#读取文章id、向量特征和标签
            content = np.genfromtxt(str(lab.get_data_path() / 'cora/cora.content'), dtype=np.dtype(str))
        with monit.section('Read citations file'):#加载引用数据，是一个int对组成的list
            citations = np.genfromtxt(str(lab.get_data_path() / 'cora/cora.cites'), dtype=np.int32)

        features = torch.tensor(np.array(content[:, 1:-1], dtype=np.float32))#读取向量特征
        self.features = features / features.sum(dim=1, keepdim=True)#正则化向量特征

        self.classes = {s: i for i, s in enumerate(set(content[:, -1]))}#读取每个类名并指定一个单独的int标签
        self.labels = torch.tensor([self.classes[i] for i in content[:, -1]], dtype=torch.long)#读取每个类的int标签

        paper_ids = np.array(content[:, 0], dtype=np.int32)#读取文章的id
        ids_to_idx = {id_: i for i, id_ in enumerate(paper_ids)}#将id与标签做映射

        self.adj_mat = torch.eye(len(self.labels), dtype=torch.bool)#生成对角线全1，其余全0的数组，初始化邻接矩阵，指定输出类型为bool的torch

        if self.include_edges:#将引用数据导入邻接矩阵
            for e in citations:
                e1, e2 = ids_to_idx[e[0]], ids_to_idx[e[1]]
                self.adj_mat[e1][e2] = True
                self.adj_mat[e2][e1] = True#无向图


class GAT(Module):#来两层GAT layer
  
    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float):
        super().__init__()

        self.layer1 = GraphAttentionLayer(in_features, n_hidden, n_heads, is_concat=True, dropout=dropout)#第一个GAT layer的各个注意力头串联起来作为输出
        self.activation = nn.ELU()#上个cell缺的激活函数放在这儿了
        self.output = GraphAttentionLayer(n_hidden, n_classes, 1, is_concat=False, dropout=dropout)#第二个GAT layer的各个注意力头均值求和作为输出
        self.dropout = nn.Dropout(dropout)#drop out一下

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):#x是[n_nodes,in_features]的向量特征
        x = self.dropout(x)#对输入做dropout
        x = self.layer1(x, adj_mat)#第一层GAT layer
        x = self.activation(x)#激活函数
        x = self.dropout(x)#再dropout一下
        return self.output(x, adj_mat)


def accuracy(output: torch.Tensor, labels: torch.Tensor):#计算准确率
    return output.argmax(dim=-1).eq(labels).sum().item() / len(labels)


class Configs(BaseConfigs):#训练配置
    model: GAT
    training_samples: int = 500#训练节点
    in_features: int#输入节点的特征数
    n_hidden: int = 64#第一层GAT的特征
    n_heads: int = 8#注意力头数
    n_classes: int#类数
    dropout: float = 0.6#dropout概率？？怎么这么高
    include_edges: bool = True
    dataset: CoraDataset
    epochs: int = 1_000
    loss_func = nn.CrossEntropyLoss()#交叉熵作为损失函数
    device: torch.device = DeviceConfigs()#指定训练设备
    optimizer: torch.optim.Adam#遇优化器不决选adam

    def run(self):#训练循环
        #对于较小的数据集，每个batch就是整个数据
        #如果需要采样训练，那么需要把经过被采样点的边全部加上
        features = self.dataset.features.to(self.device)#把向量特征扔显存
        labels = self.dataset.labels.to(self.device)#把标签扔显存
        edges_adj = self.dataset.adj_mat.to(self.device)#把邻接矩阵扔显存
        edges_adj = edges_adj.unsqueeze(-1)#给邻接矩阵加上一个第三维度

        idx_rand = torch.randperm(len(labels))#随机初始化index
        idx_train = idx_rand[:self.training_samples]#训练用节点
        idx_valid = idx_rand[self.training_samples:]#验证节点

        for epoch in monit.loop(self.epochs):#开训！
            self.model.train()#设置模型为训练模式
            self.optimizer.zero_grad()#梯度置为=0
            output = self.model(features, edges_adj)#正向传播
            loss = self.loss_func(output[idx_train], labels[idx_train])#计算训练集的损失
            loss.backward()#反向传播
            self.optimizer.step()#优化
            tracker.add('loss.train', loss)#记录loss
            tracker.add('accuracy.train', accuracy(output[idx_train], labels[idx_train]))#记录准确率

            self.model.eval()#把模型设置为验证模式，用于验证集

            with torch.no_grad():#验证集不用算梯度
                output = self.model(features, edges_adj)#正向计算输出again
                loss = self.loss_func(output[idx_valid], labels[idx_valid])#计算验证集的损失
                tracker.add('loss.valid', loss)#记录loss
                tracker.add('accuracy.valid', accuracy(output[idx_valid], labels[idx_valid]))#记录准确率

            tracker.save()


@option(Configs.dataset)#创建cora数据集
def cora_dataset(c: Configs):
    return CoraDataset(c.include_edges)


calculate(Configs.n_classes, lambda c: len(c.dataset.classes))#计算类数
calculate(Configs.in_features, lambda c: c.dataset.features.shape[1])#输入特征的数量


@option(Configs.model)#创建GAT模型
def gat_model(c: Configs):
    return GAT(c.in_features, c.n_hidden, c.n_classes, c.n_heads, c.dropout).to(c.device)


@option(Configs.optimizer)#创建参数优化器
def _optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    return opt_conf


def main():
    conf = Configs()#配置
    experiment.create(name='gat')#创建实验
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 5e-3,
        'optimizer.weight_decay': 5e-4,
    })

    with experiment.start():
        conf.run()#run！


#
if __name__ == '__main__':
    main()
```

### 可学习注意力权重的GAT

#### dynamic attention GAt

在上面的注意力机制中，key与所有query的注意力权重是相等且固定的，不仅不符合直观构想，也不符合指定权重的初衷，因此在原始的GAT上加入两个可选参数：

$$
\begin{aligned}e_{ij}&=LeakyReLU(a^T[w\overrightarrow{h_i}||W\overrightarrow{h_j}])\\&=LeakyReLU(a_1^TW\overrightarrow{h_i}+a_2^TW\overrightarrow{h_j}])\end{aligned}\tag{1}
$$

实际上，对于每一个节点i，它对其它节点的注意力分数只取决于$a_2^TW\overrightarrow{h_j}$,因此动态注意力机制GAT可以共享参数为：

$$
\begin{aligned}e_{ij}&=a^TLeakyReLU(W[\overrightarrow{h_i}||\overrightarrow{h_j}])\\&=a^TLeakyReLU(W_l\overrightarrow{h_i}+W_r\overrightarrow{h_j})\end{aligned}\tag{2}
$$

```python
import torch
from torch import nn

from labml_helpers.module import Module


class GraphAttentionV2Layer(Module):
  
    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):
        #share_weights若设置为true，将把线性升维层的参数共享给边的出点和入点（考虑的是有向图）
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        if is_concat:#计算每个注意力头的输出维数
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads#如果是把注意力头的输出串联起来
        else:
            self.n_hidden = out_features#如果是把注意力头的输出均值求和

        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)#线形层把输入中的出节点的特征升维成所有注意力头所需要的维数
        if share_weights:#若共享出入节点的参数（如上面的公式2）
            self.linear_r = self.linear_l#共享参数
        else:#若独立参数（如上面的公式1）
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)#给入节点设置独立的线形层参数
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)#用线性层计算QK
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)#激活一下QK
        self.softmax = nn.Softmax(dim=1)#用softmax计算注意力得分
        self.dropout = nn.Dropout(dropout)#dropout一下

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        """h是输入节点的embedding的shape[n)nodes,in_features]
        adj_mat是邻接矩阵的shape[n_nodes, n_nodes, n_heads],实际上是[n_nodes, n_nodes, 1]
        邻接矩阵内的元素类型是bool
        """
  
        n_nodes = h.shape[0]#节点数目
        #把\overrightarrow{h_i}加上权重W_l，把\overrightarrow{h_j}加上权重W_r
        g_l = self.linear_l(h).view(n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(n_nodes, self.n_heads, self.n_hidden)
        #接下来计算注意力得分
        #$e_{ij}=a^TLeakyReLU([\overrightarrow{g_{li}}+\overrightarrow{g_{rj}}])

        g_l_repeat = g_l.repeat(n_nodes, 1, 1)#首先计算[g_li+g_rj],这里得到的是g_l1到g_ln循环n_nodes次的序列
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=0)#这里得到的是g_r1循环n_nodes次，再g_r2循环n_nodes次……
        g_sum = g_l_repeat + g_r_repeat_interleave#求和，得到[\overrightarrow{g_{li}}+\overrightarrow{g_{rj}}]
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.n_hidden)#reshape一下使得g_sum[i,j]=[\overrightarrow{g_{li}}+\overrightarrow{g_{rj}}]

        e = self.attn(self.activation(g_sum))#计算e_{ij}，e的shape是[n_nodes, n_nodes, n_heads, 1]
        e = e.squeeze(-1)#把最后面那个一维去掉

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads#检查邻接矩阵的格式
        e = e.masked_fill(adj_mat == 0, float('-inf'))

        a = self.softmax(e)#用softmax把e_ij变成a_ij

        a = self.dropout(a)

        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)#计算每个注意力头的输出:\sum_{j\in N_i}a_{ij}^k\overrightarrow{g_{rj.k}}

        if self.is_concat:#把注意力头的输出组合起来
            return attn_res.reshape(n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=1)
```

#### run dynamic GAT on cora model

```python
import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.graphs.gat.experiment import Configs as GATConfigs
from labml_nn.graphs.gatv2 import GraphAttentionV2Layer


class GATv2(Module):
    """
    两层GAT layer
    """

    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float,
                 share_weights: bool = True):
        """
        略
        """
        super().__init__()

        self.layer1 = GraphAttentionV2Layer(in_features, n_hidden, n_heads,
                                            is_concat=True, dropout=dropout, share_weights=share_weights)#第一层的注意力头输出串联
        self.activation = nn.ELU()#第一层GAT layer的激活函数
        #第二层GAT layer叫这个
        self.output = GraphAttentionV2Layer(n_hidden, n_classes, 1,
                                            is_concat=False, dropout=dropout, share_weights=share_weights)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        """
  
        """
        x = self.dropout(x)
        x = self.layer1(x, adj_mat)
        x = self.activation(x)
        x = self.dropout(x)
        return self.output(x, adj_mat)#这一层没有激活函数


class Configs(GATConfigs):
    """
    和上面那个GAT一样的配置
    """

    share_weights: bool = False
    model: GATv2 = 'gat_v2_model'


@option(Configs.model)
def gat_v2_model(c: Configs):

    return GATv2(c.in_features, c.n_hidden, c.n_classes, c.n_heads, c.dropout, c.share_weights).to(c.device)


def main():
  
    conf = Configs()
  
    experiment.create(name='gatv2')
  
    experiment.configs(conf, {
  
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 5e-3,
        'optimizer.weight_decay': 5e-4,

        'dropout': 0.7,
    })

  
    with experiment.start():
  
        conf.run()



if __name__ == '__main__':
    main()
```
