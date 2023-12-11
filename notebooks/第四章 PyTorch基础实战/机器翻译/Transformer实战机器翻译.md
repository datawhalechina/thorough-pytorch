# Transformer 实战机器翻译

## 引言

自然语言处理与图像具有显著差异，自然语言处理任务也具有其独特性。相对于 CNN（卷积神经网络）在 CV 中的霸主地位，在很长一段时间里，RNN（循环神经网络）、LSTM（长短期记忆递归神经网络）占据了 NLP 的主流地位。作为针对序列建模的模型，RNN、LSTM 在以序列为主要呈现形式的 NLP 任务上展现出远超 CNN 的卓越性能。​但是 RNN、LSTM 虽然在处理自然语言处理的序列建模任务中得天独厚，却也有着难以忽视的缺陷：

​1. RNN 为单向依序计算，序列需要依次输入、串行计算，限制了计算机的并行计算能力，导致时间成本过高。

​2. RNN 难以捕捉长期依赖问题，即对于极长序列，RNN 难以捕捉远距离输入之间的关系。虽然 LSTM 通过门机制对此进行了一定优化，但 RNN 对长期依赖问题的捕捉能力依旧是不如人意的。

​针对上述两个问题，2017年，Vaswani 等人发表了论文《Attention Is All You Need》，抛弃了传统的 CNN、RNN 架构，提出了一种全新的完全基于 attention 机制的模型——Transformer，解决了上述问题，在较小的时间成本下取得了多个任务的 the-state-of-art 效果，并为自然语言处理任务提供了新的思路。自此，attention 机制进入自然语言处理任务的主流架构，在 Transformer 的基础上，诞生了预训练-微调范式的多种经典模型如 Bert、GPT、T5 等。当然，同样是在 Transformer 的肩膀上，引入了 RLHF 机制、实现了大量参数建模的 ChatGPT 则带领 NLP 进入了全新的大模型时代。但不管是预训练-微调范式的主流模型 Bert，还是大模型时代的主流模型 ChatGPT、LLaMA，Transformer 都是其最坚实的基座。

本文将从该论文出发，结合 Transformer 的具体代码实现，基于经典 NLP 任务——机器翻译，来深入讲解如何使用 Pytorch 搭建 Transformer 模型并解决机器翻译任务。

本文的参考论文为：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

本文参考的代码实现仓库包括：[NanoGPT](https://github.com/karpathy/nanoGPT)、[ChineseNMT](https://github.com/hemingkx/ChineseNMT)、[transformer-translator-pytorch](https://github.com/devjwsong/transformer-translator-pytorch)