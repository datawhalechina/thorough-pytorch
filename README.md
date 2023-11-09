# 深入浅出PyTorch
**在线阅读地址**：[深入浅出PyTorch-在线阅读](https://datawhalechina.github.io/thorough-pytorch/)

**配套视频教程**：[深入浅出PyTorch-bilibili视频](https://www.bilibili.com/video/BV1L44y1472Z)

## 一、项目简介

### 项目介绍及预期目标

欢迎来到thorough-PyTorch教程，这是一个专为希望在人工智能、数据科学等领域使用深度学习进行研究的学习者设计的课程。PyTorch以其灵活性、可读性和优越性能成为深度学习主流的框架。但是现有的PyTorch教程大部分为英文官方教程或只是某几个任务的特定教程，并不存在一个较为完整的中文PyTorch教程，为了能帮助更好的同学入门PyTorch，我们推出了thorough-PyTorch教程。

我们希望通过理论学习和实践操作的结合，帮助你从入门到熟练地掌握PyTorch工具，让你在理解深度学习的基本概念和实现技术的同时，能够通过动手实践提高你的技能熟练度。

我们的预期目标是，通过本教程，你不仅能够透彻理解PyTorch的基本知识和内容，也能通过项目实战磨练你的编程技能，从而更好地应用PyTorch进行深度学习，提高解决实际问题的能力。无论你是初学者还是已有一定基础的学习者，我们都将带领你深入理解并掌握PyTorch，让你在数据科学的道路上更进一步。

### 面向对象及前置技能

- 适用于所有具备基础 Python 能力，想要入门 PyTorch 的AI从业者，同学。
- 我们希望你具有基础的数理知识，深度学习基础知识，Python编程基础和使用搜索引擎解决问题的能力。

## 二、内容大纲

### 相关前置知识[选学]

1. 基础数理知识
2. 反向求导
3. 相关评价指标
4. Jupyter相关操作

### 一、PyTorch的简介和安装

1. PyTorch简介与安装

### 二、PyTorch基础知识

1. 张量及其运算
2. 自动求导机制简介
3. 并行计算、CUDA和cuDNN简介

### 三、PyTorch的主要组成模块

1. 深度学习流程需要的关键环节
2. 基本配置
3. 数据读入
4. 模型构建
5. 损失函数
6. 优化器
7. 训练和评估
8. 可视化

### 四、PyTorch基础实战

1. 基础实战——Fashion-MNIST时装分类
2. 基础实战——果蔬分类实战（notebook）

### 五、PyTorch模型定义

1. 模型定义方式
2. 利用模型块快速搭建复杂网络
3. 模型修改
4. 模型保存与读取

### 六、PyTorch进阶训练技巧

1. 自定义损失函数
2. 动态调整学习率
3. 模型微调-torchvision
4. 模型微调-timm
5. 半精度训练
6. 数据扩充
7. 超参数的修改及保存
8. PyTorch模型定义与进阶训练技巧

### 七、PyTorch可视化

1. 可视化网络结构
2. 可视化CNN卷积层
3. 使用TensorBoard可视化训练过程
4. 使用wandb可视化训练过程



## 三、贡献者

**贡献者**

- @[牛志康-核心贡献者](https://github.com/NoFish-528)（Datawhale成员-西安电子科技大学本科生）
- @[李嘉骐-核心贡献者](https://github.com/LiJiaqi96)（Datawhale成员-清华大学研究生）
- @[陈安东-贡献者](https://github.com/andongBlue)（Datawhale成员-哈尔滨工业大学博士生）
- @[刘洋-贡献者](https://github.com/liu-yang-maker)（Datawhale成员-中国科学院数学与系统科学研究所研究生）
- @[徐茗-贡献者](https://github.com/laffycat)（Datawhale成员-北京邮电大学本科生）
- @[邹雨衡-贡献者](https://github.com/logan-zou)（Datawhale成员-对外经济贸易大学研究生）
- @[潘笃驿-贡献者](https://github.com/limafang)（Datawhale成员-西安电子科技大学本科生）
- @[李鑫-贡献者](https://github.com/Mr-atomer)（Datawhale成员-西安电子科技大学本科生）

**其他**

- 非常感谢DataWhale成员 叶前坤 @[PureBuckwheat](https://github.com/PureBuckwheat) 和 胡锐锋 @[Relph1119](https://github.com/Relph1119) 对文档的细致校对！
- 如果有任何想法可以联系我们DataWhale也欢迎大家多多提出issue。
- 特别感谢以下为教程做出贡献的同学！并特别感谢MMYOLO的贡献者们！

<a href="https://github.com/datawhalechina/thorough-pytorch/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=datawhalechina/thorough-pytorch" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## 四、关于贡献

<details> 

本项目使用`Forking`工作流，具体参考[atlassian文档](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)大致步骤如下：

1. 在GitHub上Fork本仓库
2. Clone Fork后的个人仓库
3. 设置`upstream`仓库地址，并禁用`push`
4. 使用分支开发，课程分支名为`lecture{#NO}`，`#NO`保持两位，如`lecture07`，对应课程目录
5. PR之前保持与原始仓库的同步，之后发起PR请求

命令示例：

```shell
# fork
# clone
git clone git@github.com:USERNAME/thorough-pytorch.git
# set upstream
git remote add upstream git@github.com:datawhalechina/thorough-pytorch.git
# disable upstream push
git remote set-url --push upstream DISABLE
# verify
git remote -v
# some sample output:
# origin	git@github.com:NoFish-528/thorough-pytorch.git (fetch)
# origin	git@github.com:NoFish-528/thorough-pytorch.git (push)
# upstream	git@github.com:datawhalechina/thorough-pytorch.git (fetch)
# upstream	DISABLE (push)
# do your work
git checkout -b lecture07
# edit and commit and push your changes
git push -u origin lecture07
# keep your fork up to date
## fetch upstream main and merge with forked main branch
git fetch upstream
git checkout main
git merge upstream/main
## rebase brach and force push
git checkout lecture07
git rebase main
git push -f
```

### Commit Message

提交信息使用如下格式：`<type>: <short summary>`

```
<type>: <short summary>
  │            │
  │            └─⫸ Summary in present tense. Not capitalized. No period at the end.
  │
  └─⫸ Commit Type: [docs #NO]:others
```

`others`包括非课程相关的改动，如本`README.md`中的变动，`.gitignore`的调整等。
</details>


## 五、关注我们
<div align=center><img src="https://raw.githubusercontent.com/datawhalechina/easy-rl/master/docs/res/qrcode.jpeg" width = "250" height = "270" alt="Datawhale是一个专注AI领域的开源组织，以“for the learner，和学习者一起成长”为愿景，构建对学习者最有价值的开源学习社区。关注我们，一起学习成长。"></div>

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
