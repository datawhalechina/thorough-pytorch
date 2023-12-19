# FAQ
FAQ(Frequently Asked Question)文档记录一些大家经常问到的问题以及它的解决方案。

## 一、关于pip&conda换源的问题

关于换源的更多内容，可以参考

1. [PyPI 软件仓库镜像使用帮助](https://help.mirrors.cernet.edu.cn/pypi/)
2. [TUNA Anaconda 镜像使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

### 1. conda更换镜像源

​	Anaconda换源主要有两种：在Anaconda Prompt/Terminal换源以及修改`.condarc`文件。

#### 1.1 在终端使用指令换源

步骤一：打开Anaconda Prompt（如下图）/ Linux Terminal

步骤二：依次输入以下指令

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
```

步骤三：运行 `conda clean -i` 清除索引缓存，保证用的是镜像站提供的索引。

#### 1.2 修改`.condarc`文件

步骤一：找到`.condarc`

1. Windows
   1. Windows 用户无法直接创建名为 `.condarc` 的文件，可先执行 `conda config --set show_channel_urls yes` 生成该文件之后再修改。
   2. C盘找到`.condarc`文件，路径大致为`C:\Users\(用户名)\.condarc`
2. Linux/macOS文件路径大致在`${HOME}/.condarc`（`~/.condarc`）

步骤二：修改利用记事本或者vim/emacs/其他工具打开该文件，将其中内容改为以下：

```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
```

步骤三：运行 `conda clean -i` 清除索引缓存，保证用的是镜像站提供的索引。

步骤四：查看/删除当前使用的镜像源

```
conda config --show channels		# 查看anaconda已存在的源
conda config --set show_channel_urls yes		# 设置搜索时显示的通道地址
conda config --remove-key channels		# 删除已存在的镜像源
```



### 2. pip更换镜像源方法

国内常见的镜像源网址如下：

1. 清华大学：https://pypi.tuna.tsinghua.edu.cn/simple/
2. 中国科学技术大学：https://pypi.mirrors.ustc.edu.cn/simple/
3. 阿里云：https://mirrors.aliyun.com/pypi/simple/
4. 豆瓣：https://pypi.douban.com/simple/

首先，我们可以在终端输入以下命令，查看我们当前使用的是哪个镜像源：

```shell
pip config get global.index-url		# 查看当前默认的pip源地址
```

我们将在下面以替换清华源为例进行介绍，主要介绍临时更换和永久更换。

1. 临时更换

   如果在安装时只想临时更换镜像源，可以直接在安装指令后添加`-i https://pypi.tuna.tsinghua.edu.cn/simple`

   例如 `pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple`

2. 永久更换（terminal更换）

   在window下，我们可以WIN+R输入CMD。在Linux下，我们可以打开终端，输入以下指令

   ```shell
   python -m pip install --upgrade pip #将pip更新至最新版本
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

   想要永久更换，我们也可以直接修改配置文件即可。

3. 永久更换（修改配置文件）

   1. 步骤一：

      Win系统：在`User\pip\pip.ini`路径下新建`pip.ini`文件

      Linux系统：在`~/.pip/pip.conf`路径下新建`pip.conf`文件

   2. 步骤二：

      打开文件建立的文件，输入以下内容，并保存。

      ```shell
      [global]
      trusted-host=pypi.tuna.tsinghua.edu.cn
      index-url=https://pypi.tuna.tsinghua.edu.cn/simple
      ```

### 3. 换源安装pytorch

1. pip安装临时换源

```shell
pip install torch torchvision -i 镜像地址
# 指定版本换清华源安装示例
#pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple
```

2. conda安装临时换源

```shell
conda install pytorch torchvision torchaudio –c 镜像地址
conda install cudatoolkit=版本–c 镜像地址

# 指定版本换清华源安装示例
# conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 –c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/win-64/
# conda install cudatoolkit=11.3 –c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
```

## 二、如何使用 debugpy 调试 python 项目

> 在这里我们仅讲解如何在 vscode 中使用 debugpy 模块进行debug

关于使用 debugpy 调试的更多内容，可以参考：

1. [命令行pyd方式在vscode中优雅debug Python](https://zhuanlan.zhihu.com/p/615198529)
2. [Debugpy——如何使用VSCode调试无法直接执行的Python程序](https://zhuanlan.zhihu.com/p/560405414)
3. [在 Linux 上远程调试 Python 代码](https://learn.microsoft.com/zh-cn/visualstudio/python/debugging-python-code-on-remote-linux-machines?view=vs-2022)
### 1. 为什么使用 debugpy 进行debug

在使用 debugpy 模块 debug 之前，我们还有很多调试的办法，例如：`print` 大法， pdb 调试，同时 vscode ，pycharm 等 ide 也提供了基于 pdb 的调试方法，但是这些方法或多或少都存在一定的问题

- `print` 方法在调试启动时间长，运行代价大的项目上非常麻烦，而且要同时查看多个变量信息开销很大
- `pdb` 调试，记忆语句多，学习成本高，调试不方便
- `ide`调试，对于带参数的程序，需要写冗长的配置文件后才能调试，而且很难复用，而且无法调试通过 `.sh` 文件启动的python程序，在调试 python 模块的时候还需要专门配置，非常低效，有的计算集群计算和开发是分开的，在开发机开发完成后提交到集群上动态分配节点执行。由于动态分布的节点每次ip都不一样，因此没办法直接使用vscode调试。

而使用 debugpy 调试能够解决上面的绝大多数问题

### 2. 使用 debugpy debug 的具体操作

#### 2.1 debugpy的安装

在使用debugpy前，我们首先需要安装 debugpy 模块，如果没有安装的话只需要在终端中执行

```shell
pip install debugpy
```

#### 2.2 配置launch.json文件

我们点击vscode侧边栏中的运行与调试，创建 launch.json 文件

<img src="https://github.com/datawhalechina/thorough-pytorch/assets/73390819/c6ec6d84-1ee9-441a-b4a3-8c695d78f5ca" alt="set_launch" align="center" style="zoom:80%;" />

我们只需要将以下的内容复制到你的配置文件即可，需要改动的参数只有 connect 部分，修改调试要监听的地址与端口（如果没有特殊需求，可以不用修改下面的内容）：

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Attach",
            "type": "python",
            "request": "attach",
            "connect": {
              "host": "localhost",
              "port": 5678
            }
          }
    ]
}

```
#### 2.3 debugpy使用步骤

我们执行一个python文件往往使用以下的方式

```shell
python xxx.py
```

当我们配置好debugpy后，我们只需要将`python`换成`python -m debugpy --listen 设定的端口 --wait-for-client`即可，我们在命令行中运行如下的命令即可。

```shell
python -m debugpy --listen 5678 --wait-for-client xxx.py
# 或者启动的shell scripts也可以
# python -m debugpy --listen 5678 --wait-for-client xxx.sh
```
启动后，由于设置了`--wait-for-client`选择，当前进程会等待你打开调试接收端口(我们在launch.json文件中配置的地址)

<img src="https://github.com/datawhalechina/thorough-pytorch/assets/73390819/b5be1636-cc5e-4c37-8621-cdcbcc3703d9" alt="run" align="center" style="zoom:80%;" />

当我们点击启动按键后就能够使用vscode正常调试了

<img src="https://github.com/datawhalechina/thorough-pytorch/assets/73390819/0caaec85-6303-4a7f-8c8a-563b0af29063" alt="start_debug" align="center" style="zoom:80%;" />

但是每一次输入那么长的命令也显得过于复杂，因此我们可以使用 alias 添加别称的方式对debug命令进行简化。

```shell
# 原始的命令
python -m debugpy --listen 5678 --wait-for-client
```

我们可以在Linux系统中的 `~/.bashrc` 文件中添加以下命令

```
alias pyd='python -m debugpy --wait-for-client --listen 5678'
```

保存完后，我们只需要执行即可

```shell
source ~/.bashrc
```

当我们下次启动时就可以直接使用 `pyd` 命令代替`python -m debugpy --wait-for-client --listen 5678`，这样会使得我们的启动更加的方便简洁。

```
pyd xxx.py
```

同样，我们也可以不在命令行中指定连接地址，而是在代码中添加如下代码来设定连接（我们更加推荐第一种方式）。

```python
import debugpy;debugpy.connect(('localhost', 5678))
```

当我们使用上面的方法时，我们需要修改`launch.json`文件内容

```json
{
  "version": "0.2.0",
  "configurations": [
      {
          "name": "Python: Attach",
          "type": "python",
          "request": "attach",
          "listen": {
              "host": "localhost",
              "port": 5678
          },
          "pathMappings": [
              {
                  "localRoot": "${workspaceFolder}", 
                  "remoteRoot": "."
              }
          ]
      }
  ]
}
```

修改好后先启动python远程调试，接着在命令行正常运行命令就能够启动调试了。在学习一些成熟的基于PyTorch二次开发的框架时，debug可以帮助我们更好的了解整个项目的组成以及运行的逻辑，而debugpy可以大大提高你的debug的效率。