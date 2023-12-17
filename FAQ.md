# FAQ
FAQ(Frequently Asked Question)文档记录一些大家经常问到的问题以及它的解决方案。

## pip&conda换源

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

## 使用 debugpy 调试 python 项目

> 在这里我们仅讲解如何在 vscode 中使用 debugpy 模块进行debug

关于使用 debugpy 调试的更多内容，可以参考：

1. [命令行pyd方式在vscode中优雅debug Python](https://zhuanlan.zhihu.com/p/615198529)
2. [Debugpy——如何使用VSCode调试无法直接执行的Python程序](https://zhuanlan.zhihu.com/p/560405414)
3. [在 Linux 上远程调试 Python 代码](https://learn.microsoft.com/zh-cn/visualstudio/python/debugging-python-code-on-remote-linux-machines?view=vs-2022)
### 为什么使用 debugpy 进行debug

在使用 debugpy 模块 debug 之前，我们还有很多调试的办法，例如：`print` 大法， pdb 调试，同时 vscode ，pycharm 等 ide 也提供了基于 pdb 的调试方法，但是这些方法或多或少都存在一定的问题

`print` 方法在调试启动时间长，运行代价大的项目上非常麻烦，而且要同时查看多个变量信息开销很大

pdb 调试，记忆语句多，学习成本高，调试不方便

利用 ide 调试，对于带参数的程序，需要写冗长的配置文件后才能调试，而且很难复用，而且无法调试通过 `.sh` 文件启动的python程序，在调试 python 模块的时候还需要专门配置，非常低效，有的计算集群计算和开发是分开的，在开发机开发完成后提交到集群上动态分配节点执行。由于动态分布的节点每次ip都不一样，因此没办法直接使用vscode调试。

而使用 debugpy 调试能够解决上面的绝大多数问题

### 使用 debugpy debug 的具体操作

首先确保你获取了 debugpy 模块
在命令行中执行

```shell
pip install debugpy
```
  
接着配置vscode调试配置文件

[![pi5NKMT.png](https://s11.ax1x.com/2023/12/17/pi5NKMT.png)](https://imgse.com/i/pi5NKMT)

选择侧边栏中运行与调试，创建 launch.json 文件，如果你看不到创建文件的选项的话，把之前的配置删掉就好了

配置文件内容如下：

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
需要改动的参数只有 connect 部分，修改调试要监听的地址与端口，与程序中设定的地址与端口一致就可以
,这里我选择调试本地的文件，所以host就写的localhost

接着在命令行中运行
```shell
python -m debugpy --listen 5678 --wait-for-client xxx.py
```
或者
```shell
python -m debugpy --listen 5678 --wait-for-client xxx.sh
```
启动后，由于设置了--wait-for-client选择，当前进程会等待你打开调试接收端口(我们在launch.json文件中配置的地址)

[![pi5UUts.png](https://s11.ax1x.com/2023/12/17/pi5UUts.png)](https://imgse.com/i/pi5UUts)

启动后就能够使用vscode正常调试了

[![pi5U3X8.md.png](https://s11.ax1x.com/2023/12/17/pi5U3X8.md.png)](https://imgse.com/i/pi5U3X8)

同时，我们可以使用 alias 添加别称的方式对debug命令进行简化。

```shell
python -m debugpy --listen 5678 --wait-for-client
```

我们通过在Linux系统中的 ~/.bashrc 文件中添加以下命令

```
alias pyd='python -m debugpy --wait-for-client --listen 5678'
```

然后执行

```shell
source ~/.bashrc
```

下次启动时就可以直接使用 pyd 命令代替

```
pyd xxx.py
```

也可以不在命令行中指定连接地址，在代码中添加如下代码来设定连接

```python
import debugpy;debugpy.connect(('localhost', 5678))
```

使用上面的方法时，要修改launch.json文件内容

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

修改好后先启动python远程调试，接着在命令行正常运行命令就能够启动调试了