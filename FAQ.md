## （一）换源

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
