# TinyMind-Chinese-Character-Calligraphy-Recognition
竞赛官网: [https://www.tinymind.cn/competitions/41#overview](https://www.tinymind.cn/competitions/41#overview)
## Dataset
竞赛数据提供100个汉字书法单字，包括碑帖，手写书法，古汉字等等。图片全部为单通道灰度jpg，宽高不定。

训练集: [https://pan.baidu.com/s/1UxvN7nVpa0cuY1A-0B8gjg ](https://pan.baidu.com/s/1UxvN7nVpa0cuY1A-0B8gjg) 密码: `aujd`

测试集: [https://pan.baidu.com/s/1tzMYlrNY4XeMadipLCPzTw](https://pan.baidu.com/s/1tzMYlrNY4XeMadipLCPzTw) 密码: `4y9k`

## ENVS
* Ubuntu16.04
* python==2.7
* pytorch==0.4.1

## File Structure
```
TinyMind-Chinese-Character-Calligraphy-Recognition/
▾ checkpoints/
    ResNet18_top1.pth
    ResNet18_top5.pth
▾ data/
    label_list.txt
    test1.txt
    test2.txt
    train.txt
    valid.txt
▾ dataset/
    __init__.py
    create_img_list.py
    dataset.py
▾ figs/
    acc.jpg
    fig1.jpg
▾ log/
    log.txt
▾ metrics/
    __init__.py
    metric.py
▾ networks/
    __init__.py
    lr_schedule.py
    network.py
▾ utils/
    __init__.py
    plot.py
  __init__.py
  config.py
  inference.py
  README.md
  train.py  
```
## Network Architecture
全卷积的网络，将ResNet的第一层替换为单通道输入，3层3\* 3的卷积核，
ResNet的网络layer4的最后增加一层卷积卷基层，并使用Global Average Pooling 代替全连接

损失函数: 交叉熵 Cross Entropy Loss

优化器: Adam

## RUN
* STEP0
```
git clone https://github.com/xungeer29/TinyMind-Chinese-Character-Calligraphy-Recognition
cd TinyMind-Chinese-Character-Calligraphy-Recognition
```
* STEP1
添加文件搜索路径，更改数据集根目录

将所有的`.py`文件的`sys.path.append`中添加的路径改为自己的项目路径

更改`config.py`中的`data_root`为数据集存放的根目录
* STEP2
划分训练集和本地验证集

```
python dataset/create_img_list.py
```

* STEP3
train

```
python train.py
```

* STEP4
inference
```
python inference.py
```

## TODO
* soft label
* data distulation
* TTA
* OHEM



## Experiments
* 128\*128的输入, ResNet18, 固定layer1,2,3的参数，使用了随机裁剪，随机旋转10的数据扩充，使用Adam，CrossEntropyLoss, 最终结果线上95.14
* 训练200个epoch，95.33
* ResNet18, 去掉全连接层，使用卷积层+global_average_pooling代替，acc@top5=96.21, 但是计算量反而增大
* ResNet18, 增加图像反色的数据扩充, acc@top5=96.42
* 3TTA acc@top5=96.92
* OHEM acc@top5=
* ResNet152 acc@top5=
