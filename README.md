# TinyMind-Chinese-Character-Calligraphy-Recognition

* soft label
* data distulation
* TTA
* OHEM


## Experiments
* 128\*128的输入, ResNet18, 固定layer1,2,3的参数，使用了随机裁剪，随机旋转10的数据扩充，使用Adam，CrossEntropyLoss, 最终结果线上95.14
* 训练200个epoch，95.33
* ResNet18, 去掉全连接层，使用卷积层+global_average_pooling代替，acc@top5=96.21, 但是计算量反而增大
* ResNet18, 增加图像反色的数据扩充, acc@top5=96.42
* 3TTA acc@top5=
* OHEM acc@top5=
