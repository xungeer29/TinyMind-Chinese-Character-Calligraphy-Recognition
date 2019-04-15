# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    data_root = '/media/gfx/data1/DATA/Tinymind' # 数据集的根目录
    model = 'ResNet152' # ResNet152 使用的模型
    freeze = True # 是否冻结卷基层

    seed = 1000 # 固定随机种子
    num_workers = 12 # DataLoader 中的多线程数量
    num_classes = 100 # 分类类别数
    num_epochs = 300
    batch_size = 24 # 128
    lr = 0.01 # 初始lr
    width = 128 # 输入图像的宽
    height = 128 # 输入图像的高
    iter_smooth = 10 # 打印&记录log的频率

    # OHEM
    OHEM = False
    OHEM_ratio = 0.5

    # resume checkpoint
    resume = False #
    checkpoint = 'ResNet18_top5.pth' # 训练完成的模型名

    # smooth label
    smooth_label = False

config = DefaultConfigs()
