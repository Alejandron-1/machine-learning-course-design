# -*- coding = utf-8 -*-
# @Time : 2022/6/23 17:15
# @Author :
# @File : build_dataset.py
# @Software : PyCharm

# 按照上面提到的比例将我们的数据集拆分为训练，验证和测试集 - 80%用于训练（其中10%用于验证），20%用于测试。
# 使用Keras的ImageDataGenerator，我们将提取成批的图像，以避免一次在内存中为整个数据集腾出空间

from cancernet import config
from imutils import paths
import random, shutil, os

originalPaths=list(paths.list_images(config.INPUT_DATASET))
random.seed(7)
random.shuffle(originalPaths)

index=int(len(originalPaths)*config.TRAIN_SPLIT)
trainPaths=originalPaths[:index]
testPaths=originalPaths[index:]

index=int(len(trainPaths)*config.VAL_SPLIT)
valPaths=trainPaths[:index]
trainPaths=trainPaths[index:]

datasets=[("training", trainPaths, config.TRAIN_PATH),
          ("validation", valPaths, config.VAL_PATH),
          ("testing", testPaths, config.TEST_PATH)
]

for (setType, originalPaths, basePath) in datasets:
        print(f'Building {setType} set')

        if not os.path.exists(basePath):
                print(f'Building directory {base_path}')
                os.makedirs(basePath)

        for path in originalPaths:
                file=path.split(os.path.sep)[-1]
                label=file[-5:-4]

                labelPath=os.path.sep.join([basePath,label])
                if not os.path.exists(labelPath):
                        print(f'Building directory {labelPath}')
                        os.makedirs(labelPath)

                newPath=os.path.sep.join([labelPath, file])
                shutil.copy2(inputPath, newPath)


# 构建图像的原始路径列表，对列表进行随机排序。
# 将此列表的长度乘以 0.8 来计算索引，
# 以便我们可以对此列表进行切片以获取训练和测试数据集的子列表。
# 进一步计算一个索引，为训练数据集保存 10% 的列表以进行验证，并将其余部分保留用于训练本身

# 数据集是一个包含元组的列表，其中包含有关训练集、验证集和测试集的信息。它们保存每个路径和基本路径。
# 对于此列表中的每个 setType、path 和 base path，我们将打印“构建测试集”。
# 如果基本路径不存在，我们将创建目录。对于原始路径中的每个路径，我们将提取文件名和类标签。
# 我们将构建标签目录（0 或 1）的路径 - 如果它尚不存在，我们将显式创建此目录。
# 现在，我们将构建指向生成的映像的路径，并将映像复制到此处 - 它所属的位置
