# -*- coding = utf-8 -*-
# @Time : 2022/6/23 17:25
# @Author :
# @File : cancernet.py
# @Software : PyCharm

# 简历一个CNN神经网络：CancerNet
# 使用 3×3 个 CONV 滤波器
# 将这些过滤器堆叠在一起
# 执行最大池化
# 使用深度可分离卷积（效率更高，占用更少内存）

# 使用顺序 API 来构建 CancerNet 和 SeparableConv2D 来实现深度卷积。
# 类CancerNet具有静态方法构建，该构建需要四个参数 - 图像的宽度和高度，其深度（每个图像中的颜色通道数）以及网络将预测的类的数量，
# 这是2（0和1）。
# 在此方法中，初始化模型和形状。使用channels_first时，会更新形状和通道尺寸。
# 将定义三个DEPTHWISE_CONV = > RELU = > POOL层;每个都有更高的堆叠和更多的过滤器。softmax 分类器输出每个类的预测百分比。
# 返回模型

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class CancerNet:
  @staticmethod
  def build(width,height,depth,classes):
    model=Sequential()
    shape=(height,width,depth)
    channelDim=-1

    if K.image_data_format()=="channels_first":
      shape=(depth,height,width)
      channelDim=1

    model.add(SeparableConv2D(32, (3,3), padding="same",input_shape=shape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(SeparableConv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=channelDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model