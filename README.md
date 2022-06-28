# 机器学习课设
# 交通标志识别

## 交通标志识别的作用：
    有几种不同类型的交通标志，如限速，禁止进入，交通信号灯，左转或右转，儿童交叉口，不通过重型车辆等。交通标志分类是识别交通标志所属类别的过程。
    在本项目中，通过构建一个深度神经网络模型，可以将图像中存在的交通标志分类为不同的类别。通过该模型，我们能够读取和理解交通标志，这对所有自动驾驶汽车来说都是一项非常重要的任务。

## 交通标志数据集：
    数据集包含超过50，000张不同交通标志的图像。它被进一步分为43个不同的类。数据集变化很大，一些类有许多图像，而一些类有很少的图像。数据集的大小约为 300 MB。数据集有一个训练文件夹，其中包含每个类中的图像和一个测试文件夹，我们将用于测试我们的模型。

## 构建此交通标志分类模型的方法分为四个步骤：


1. 浏览分析数据集
1. 构建 CNN 模型
1. 训练和验证模型
1. 使用测试数据集测试模型

### 步骤1：浏览分析数据集
“train”文件夹包含 43 个文件夹，每个文件夹代表不同的类别。文件夹的范围是从 0 到 42。迭代所有类，并在数据和标签列表中附加图像及其各自的标签。
所有图像及其标签存储到列表（数据和标签）中，再将列表转换为 numpy 数组，以便提供给模型使用，数据的形状为 （39209， 30， 30， 3），这意味着有 39，209 张大小为 30×30 像素的图像，最后的 3 表示数据包含彩色图像（RGB 值）。

```
folders = os.listdir(train_path)

train_number = [] # 训练集中该类别的数量
class_num = []  # 类别

for folder in folders:
    if folder=='.ipynb_checkpoints':
        continue
    print(folder)
    print(classes[int(folder)])
    train_files = os.listdir(train_path + '/' + folder)
    train_number.append(len(train_files))
    class_num.append(classes[int(folder)])
    
# 根据每个类中的图像数量对数据集进行排序
zipped_lists = zip(train_number, class_num)
sorted_pairs = sorted(zipped_lists)
tuples = zip(*sorted_pairs)
train_number, class_num = [ list(tuple) for tuple in  tuples]

# 绘制每个类中的图像数量
plt.figure(figsize=(21,10))  
plt.bar(class_num, train_number)
plt.xticks(class_num, rotation='vertical')
plt.show()

# 测试数据的 25 张随机图像
import random
from matplotlib.image import imread

test = pd.read_csv(data_dir + '/Test.csv')
imgs = test["Path"].values

plt.figure(figsize=(25,25))

for i in range(1,26):
    plt.subplot(5,5,i)
    random_img_path = data_dir + '/' + random.choice(imgs)
    rand_img = imread(random_img_path)
    plt.imshow(rand_img)
    plt.grid(b=None)
    plt.xlabel(rand_img.shape[1], fontsize = 20)#图片宽度
    plt.ylabel(rand_img.shape[0], fontsize = 20)#图片高度

# 训练数据：
image_data = []
image_labels = []

for i in range(NUM_CATEGORIES):
    path = data_dir + '/Train/' + str(i)
    if i == 43:
        continue
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
            image_data.append(np.array(resize_image))
            image_labels.append(i)
        except:
            print("Error in " + img)

# 将 list 更改为 numpy 数组
image_data = np.array(image_data)
image_labels = np.array(image_labels)

print(image_data.shape, image_labels.shape)

# 打乱数据集

shuffle_indexes = np.arange(image_data.shape[0])
np.random.shuffle(shuffle_indexes)
image_data = image_data[shuffle_indexes]
image_labels = image_labels[shuffle_indexes]

# 将数据拆分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)
# 图形归一化
X_train = X_train/255 
X_val = X_val/255

print("X_train.shape", X_train.shape)
print("X_valid.shape", X_val.shape)
print("y_train.shape", y_train.shape)
print("y_valid.shape", y_val.shape)

# 独热编码：
y_train = keras.utils.to_categorical(y_train, NUM_CATEGORIES)
y_val = keras.utils.to_categorical(y_val, NUM_CATEGORIES)
# 分为了43个独热编码：
print(y_train.shape)
print(y_val.shape)
```
使用 train_test_split（） 方法来拆分训练和测试数据：
使用to_categorical方法将y_train和t_test中存在的标签转换为单热编码：

### 步骤2：构建 CNN 模型
把图像分类到各自的类别中，构建一个CNN模型（卷积神经网络）（多分类）：


```
####################################################################
# 模型的训练：
model = keras.models.Sequential([    
    #卷积层VGG
    #图像空间的2维卷积
	# Conv2D层 滤波器=16 kernel_size=3*3 激活函数='relu' 输入input_shape
	# Conv2D层 滤波器=32 kernel_size=3*3 激活函数='relu' 
	# 最大池化MaxPool2D层 pool_size=(2,2)
	# 
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,channels)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    # 全连接层 512个节点 激活函数='relu'
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'), # 全连接层
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),

    #密集层43个节点，激活='softmax'
    keras.layers.Dense(43, activation='softmax')
])

lr = 0.001
# 训练30次
epochs = 30
# Adam优化器编译模型，多分类损失用categorical_crossentropy
opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# 扩充数据并训练模型
# 批处理保存图片
aug = ImageDataGenerator(
    rotation_range=10, # 整数，随机选择图片的角度
    zoom_range=0.15,  # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数
    width_shift_range=0.1, # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

model.summary()            # 显示输出网络结构

history = model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=epochs, validation_data=(X_val, y_val))

model.save('traffic_classifier.h5')
##################################################################

```


### 步骤3：验证模型

```
# 验证评估模型：
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
```


### 步骤4：使用测试数据集测试模型

```
# 加载测试数据并运行预测
test = pd.read_csv(data_dir + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data =[]

for img in imgs:
    try:
        image = cv2.imread(data_dir + '/' +img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)
X_test = np.array(data)
X_test = X_test/255

pred = model.predict_classes(X_test)

# 测试集的精确度：
print('Test Data accuracy: ',accuracy_score(labels, pred)*100)

# 可视化混淆矩阵：
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, pred)

import seaborn as sns
df_cm = pd.DataFrame(cf, index = classes,  columns = classes)
plt.figure(figsize = (20,20))
sns.heatmap(df_cm, annot=True)

# 分类报告：
from sklearn.metrics import classification_report

print(classification_report(labels, pred))

# 测试数据的预测： 获取预测的精确度
plt.figure(figsize = (25, 25))

start_index = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = pred[start_index + i]
    actual = labels[start_index + i]
    col = 'g'
    if prediction != actual:
        col = 'r'
    plt.xlabel('Actual={} || Pred={}'.format(actual, prediction), color = col)
    plt.imshow(X_test[start_index + i])
plt.show()

```

