from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot
import numpy as np
import tensorflow as tf
from keras import layers


# 载入MNIST数据集，如果指定地址下没有下载好的数据，那么TensorFlow会自动在网站上下载数据
mnist = input_data.read_data_sets("/MNIST_data")

# 打印训练数据大小
print("Training data size: ", mnist.train.num_examples)
# 打印验证集数据大小
print("Validating data size: ", mnist.validation.num_examples)
# 打印测试集数据大小
print("Testing data size: ", mnist.test.num_examples)

# 批量获取数据和标签【使用next_batch(batch_size)】
# x_train 是数据，y_train 是对应的标签, batch_size=50000，批量读取50000张样本
x_train, y_train = mnist.train.next_batch(50000)

# 先获取数据值
# x_valid 是验证集的所有数据
x_valid = mnist.validation.images

# 再获取标签值，label=[0,0,...,0,1],是一个1*10的向量
# y_valid 是验证集所有数据的标签
y_valid = mnist.validation.labels
# 显示测试集第一张
# pyplot.imshow(x_train[0].reshape((28, 28)), cmap='gray')
# pyplot.show()
# print(y_train[0])

model = tf.keras.Sequential()
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# 回归问题中设置为1，是为了得到一个结果值。
# 现在设置为10个，是有十个类别，数据属于哪个类别
# tf.losses.CategoricalCrossentropy
# keras.losses.CategoricalCrossentropy
try:
    model.compile(optimizer=tf.keras.optimizers.Adam(0.005),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(x_train, y_train, epochs=6, batch_size=64, validation_data=(x_valid, y_valid))
finally:
    print()
