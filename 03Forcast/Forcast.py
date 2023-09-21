import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
import keras
import warnings

warnings.filterwarnings("ignore")
# features = pd.read_csv('')
# features.head()
f1 = matplotlib.font_manager.FontProperties('simhei', size=14)
plt.plot((1, 2, 3), (4, 3, -1))
plt.xlabel(u'横坐标', fontproperties=f1)  # 这一段
plt.ylabel(u'纵坐标', fontproperties=f1)  # 这一段
plt.show()  # 有了%matplotlib inline 就可以省掉plt.show()了
model = tf.keras.Sequential()
model.add(layers.Dense(16))
model.add(layers.Dense(32))
model.add(layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
              loss='mean_squared_error')
model.fit(input_features, lables, validation_split=0.25, epochs=10, batch_size=64)
model.summary()
# 加入正则化惩罚
model = tf.keras.Sequential()
model.add(layers.Dense(16, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.12(0.03)))
model.add(layers.Dense(32, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.12(0.03)))
model.add(layers.Dense(1, kernel_initializer='random_normal', kernel_regularizer=tf.keras.regularizers.12(0.03)))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
              loss='mean_squared_error')
model.fit(input_features, lables, validation_split=0.25, epochs=10, batch_size=64)
predict = model.predict(input_features)
predict.shape
