import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer  # 提取词干
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout, Embedding
from keras.layers import SpatialDropout1D  # 丢弃整个1D的特征图而不是丢弃单个元素，提高特征图之间的独立性
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

warnings.filterwarnings('ignore')
nltk.download("stopwords")  # 下载停用词
# 加载数据集
dataset = pd.read_csv("D:\\Code\\Sentiment\\training.1600000.processed.noemoticon.csv",
                      engine='python',
                      header=None,
                      encoding='Latin-1')
# 为数据集重置表头 header
dataset.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
# 丢弃无用的列
df = dataset.drop(['id', 'date', 'query', 'user_id'], axis=1)
# 停用词
stop_words = stopwords.words("english")
# 词干
stemmer = SnowballStemmer('english')
# 正则化表达式
text_cleaning_re = '@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'


# 对文本进行清洗
def preprocessing(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))  # 提取词干
            else:
                tokens.append(token)  # 直接保存单词
    return ' '.join(tokens)


# 对数据集中的text列中每行文本进行清洗
df.text = df.text.apply(lambda x: preprocessing(x))
MAX_WORDS = 100000  # 最大词汇量10万
MAX_SEQ_LENGTH = 30  # 最大序列长度30
train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=666, shuffle=True)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_dataset.text)
# 每个单词对应一个索引
word_index = tokenizer.word_index
# 训练集词汇表大小
vocab_size = len(word_index) + 1
# 固定每一条文本的长度
x_train = pad_sequences(tokenizer.texts_to_sequences(train_dataset.text),
                        maxlen=MAX_SEQ_LENGTH)

x_test = pad_sequences(tokenizer.texts_to_sequences(test_dataset.text),
                       maxlen=MAX_SEQ_LENGTH)
# 标签类别进行LabelEncoding，将类别编码成连续的编号
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_dataset.sentiment.tolist())
y_test = encoder.fit_transform(test_dataset.sentiment.tolist())
y_train = y_train.reshape(-1, 1)  # shape转置
y_test = y_test.reshape(-1, 1)

# word embedding 词嵌入： 将单词用特征向量来表示，这里使用预训练的词向量 GloVe
GloVe = "D:\\Code\\Sentiment\\glove.6B.300d.txt"
EMBEDDING_DIM = 300  # 300 维
BATCH_SIZE = 10000  # 批处理的大小
EPOCHS = 10  # 循环的次数
LR = 1e-3  # 学习率
MODEL_PATH = "./best_model.hdf5"  # 保存模型路径
# 构建字典： 格式： { 单词 ： 词嵌入向量}
embedding_index = {}

with open(GloVe, encoding='UTF-8') as f:
    for line in f:
        values = line.split()  # 按空格分割
        word = values[0]  # 第一个位置上是单词
        embeddings = np.asarray(values[1:], dtype="float32")  # 每个单词对应的词嵌入
        embedding_index[word] = embeddings  # 键值对

# 获取词嵌入矩阵
# vocab_size : 训练集词汇表大小，EMBEDDING_DIM : 词嵌入维度
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
num = 0

for word, values in embedding_index.items():
    embedding_vector = embedding_index.get(word)  # 单词对应的词嵌入向量
    if embedding_vector is not None:
        if num < vocab_size:
            embedding_matrix[num, :] = embedding_vector
        num += 1

# 模型搭建 设置输入序列长度 MAX_SEQ_LENGTH
sequence_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
embedding_layer = Embedding(vocab_size,  # 词汇表大小
                            EMBEDDING_DIM,  # 词嵌入维度
                            weights=[embedding_matrix],  # 预训练词嵌入
                            input_length=MAX_SEQ_LENGTH,  # 序列长度
                            trainable=False)

embedding_sequences = embedding_layer(sequence_input)

x = SpatialDropout1D(0.2)(embedding_sequences)
print(x.shape)
x = Conv1D(64, 5, activation='relu')(x)
print(x.shape)
x = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x)
print(x.shape)
x = Dense(512, activation='relu')(x)
print(x.shape)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
print(x.shape)
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(sequence_input, outputs)

# 模型变异
model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy', metrics=['accuracy'])
ReduceLR = ReduceLROnPlateau(factor=0.1, min_lr=0.01, monitor='val_loss', verbose=1)
# factor : 学习速率被降低的因数， 新的学习速率 = 学习率 * 因素
# min_lr : 学习率的下边界
# monitor : 被监测的数据

# 模型训练
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=(x_test, y_test),
                    callbacks=[ReduceLR])

# 绘制训练和验证结果
s, (one, two) = plt.subplots(2, 1)

one.plot(history.history['accuracy'], c='b')
one.plot(history.history['val_accuracy'], c='r')
one.set_title('Model Accuracy')
one.set_ylabel("accuracy")
one.set_xlabel("epoch")
one.legend(['LSTM train', 'LSTM val'], loc='upper left')

two.plot(history.history['loss'], c='m')
two.plot(history.history['val_loss'], c='c')
two.set_title('Model Loss')
two.set_ylabel('loss')
two.set_xlabel('epoch')


# 模型的输出概率在 0 - 1 之间，这里我们设定一个阈值： 0.5， 如果概率 > 0.5，则会 正面，否则，为负面评论
def Judge(score):
    return 1 if score > 0.5 else 0


# 模型在test上预测

scores = model.predict(x_test, verbose=1, batch_size=10000)
# 最终的预测结果

y_pred = [Judge(score) for score in scores]
y_test.squeeze()

import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.subplots as tls
import plotly.figure_factory as ff
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def plot_ROC_AUC(y_test, y_pred):
    fpr, tpr, t = roc_curve(y_test, y_pred)
    model_roc_auc = roc_auc_score(y_test, y_pred)  # 得分
    trace = go.Scatter(x=fpr, y=tpr,
                       name="ROC : " + str(model_roc_auc),
                       line=dict(color=('rgb(22, 96, 167)'), width=2), fill='tozeroy')
    fig = tls.make_subplots(rows=1, cols=1, print_grid=False, subplot_titles=("ROC_AUC_curve",))
    fig.append_trace(trace, 1, 1)
    fig.layout.xaxis.update(dict(title="FPR"), range=[0, 1.05])
    fig.layout.yaxis.update(dict(title="TPR", range=[0, 1.05]))
    fig.layout.titlefont.size = 14  # 标题字体大小
    py.iplot(fig)


plot_ROC_AUC(y_test.squeeze(), y_pred)


# 模型性能评估
from sklearn.metrics import confusion_matrix
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.subplots as tls
import plotly.figure_factory as ff


def show_matrics(y_test, y_pred):
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    trace1 = go.Heatmap(z=conf_matrix, x=["0(pred)", "1(pred)"],
                        y=["0(True)", "1(True)"], xgap=2, ygap=2,
                        colorscale='Viridis', showscale=False)
    # 根据混淆矩阵，获取对应的参数值
    TP = conf_matrix[1, 1]
    FN = conf_matrix[1, 0]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]

    # 计算accuracy, precision, recall, f1_score
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率
    precision = TP / (TP + FP)  # 精准率
    recall = TP / (TP + FN)  # 召回率
    F1_score = 2 * precision * recall / (precision + recall)

    # 显示以上四个指标
    show_metrics = pd.DataFrame(data=[[accuracy, precision, recall, F1_score]])
    show_metrics = show_metrics.T

    # 可视化显示
    colors = ['gold', 'lightgreen', 'lightcoral', 'lightskyblue']
    trace2 = go.Bar(x=show_metrics[0].values,
                    y=['Accuracy', 'Precision', 'Recall', 'F1_score'],
                    text=np.round_(show_metrics[0].values, 4),
                    textposition='auto',
                    orientation='h',
                    opacity=0.8,
                    marker=dict(color=colors, line=dict(color="#000000", width=1.5)))

    fig = tls.make_subplots(rows=2, cols=1, subplot_titles=('Confusion Matrix', 'Metrics'))
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)
    py.iplot(fig)


show_matrics(y_test.squeeze(), y_pred)
