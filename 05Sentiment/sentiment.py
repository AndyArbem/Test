import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from torchtext import data
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader

warnings.filterwarnings('ignore')
dataset = pd.read_csv("D:\\Code\\Sentiment\\training.1600000.processed.noemoticon.csv",
                      engine='python',
                      header=None,
                      encoding='Latin-1')

# print(dataset.describe())
dataset['sentiment_category'] = dataset[0].astype('category')
dataset['sentiment_category'].value_counts()
dataset['sentiment'] = dataset['sentiment_category'].cat.codes

dataset.to_csv('training-processed.csv', header=None, index=None)
# 随机选择10000个样本当作测试集
dataset.sample(10000).to_csv("test_sample.csv", header=None, index=None)
LABEL = data.LabelField()  # 标签
TWEET = data.Field(lower=True)  # 内容/文本
# 设置表头
fields = [('score', None), ('id', None), ('date', None), ('query', None),
          ('name', None), ('tweet', TWEET), ('category', None), ('label', LABEL)]

# 读取数据
twitterDataset = data.TabularDataset(
    path='training-processed.csv',
    format='CSV',
    fields=fields,
    skip_header=False
)

# 分离 train, test, val
train, test, val = twitterDataset.split(split_ratio=[0.8, 0.1, 0.1], stratified=True, strata_field='label')
# 构建词汇表
vocab_size = 20000
TWEET.build_vocab(train, max_size=vocab_size)
LABEL.build_vocab(train)

device = "cuda" if torch.cuda.is_available() else "cpu"
# 文本批处理，即一批一批地读取数据
train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test),
                                                             batch_size=32,
                                                             device=device,
                                                             sort_within_batch=True,
                                                             sort_key=lambda x: len(x.tweet))


class simple_LSTM(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size):
        super(simple_LSTM, self).__init__()  # 调用父类的构造方法
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # vocab_size词汇表大小， embedding_dim词嵌入维度
        self.encoder = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, 2)  # 全连接层

    def forward(self, seq):
        output, (hidden, cell) = self.encoder(self.embedding(seq))
        # output :  torch.Size([24, 32, 100])
        # hidden :  torch.Size([1, 32, 100])
        # cell :  torch.Size([1, 32, 100])
        preds = self.predictor(hidden.squeeze(0))
        return preds


lstm_model = simple_LSTM(hidden_size=100, embedding_dim=300, vocab_size=20002)

lstm_model.to(device)
# 优化器
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# 损失函数
criterion = nn.CrossEntropyLoss()  # 多分类 （负面、正面、中性）


def train_val_test(model, optimizer, criterion, train_iter, val_iter, test_iter, epochs):
    for epoch in range(1, epochs + 1):
        train_loss = 0.0  # 训练损失
        val_loss = 0.0  # 验证损失
        model.train()  # 声明开始训练
        for indices, batch in enumerate(train_iter):
            optimizer.zero_grad()  # 梯度置0
            outputs = model(batch.tweet)  # 预测后输出 outputs shape :  torch.Size([32, 2])
            # batch.label shape :  torch.Size([32])
            loss = criterion(outputs, batch.label)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # batch.tweet shape :  torch.Size([26, 32]) --> 26:序列长度， 32:一个batch_size的大小
            train_loss += loss.data.item() * batch.tweet.size(0)  # 累计每一批的损失值
        train_loss /= len(train_iter)  # 计算平均损失 len(train_iter) :  40000
        print("Epoch : {}, Train Loss : {:.2f}".format(epoch, train_loss))

        model.eval()  # 声明模型验证
        for indices, batch in enumerate(val_iter):
            context = batch.tweet.to(device)  # 部署到device上
            target = batch.label.to(device)
            pred = model(context)  # 模型预测
            loss = criterion(pred, target)  # 计算损失 len(val_iter) :  5000
            val_loss += loss.item() * context.size(0)  # 累计每一批的损失值
        val_loss /= len(val_iter)  # 计算平均损失
        print("Epoch : {}, Val Loss : {:.2f}".format(epoch, val_loss))

        model.eval()  # 声明
        correct = 0.0  # 计算正确率
        test_loss = 0.0  # 测试损失
        with torch.no_grad():  # 不进行梯度计算
            for idx, batch in enumerate(test_iter):
                context = batch.tweet.to(device)  # 部署到device上
                target = batch.label.to(device)
                outputs = model(context)  # 输出
                loss = criterion(outputs, target)  # 计算损失
                test_loss += loss.item() * context.size(0)  # 累计每一批的损失值
                # 获取最大预测值索引
                preds = outputs.argmax(1)
                # 累计正确数
                correct += preds.eq(target.view_as(preds)).sum().item()
            test_loss /= len(test_iter)  # 平均损失 len(test_iter) :  5000
            print("Epoch : {}, Test Loss : {:.2f}".format(epoch, test_loss))
            print("Accuracy : {}".format(100 * correct / (len(test_iter) * batch.tweet.size(1))))


# 开始训练和验证
train_val_test(lstm_model, optimizer, criterion, train_iter, val_iter, test_iter, epochs=5)
