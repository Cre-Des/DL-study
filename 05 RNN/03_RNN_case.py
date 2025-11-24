"""

AI歌词生成器

构建词表
    - 获取文本数据
    - 分词，并进行去重
    - 构建词表
构建数据集对象
构建网络模型
    - 词嵌入层: 用于将语料转换为词向量
    - 循环网络层: 提取句子语义
    - 全连接层: 输出对词典中每个词的预测概率
模型训练
模型测试
"""

import torch
import jieba
from numpy.ma.core import shape
from numpy.ma.extras import unique
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

# 1. 获取文本数据, 分词，并进行去重, 构建词表
def build_vocab():
    # 定义变量，记录去重后的所有词，每行文本分词结果
    unique_words, all_words = [], []
    # 遍历数据集，获取每行文本
    for line in open('./data/jaychou_lyrics.txt','r',encoding='utf-8'):
        # 分词
        words = jieba.lcut(line) # ['唱歌', '要', '拿', '最佳', '男歌手', '\n']
        # 添加到all_words中
        all_words.append(words)
        # 去重
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    # 统计语料中的词数
    word_count = len(unique_words)
    # print(f'语料中的词数: {len(unique_words)}') # 5703
    # 构建词表，字典形式，key是词，value是索引
    word_to_index = {word:index for index, word in enumerate(unique_words)}
    # 歌词文本用索引表示
    corpus_idx = []
    # 遍历每行文本
    for words in all_words:
        # 定义一个空列表，用于存储每行文本的索引表示
        tmp = []
        # 遍历每行文本中的词
        for word in words:
            tmp.append(word_to_index[word])
        # 在每行词之间添加空格隔开
        tmp.append(word_to_index[' '])
        # 获取文档中每个词的索引， 添加到corpus_idx中
        corpus_idx.extend(tmp)
    # 返回结果：唯一词列表（5703），词表{‘想要’：0, ... :5702}, 去重后词的数量，歌词文本用词表索引表示
    return unique_words, word_to_index, word_count, corpus_idx

# 2. 数据预处理，构建数据集对象
# 构建数据集对象 继承torch.utils.data.Dataset
class LyricDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        # 文档数据中的索引
        self.corpus_idx = corpus_idx
        # 每个句子中词的个数
        self.num_chars = num_chars
        # 文档数据中词的数量，不去重
        self.word_count = len(self.corpus_idx)
        # 句子数量
        self.number = self.word_count // self.num_chars

    # 当使用 len(obj) 时，自动调用该方法
    def __len__(self):
        """
        返回句子数量
        """
        return self.number

    # 当使用 obj[i] 时，自动调用该方法
    def __getitem__(self, idx):
        """
        idx 表示词的索引，并将其修正索引值到文档范围内
        """
        # 确保索引 start 在文档合法范围内
        start = min(max(idx,0), self.word_count - self.num_chars - 1)
        # 确保索引 end 在文档合法范围内
        end = start + self.num_chars
        # 输入值，从文档中取出 start 到 end 的索引的词 -> 作为 x
        x = self.corpus_idx[start:end]
        # 输出值，从文档中取出 start + 1 到 end 的索引的词 -> 作为 y
        y = self.corpus_idx[start+1:end+1]
        # 返回结果：输入值，输出值 -> 输入值和输出值都为张量
        return torch.tensor(x), torch.tensor(y)

# 3. 构建网络模型
class TextGenerator(nn.Module):
#     - 词嵌入层: 用于将语料转换为词向量
#     - 循环网络层: 提取句子语义
#     - 全连接层: 输出对词典中每个词的预测概率
    def __init__(self, unique_word_count):
        super().__init__()
        # 初始化词嵌入层   输入参数： 词典中词的数量，词向量的维度
        self.embedding = nn.Embedding(unique_word_count, 128)
        # 初始化循环网络层 输入参数： 词向量的维度x_t（RNN看到的细节）； 循环网络层的隐藏状态维度h（类比大脑能记住的细节）； 循环网络层的层数
        self.rnn = nn.RNN(128, 256, 1)
        # 输出层 输入参数： 循环网络层的输出维度 词典中词的数量
        self.fc = nn.Linear(256, unique_word_count)

    def forward(self, inputs,  hidden):
        # 初始化词嵌入层
        # x 格式 ： （batch句子的数量，句子的长度，词向量维度）
        x = self.embedding(inputs)
        # 循环网络层
        # rnn格式：(句子的长度，batch句子的数量，隐藏层维度)
        x, hidden = self.rnn(x.transpose(0,1), hidden)
        # 全连接层 输入内容需要为二维数据， 词的数量 * 词的维度
        # 输入维度： 句子数量 * 句子长度 ，词向量维度
        # 输出维度： 句子数量 * 句子长度 ，词典中词的数量
        output = self.fc(x.reshape(shape=(-1, x.shape[-1])))
        return output, hidden

    # 初始化隐藏层 不会自动调用
    def init_hidden(self, batch_size):
        # 隐藏层初始化 [网络层数，batch，隐藏层向量维度]
        return torch.zeros(1, batch_size, 256)

# 4. 模型训练
def train():
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    dataset = LyricDataset(corpus_idx, 32)
    # 创建模型
    model = TextGenerator(word_count)
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练
    for epoch in range(100):
        # 定义 本轮开始训练时间，迭代次数，训练总损失
        start, batch_num, total_loss = time.time(), 0, 0.0
        # 迭代数据
        for x, y in train_loader:
            # 获取隐藏层
            hidden = model.init_hidden(5)
            # 获取输出值
            output, hidden = model(x, hidden)
            # 计算损失 y 的维度为 [batch_size, seq_len句子长度，词向量维度] - 转成1 维向量 - 每个词的下标索引
            # output 的维度为 [seq_len句子长度, batch_size, 词向量维度]
            y = torch.transpose(y, 0, 1).reshape(shape=(-1,))
            loss = criterion(output, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 累计损失
            total_loss += loss.item()
            batch_num += 1
        print(f'第 {epoch + 1} 轮训练结束，损失为: { total_loss/batch_num:.4f}， 训练时间: {time.time()-start:.2f}s')

    torch.save(model.state_dict(), './model/text_generator.pth')

# 5. 模型测试
def evaluate(start_text, num_words):
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    model = TextGenerator(word_count)
    model.load_state_dict(torch.load('./model/text_generator.pth'))
    hidden = model.init_hidden(1)
    word_index = word_to_index[start_text]
    # 定义列表，存放生成词的索引
    generated_words = [word_index]
    # 遍历句子长度
    for i in range(num_words):
        output, hidden = model(torch.tensor([[word_index]]), hidden)
        # 获取概率最大的索引
        word_index = torch.argmax(output)

        # 添加索引
        generated_words.append(word_index)

    # 获取生成的文本
    for i in generated_words:
        print(unique_words[i], end=' ')


# 6. 测试
if __name__ == '__main__':
    # unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    # print(f'词的数量：{word_count}')
    # print(f'语料中的词数: {len(unique_words)}')
    # print(f'构建词表成功，词表大小为: {len(word_to_index)}')
    # print(f'每个词的索引: {word_to_index}')
    # print(f'歌词文本用索引表示: {corpus_idx}')

    # 构建数据集对象
    # dataset = LyricDataset(corpus_idx, 5)
    #print(f'句子数量: {len( dataset)}')
    # 查看输入值 目标值
    # x, y = dataset[0]
    # print(f'输入值: {x}')
    # print(f'目标值: {y}')

    # 创建网络模型
    # model = TextGenerator(word_count)
    # for name, param in model.named_parameters():
    #    print(f'参数名称: {name}，参数维度: {param.shape}')
    # train()

    evaluate('分手', 50)

