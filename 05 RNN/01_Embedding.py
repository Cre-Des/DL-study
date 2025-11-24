"""
词嵌入层
"""

import torch
import torch.nn as nn
import jieba

# 0.文本数据
text = '北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。'

# 1.分词
words = jieba.lcut(text) # lcut() 分割中文
print(f'分词结果: {words}')

# 2.构建词嵌入层
# 词嵌入层参数: num_embeddings, embedding_dim
# num_embeddings: 词表大小
# embedding_dim: 词嵌入的维度
embedding = nn.Embedding(num_embeddings=len(words), embedding_dim=4)

# 3. 获取每个词对象的下标索引 enumerate()返回索引和值
for i, word in enumerate(words):
    # 把词索引(张量形式)转换为词向量
    word_vector = embedding(torch.tensor([i]))  # 随机的，每次不同
    print(f'{word} 的词向量: {word_vector}')