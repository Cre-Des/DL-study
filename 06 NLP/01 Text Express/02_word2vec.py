import jieba
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pandas as pd



# 1 使用公开词向量
# 加载词向量模型
def PublicWord2Vec():
    model = KeyedVectors.load_word2vec_format('./data/sgns.weibo.word')
    print(model.vector_size)

    print(model['中国'])

    similarity = model.similarity('中国', '美国')
    print(similarity)

    similar_words = model.most_similar('中国', topn=10)
    print(similar_words)

# 2 自行训练词向量
def SelfTrainWord2Vec():
    df = pd.read_csv('./data/online_shopping_10_cats.csv', encoding='utf-8',usecols=['review'])
    df = df.dropna()

    # Word2Vec的训练语料需要是已分词的文本序列
    sentences = [[token for token in jieba.lcut(review) if token.strip() != ''] for review in df['review']]
    print( sentences)
    model = Word2Vec(
        sentences,  # 训练语料
        vector_size=100, # 词向量维度
        window=5, # 上下文窗口大小
        min_count=2, # 最小词频（低于将被忽略）
        workers=4,  # 训练线程数
        sg=1 # 训练算法，0为CBOW，1为skip-gram
    )

    model.wv.save_word2vec_format('./model/my_vec.kv')

    my_model = KeyedVectors.load_word2vec_format('./model/my_vec.kv')
    print(my_model)
    print(my_model.vector_size)

# 3 测试
def Test():
    word_vectors = KeyedVectors.load_word2vec_format('./model/my_vec.kv')

    word2index = word_vectors.key_to_index
    embedding_dim = word_vectors.vector_size
    num_embeddings = len(word2index)

    embedding_matrix = torch.zeros(num_embeddings, embedding_dim)
    for word, index in word2index.items():
        embedding_matrix[index] = torch.from_numpy(word_vectors[word])

    embedding_layer = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

    input_words = ["我", "喜欢", "乘坐", "地铁"]
    input_indices = [word2index[word] for word in input_words]
    input_tensor = torch.tensor([input_indices])
    output = embedding_layer(input_tensor)

    print(output)


if __name__ == '__main__':
    # PublicWord2Vec()
    # SelfTrainWord2Vec()
    Test()