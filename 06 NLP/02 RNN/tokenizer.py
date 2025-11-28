"""

本模块负责分词、词表构建等功能。

"""

import jieba
from tqdm import tqdm

jieba.setLogLevel(jieba.logging.WARNING)

class Tokenizer:
    """
    分词器
    """

    unknown_token = '<unk>'

    def __init__(self,vocab_list):
        """
        初始化 tokenizer
        param : vocab_list 词表
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        # 构建词到索引的映射
        self.vocab2index = {word: index for index, word in enumerate(vocab_list)}
        self.index2vocab = {index: word for index, word in enumerate(vocab_list)}

        # 构建未知词的索引
        self.unknown_index = self.vocab2index[self.unknown_token]

    def encode(self, sentence):
        """
        对输入的文本进行编码
        param : sentence 输入的文本
        return : 编码后的文本
        """
        # 分词
        words = jieba.lcut(sentence)
        # 索引列表
        indices = [self.vocab2index.get(word, self.unknown_index) for word in words]
        return indices

    @classmethod
    def build_vocab(cls, sentences,vocab_file):
        """
        构建并保存词表
        param : sentences 输入的文本列表
        param : vocab_file 词表保存路径
        """
        unique_words = set() # Build an unordered collection of unique elements.
        for sentence in tqdm(sentences, desc='分词'):
            # 收集所有唯一词
            for word in jieba.lcut(sentence):
                unique_words.add(word)

        vocab_list = [cls.unknown_token] + list(unique_words)

        # 保存词表到文件
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')

    @classmethod
    def load_vocab(cls, vocab_file):
        """
        加载词表
        param : vocab_file 词表保存路径
        return : 词表列表
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.strip() for line in f.readlines()]
        return cls(vocab_list)