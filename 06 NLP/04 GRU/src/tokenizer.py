import jieba
from tqdm import tqdm

jieba.setLogLevel(jieba.logging.WARNING)

class Tokenizer:
    """
    基于 jieba 的分词器，用于分词、编码和词表管理。
    """
    unk_token = '<unk>'
    pad_token = '<pad>'

    def __init__(self, vocab_list):
        """
        初始化分词器。

        Args:
            vocab_file (str): 词表文件路径。
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(self.vocab_list)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab_list)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab_list)}
        self.unk_idx = self.word2idx[self.unk_token]
        self.pad_idx = self.word2idx[self.pad_token]

    @staticmethod
    def tokenize(sentence):
        """
        分词。

        Args:
            sentence (str): 输入句子。

        Returns:
            list: 分词结果。
        """
        return jieba.lcut(sentence)

    @classmethod
    def build_vocab(cls, sentences,vocab_file):
        """
        构建词表。

        Args:
            sentences (list): 输入句子列表。
            vocab_file (str): 词表文件路径。
        """
        unique_vocab = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            # 收集所有唯一词
            for word in cls.tokenize(sentence):
                unique_vocab.add(word)

        vocab = [cls.unk_token, cls.pad_token] + list(unique_vocab)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write(word + '\n')

    @classmethod
    def load_vocab(cls, vocab_file):
        """
        加载词表。

        Args:
            vocab_file (str): 词表文件路径。

        Returns:
            list: 词表列表。
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f.readlines()]
        return cls(vocab)

    def encode(self, sentence, seq_len):
        """
        编码句子。

        Args:
            sentence (str): 输入句子。

        Returns:
            list: 编码结果。
        """
        tokens = self.tokenize(sentence)
        indexes = [self.word2idx.get(token, self.unk_idx) for token in tokens]

        # 填充或截断
        if len(indexes) >= seq_len:
            return indexes[:seq_len]
        else:
            return indexes + [self.pad_idx] * (seq_len - len(indexes))
