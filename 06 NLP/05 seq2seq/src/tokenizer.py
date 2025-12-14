from abc import abstractmethod
from nltk import word_tokenize, TreebankWordDetokenizer
from tqdm import tqdm


class BaseTokenizer:
    """
    分词器基类
    """
    unk_token = "<unk>"
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"

    def __init__(self, vocab_list):
        """
        初始化

        :param vocab_list: 词表列表。
        """
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2idx = {word: idx for idx, word in enumerate(vocab_list)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab_list)}
        self.unk_token_idx = self.word2idx[self.unk_token]
        self.sos_token_idx = self.word2idx[self.sos_token]
        self.eos_token_idx = self.word2idx[self.eos_token]
        self.pad_token_idx = self.word2idx[self.pad_token]

    @staticmethod
    @abstractmethod
    def tokenize(sentence):
        """
        分词
        """
        pass

    @abstractmethod
    def detokenize(self, indexes):
        """
        解码
        """
        pass

    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        """
        构建词典

        param sentences: 语料
        param vocab_file: 词典保存路径
        """
        unique_words = set()
        for sentence in tqdm(sentences, desc="分词"):
            for word in cls.tokenize(sentence):
                unique_words.add(word)

        vocab_list = [cls.pad_token, cls.sos_token, cls.eos_token, cls.unk_token] + list(unique_words)

        with open(vocab_file, "w", encoding="utf-8") as f:
            for word in vocab_list:
                f.write(word + "\n")

    @classmethod
    def load_vocab(cls, vocab_file):
        """
        加载词典

        param vocab_file: 词典保存路径
        """
        vocab_list = []
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f.readlines()]

        return cls(vocab_list)

    def encode(self, sentence,seq_len,add_sos_eos = False):
        """
        编码

        param sentence: 句子
        """
        tokens = self.tokenize(sentence)
        indexes = [self.word2idx.get(token, self.unk_token_idx) for token in tokens]

        if add_sos_eos:
            indexes = indexes[:seq_len-2]
            indexes = [self.sos_token_idx] + indexes + [self.eos_token_idx]
        else:
            indexes = indexes[:seq_len]

        if len(indexes) < seq_len:
            indexes = indexes + [self.pad_token_idx] * (seq_len - len(indexes))

        return indexes

class ChineseTokenizer(BaseTokenizer):
    """
    中文分词器
    """
    @staticmethod
    def tokenize(sentence):
        """
        分词
        """
        return list(sentence)

    def decode(self, indexes):
        """
        解码
        """
        return "".join([self.idx2word[idx] for idx in indexes])

class EnglishTokenizer(BaseTokenizer):
    """
    英文分词器
    """
    @staticmethod
    def tokenize(sentence):
        """
        分词
        """
        return word_tokenize(sentence)

    def decode(self, indexes):
        """
        解码
        """
        tokens = [self.idx2word[idx] for idx in indexes]
        return TreebankWordDetokenizer().detokenize(tokens)
