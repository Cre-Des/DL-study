import torch
import torch.nn as nn
from torchinfo import summary

import config

class TranslationEncoder(nn.Module):
    """
    编码器
    """

    def __init__(self, vocab_size, padding_idx):
        """
        初始化
        """
        super(TranslationEncoder, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx)

        # 双向GRU
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.ENCODER_HIDDEN_DIM,
            num_layers=config.ENCODER_LAYERS,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, src):
        """
        前向传播
        """
        embedded = self.embedding(src)
        output, hidden = self.gru(embedded)
        return output, hidden

class TranslationDecoder(nn.Module):
    """
    解码器
    """
    def __init__(self, vocab_size, padding_idx):
        """
        初始化
        """
        super(TranslationDecoder, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, config.EMBEDDING_DIM, padding_idx)

        # GRU
        self.gru = nn.GRU(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.DECODER_HIDDEN_DIM,
            batch_first=True
        )

        self.fc = nn.Linear(config.DECODER_HIDDEN_DIM, vocab_size)

    def forward(self, tgt, hidden):
        """
        前向传播
        :param tgt: 输入张量，形状 (batch_size, 1)。
        :param hidden: 隐藏状态张量，形状 (1, batch_size, hidden_dim)。
        :return: (输出张量, 新的隐藏状态)。
        """
        embedded = self.embedding(tgt)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output)
        return output, hidden

if __name__ == '__main__':
    encoder = TranslationEncoder(vocab_size=10000, padding_idx=0)
    dummy_encoder_input = torch.randint(0, 10000, size=(config.BATCH_SIZE, config.SEQ_LEN))
    summary(encoder, input_data=dummy_encoder_input)

    print('---------------------------------------------------------------------------------------------------------')

    decoder = TranslationDecoder(vocab_size=10000, padding_idx=0)
    dummy_decoder_input = torch.randint(0, 10000, size=(config.BATCH_SIZE, 1))
    dummy_decoder_hidden = torch.randn(1, config.BATCH_SIZE, config.DECODER_HIDDEN_DIM)
    summary(decoder, input_data=[dummy_decoder_input, dummy_decoder_hidden])