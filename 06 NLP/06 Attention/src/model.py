import torch
import torch.nn as nn
from torchinfo import summary

import config

class Attention(nn.Module):
    """
    注意力机制：计算当前 decoder 状态与 encoder 输出的注意力上下文向量。

        :param decoder_hidden: 当前时间步解码器的隐藏状态 (1, batch_size, decoder_hidden_dim)
        :param encoder_outputs: 编码器所有时间步输出 (batch_size, seq_len, decoder_hidden_dim)
        :return: 上下文向量 (batch_size, 1, decoder_hidden_dim)

    """
    def forward(self,decoder_hidden, encoder_outputs):
        attention_scores = torch.bmm(decoder_hidden.transpose(0,1), encoder_outputs.transpose(1,2))
        attention_weights = nn.functional.softmax(attention_scores, dim=2)

        context_vector = torch.bmm(attention_weights, encoder_outputs)
        return context_vector


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
    解码器：单向 GRU + Attention，逐步生成英文翻译。
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

        self.fc = nn.Linear(2 * config.DECODER_HIDDEN_DIM, vocab_size)
        self.attention = Attention()

    def forward(self, tgt, hidden, encoder_outputs):
        """
        前向传播
        :param tgt: 输入张量，形状 (batch_size, 1)。
        :param hidden: 隐藏状态张量，形状 (1, batch_size, hidden_dim)。
        :param encoder_outputs: 编码器输出张量，形状 (batch_size, seq_len, hidden_dim)。
        :return: (输出张量, 新的隐藏状态)。
        """
        embedded = self.embedding(tgt)
        output, hidden = self.gru(embedded, hidden)
        context_vector = self.attention(hidden, encoder_outputs)
        combined = torch.cat([output, context_vector], dim=2)
        output = self.fc(combined)
        return output, hidden

if __name__ == '__main__':
    encoder = TranslationEncoder(vocab_size=10000, padding_idx=0)
    dummy_encoder_input = torch.randint(0, 10000, size=(config.BATCH_SIZE, config.SEQ_LEN))
    summary(encoder, input_data=dummy_encoder_input)

    print('---------------------------------------------------------------------------------------------------------')

    decoder = TranslationDecoder(vocab_size=10000, padding_idx=0)
    dummy_decoder_input = torch.randint(0, 10000, size=(config.BATCH_SIZE, 1))
    dummy_decoder_hidden = torch.randn(size = (1, config.BATCH_SIZE, config.DECODER_HIDDEN_DIM))
    dummy_encoder_outputs = torch.randn(size=(config.BATCH_SIZE, config.SEQ_LEN, config.DECODER_HIDDEN_DIM))
    summary(decoder,  input_data=[dummy_decoder_input, dummy_decoder_hidden, dummy_encoder_outputs])