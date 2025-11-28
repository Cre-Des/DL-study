import torch
from torch import nn
from torchinfo import summary

import config

class RNNModel(nn.Module):
    """
    RNN 模型
    """
    def __init__(self, vocab_size):
        """
        初始化模型
        param : vocab_size 词表大小
        """
        super(RNNModel, self).__init__()
        # 词嵌入层 将 token 索引映射为稠密向量
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM)

        # RNN 层
        self.rnn = nn.RNN(
            input_size=config.EMBEDDING_DIM,  # 输入维度
            hidden_size=config.HIDDEN_SIZE,   # 隐藏状态维度
            batch_first=True
        )

        # 全连接层
        self.fc = nn.Linear(in_features=config.HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, x):
        """
        前向传播
        param : x 输入张量 形状 (batch_size, seq_len)。
        return : 输出张量  形状 (batch_size, vocab_size)
        """
        # 词嵌入
        embed = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        # RNN 层
        output, _ = self.rnn(embed) # (batch_size, seq_len, hidden_size)
        # 全连接层 取最后一个时间步的输出进行分类
        result = self.fc(output[:, -1, :]) # (batch_size, vocab_size)

        return result

if __name__ == '__main__':
    model = RNNModel(vocab_size=20000).to('cuda')
    # 创建随机 dummy 输入用于展示模型结构
    dummy_input = torch.randint(
        low=0,
        high=20000,
        size=(config.BATCH_SIZE, config.SEQ_LEN),
        dtype=torch.long,
        device='cuda'
    )
    # 打印模型摘要
    summary(model, input_data=dummy_input)

