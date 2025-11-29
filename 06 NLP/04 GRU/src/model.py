import torch
import torch.nn as nn
import config
from torchinfo import summary

class ReviewAnalyzeModel(nn.Module):
    """
    评论情感分析模型，基于 LSTM。
    """
    def __init__(self, vocab_size, padding_index):
        super(ReviewAnalyzeModel, self).__init__()

        # 嵌入层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=config.EMBEDDING_DIM, padding_idx=padding_index)

        # LSTM
        self.lstm = nn.GRU(input_size=config.EMBEDDING_DIM, hidden_size=config.HIDDEN_DIM, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(config.HIDDEN_DIM, 1)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入数据。

        Returns:
            torch.Tensor: 输出结果。
        """
        x = self.embedding(x) # (batch_size, seq_len, embedding_dim)
        output, _ = self.lstm(x) # (batch_size, seq_len, hidden_dim)
        output = self.fc(output[:, -1, :]).squeeze(dim = 1) # (batch_size, 1)

        return output

if __name__ == '__main__':
    model = ReviewAnalyzeModel(vocab_size=1000, padding_index = 1)

    dummy_input = torch.randint(low=0, high=1000, size=(config.BATCH_SIZE, config.SEQ_LEN), dtype=torch.long)

    summary(model, input_data=dummy_input)