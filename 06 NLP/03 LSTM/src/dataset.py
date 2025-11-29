import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import config

class ReviewAnalyzeDataset(Dataset):
    """
    评论分析数据集
    """
    def __init__(self, file_path):
        """
        初始化数据集。

        Args:
            file_path (str): 数据文件路径。
        """
        self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

    def __len__(self):
        """
        获取数据集大小。

        Returns:
            int: 数据集大小。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据。

        Args:
            idx (int): 数据索引。

        Returns:
            input_tensor (torch.Tensor): 输入张量。
            target_tensor (torch.Tensor): 目标张量。
        """
        input_tensor = torch.tensor(self.data[idx]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['label'], dtype=torch.long)
        return input_tensor, target_tensor

def get_dataloader(train = True):
    """
    生成数据加载器。

    param : train 是否加载训练集
    return : 数据加载器
    """
    file_name = 'indexed_train.json' if train else 'indexed_test.json'
    dataset = ReviewAnalyzeDataset(config.PROCESSED_DATA_DIR / file_name)

    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    train_loader = get_dataloader()

    for input_tensor, target_tensor in train_loader:
        print(input_tensor.shape)
        print(target_tensor.shape)
        break