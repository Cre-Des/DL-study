import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import config

class InputMethodDataset(Dataset):
    """
    输入法数据集，用于加载 JSONL 文件并生成张量。
    """

    def __init__(self, file_path):
        """
        初始化数据集。
        param : file_path JSONL 文件路径
        """
        self.data = pd.read_json(file_path, lines=True).to_dict(orient='records')

    def __len__(self):
        """
        返回数据集的长度。
        return : 数据集的长度
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回指定索引处的数据项。
        param : idx 索引
        return : 数据项
        """
        input_tensor = torch.tensor(self.data[idx]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['target'], dtype=torch.long)
        return input_tensor, target_tensor

def get_dataloader(train = True):
    """
    生成数据加载器。
    param : train 是否加载训练集
    return : 数据加载器
    """
    file_name = 'train.jsonl' if train else 'test.jsonl'
    datasets = InputMethodDataset(config.PROCESSED_DATA_DIR / file_name)
    return DataLoader(datasets, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    dataloader = get_dataloader()
    for input_tensor, target_tensor in dataloader:
        print(input_tensor.shape, target_tensor.shape)
        break
