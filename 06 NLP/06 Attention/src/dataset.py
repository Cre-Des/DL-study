import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config

class TranslationDataset(Dataset):
    """
    翻译数据集
    """
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, lines=True).to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['en'], dtype=torch.long)
        return input_tensor, target_tensor

def get_dataloaders(train = True):
    """
    获取数据加载器
    """
    data_path = config.PROCESSED_DATA_PATH / 'indexed_train.jsonl' if train else config.PROCESSED_DATA_PATH / 'indexed_test.jsonl'
    dataset = TranslationDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    return dataloader

if __name__ == '__main__':
    train_dataloader = get_dataloaders(train=True)
    for input, target in train_dataloader:
        print(input.shape, target.shape)
        break