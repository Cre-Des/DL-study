import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.xpu import device
from tqdm import tqdm

from dataset import get_dataloader
from model import RNNModel
from tokenizer import Tokenizer
import config

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个epoch

    param: model 模型
    param: dataloader 数据加载器
    param: optimizer 优化器
    param: criterion 损失函数
    param: device 设备
    """
    total_loss = 0
    model.train()

    for input_tensor, target_tensor in tqdm(dataloader, desc='训练'):
        # 将数据移动到设备
        input_tensor , target_tensor = input_tensor.to(device), target_tensor.to(device)

        optimizer.zero_grad()

        # 前向传播
        output = model(input_tensor)

        # 计算损失
        loss = criterion(output, target_tensor)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def train():
    """
    训练模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')

    # 创建数据加载器
    train_dataloader = get_dataloader(train=True)

    # 创建模型
    tokenizer = Tokenizer.load_vocab(config.MODELS_DIR / 'vocab.txt')
    model = RNNModel(vocab_size=tokenizer.vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Tensorboard 日志
    writer = SummaryWriter(log_dir=config.LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1,config.EPOCHS + 1):
        print(f'========== Epoch: {epoch} ==========')
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        print(f'Loss: {avg_loss:.4f}')

        writer.add_scalar('Loss/train', avg_loss, epoch)

        # 保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print('保存模型')

if __name__ == '__main__':
    train()