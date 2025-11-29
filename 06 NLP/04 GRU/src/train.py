import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from tokenizer import Tokenizer
from model import ReviewAnalyzeModel
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
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

        target_tensor = target_tensor.float()
        optimizer.zero_grad()
        # 前向传播
        output = model(input_tensor)
        # 计算损失
        loss = criterion(output, target_tensor)
        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train():
    """
    训练模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')
    train_loader = get_dataloader()
    tokenizer = Tokenizer.load_vocab(config.MODELS_DIR / 'vocab.txt')

    model = ReviewAnalyzeModel(
        vocab_size=tokenizer.vocab_size,
        padding_index=tokenizer.pad_idx).to(device)

    criterion  = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    writer = SummaryWriter(config.LOG_DIR/time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')

    for epoch in range(1,config.EPOCHS + 1):
        print(f'========== Epoch: {epoch} ==========')
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f'Loss: {train_loss:.4f}')
        writer.add_scalar('Loss/train', train_loss, epoch)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'model.pt')
            print('保存模型')

if __name__ == '__main__':
    train()

