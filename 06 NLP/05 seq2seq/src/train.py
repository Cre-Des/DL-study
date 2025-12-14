import time
from itertools import chain

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from tokenizer import ChineseTokenizer, EnglishTokenizer
import config
from model import TranslationEncoder, TranslationDecoder

def train_one_epoch(dataloader, encoder, decoder, optimizer, criterion, device):
    """
    训练一个epoch

    param: dataloader 数据加载器
    param: encoder 编码器
    param: decoder 解码器
    param: optimizer 优化器
    param: criterion 损失函数
    param: device 设备
    """
    encoder.train()
    decoder.train()
    total_loss = 0

    for src, tgt in tqdm(dataloader, desc='训练'):
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        _, encoder_hidden = encoder(src)

        forward_hidden = encoder_hidden[-2]
        backward_hidden = encoder_hidden[-1]
        context_vector = torch.cat([forward_hidden, backward_hidden], dim=1)

        decoder_input = tgt[:,0:1]
        decoder_hidden = context_vector.unsqueeze(0)

        decoder_outputs = []
        for step in range(1, config.SEQ_LEN):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            decoder_input = tgt[:,step:step+1]

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_targets = tgt[:,1:]

        loss = criterion(decoder_outputs.reshape(-1, decoder_outputs.shape[-1]), decoder_targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train():
    """
    训练
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataloader = get_dataloaders(train=True)

    en_tokenizer = EnglishTokenizer.load_vocab(config.MODELS_PATH / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.load_vocab(config.MODELS_PATH / 'zh_vocab.txt')

    encoder = TranslationEncoder(vocab_size=zh_tokenizer.vocab_size, padding_idx=zh_tokenizer.pad_token_idx).to(device)
    decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size, padding_idx=en_tokenizer.pad_token_idx).to(device)

    criterion = CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_idx)
    optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=config.LEARNING_RATE)

    writer = SummaryWriter(config.LOG_PATH / time.strftime('%Y-%m-%d_%H-%M-%S'))

    best_loss = float('inf')

    for epoch in range(config.EPOCHS):
        print(f'========== Epoch {epoch} ==========')
        avg_loss = train_one_epoch(train_dataloader, encoder, decoder, optimizer, criterion, device)

        print(f'Loss: {avg_loss:.4f}')
        writer.add_scalar('Loss/train', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), config.MODELS_PATH / 'encoder.pt')
            torch.save(decoder.state_dict(), config.MODELS_PATH / 'decoder.pt')
            print('保存模型')

if __name__ == '__main__':
    train()
