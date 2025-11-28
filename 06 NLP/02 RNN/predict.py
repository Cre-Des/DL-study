import torch
from torch.xpu import device

from model import RNNModel
from tokenizer import Tokenizer
import config

def predict_batch(input_tensor, model):
    """
    预测一个批次的数据
    param : input_tensor 输入张量
    param : model 模型
    return : 预测结果
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

        # 获取概率最大的5个索引
        predicted_indices = torch.topk(output, k=5, dim=-1).indices # (batch_size, 5)

    return predicted_indices.tolist()

def predict(text, model, tokenizer, device):
    """
    预测输入的文本
    param : text 输入的文本
    param : model 模型
    param : tokenizer 分词器
    param : device 设备
    return : 预测结果
    """
    # 编码输入
    input_tensor = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)

    # 预测
    predicted_indices = predict_batch(input_tensor, model)[0]

    return [tokenizer.index2vocab[index] for index in predicted_indices]

def run_predict():
    """
    启动交互程序
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer.load_vocab(config.MODELS_DIR / 'vocab.txt')

    model = RNNModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    print('请输入词语：（输入q或者quit退出系统）')
    text = ''
    while True:
        user_input = input('> ')
        if user_input == 'q' or user_input == 'quit':
            print('退出系统')
            break

        if not user_input:
            print('请输入词语：（输入q或者quit退出系统）')
            continue

        text += user_input
        print(f'历史输入：{text}')

        # 预测
        predictions = predict(text, model, tokenizer, device)
        print(f'预测结果：{predictions}')

if __name__ == '__main__':
    run_predict()
