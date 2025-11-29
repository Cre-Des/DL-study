import torch
from tokenizer import Tokenizer
import config
from model import ReviewAnalyzeModel

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

        probs = torch.sigmoid(output)

    return probs.tolist()

def predict(user_input, model, tokenizer, device):
    """
    预测输入的文本
    param : user_input 输入的文本
    param : model 模型
    param : tokenizer 分词器
    return : 预测结果
    """
    input_idx = tokenizer.encode(user_input, seq_len=config.SEQ_LEN)
    input_tensor = torch.tensor([input_idx], dtype=torch.long).to(device)

    probs = predict_batch(input_tensor, model)
    prob = probs[0]

    return prob

def run_predict():
    """
    启动交互程序
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer.load_vocab(config.MODELS_DIR / 'vocab.txt')

    model = ReviewAnalyzeModel(
        vocab_size=tokenizer.vocab_size,
        padding_index=tokenizer.pad_idx).to(device)

    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    print('请输入要预测的评论：（输入 q 或 quit 退出）')

    while True:
        user_input = input('请输入要预测的文本：')
        if user_input == 'q' or user_input == 'quit':
            print('退出系统')
            break

        if not user_input:
            print('请输入要预测的文本')
            continue

        prob = predict(user_input, model, tokenizer, device)
        if prob > 0.5:
            print(f'预测结果：正面评论，概率为：{prob:.4f}')
        else:
            print(f'预测结果：负面评论，概率为：{1 - prob:.4f}')

if __name__ == '__main__':
    run_predict()
