import torch
from tqdm import tqdm

from tokenizer import Tokenizer
import config
from model import RNNModel
from dataset import get_dataloader
from predict import predict_batch

def evaluate(model, dataloader, device):
    """
    评估模型

    param: model 模型
    param: dataloader 数据加载器
    param: device 设备
    return: 评估结果
    """
    total_count = 0
    top1_correct = 0
    top5_correct = 0

    model.eval()
    for input_tensor, target_tensor in tqdm(dataloader, desc='评估'):
        input_tensor , target = input_tensor.to(device), target_tensor.tolist()

        # 获取 top-5 预测结果
        predicted_ids = predict_batch(input_tensor, model)

        # 计算准确率
        for predicted_ids, target_id in zip(predicted_ids, target):
            if predicted_ids[0] == target:
                top1_correct += 1
            if target_id in predicted_ids:
                top5_correct += 1
            total_count += 1

    top1_acc = top1_correct / total_count
    top5_acc = top5_correct / total_count
    return top1_acc, top5_acc

def run_evaluate():
    """
    启动评估程序
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer.load_vocab(config.MODELS_DIR / 'vocab.txt')

    model = RNNModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    test_dataloader = get_dataloader(train=False)

    top1_acc, top5_acc = evaluate(model, test_dataloader, device)

    # 输出评估结果
    print("======= 评估结果 =======")
    print(f"Top-1 准确率: {top1_acc:.4f}")
    print(f"Top-5 准确率: {top5_acc:.4f}")
    print("========================")

if __name__ == '__main__':
    run_evaluate()

