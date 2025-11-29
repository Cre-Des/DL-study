import torch
from tokenizer import Tokenizer
import config
from model import ReviewAnalyzeModel
from dataset import get_dataloader
from predict import predict_batch
from tqdm import tqdm

def evaluate(model, dataloader, device):
    """
    评估模型。

    param: model 模型
    param: dataloader 数据加载器
    param: device 设备
    """
    model.eval()
    total_count = 0
    total_correct = 0

    for input_tensor, target_tensor in tqdm(dataloader, desc='评估'):
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.tolist()
        predicted = predict_batch(input_tensor, model)

        for predicted, target in zip(predicted, target_tensor):
            pred_label = 1 if predicted > 0.5 else 0
            if pred_label == target:
                total_correct += 1
            total_count += 1
    return total_correct / total_count

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')
    test_loader = get_dataloader(train=False)
    tokenizer = Tokenizer.load_vocab(config.MODELS_DIR / 'vocab.txt')

    model = ReviewAnalyzeModel(
        vocab_size=tokenizer.vocab_size,
        padding_index=tokenizer.pad_idx).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))

    acc = evaluate(model, test_loader, device)

    print("========== 评估结果 ==========")
    print(f"准确率：{acc:.4f}")
    print("=============================")


