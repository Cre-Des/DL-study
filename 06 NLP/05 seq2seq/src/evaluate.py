import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import config
from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationEncoder, TranslationDecoder
from dataset import get_dataloaders
from predict import predict_one_batch

def evaluate(dataloader, encoder, decoder, en_tokenizer, zh_tokenizer, device):
    """
    评估模型

    param: dataloader 数据加载器
    param: encoder 编码器
    param: decoder 解码器
    param: en_tokenizer 英文分词器
    param: zh_tokenizer 中文分词器
    param: device 设备
    """
    all_references = []
    all_predictions = []

    special_tokens = [zh_tokenizer.sos_token, zh_tokenizer.eos_token, zh_tokenizer.pad_token]

    for src, tgt in tqdm(dataloader, desc='评估'):
        src = src.to(device)
        tgt = tgt.tolist()

        predictions = predict_one_batch(src, encoder, decoder, en_tokenizer, device)
        all_predictions.extend(predictions)

        for indexes in tgt:
            indexes = [index for index in indexes if index not in special_tokens]
            all_references.append([indexes])


    bleu_score = corpus_bleu(all_references, all_predictions)
    return bleu_score

def run_evaluate():
    """
    启动评估程序
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataloader = get_dataloaders(train=False)

    en_tokenizer = EnglishTokenizer.load_vocab(config.MODELS_PATH / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.load_vocab(config.MODELS_PATH / 'zh_vocab.txt')

    encoder = TranslationEncoder(vocab_size=zh_tokenizer.vocab_size, padding_idx=zh_tokenizer.pad_token_idx).to(device)
    decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size, padding_idx=en_tokenizer.pad_token_idx).to(device)

    encoder.load_state_dict(torch.load(config.MODELS_PATH / 'encoder.pt'))
    decoder.load_state_dict(torch.load(config.MODELS_PATH / 'decoder.pt'))

    bleu_score = evaluate(test_dataloader, encoder, decoder, en_tokenizer, zh_tokenizer, device)

    print('========== 评估结果 ==========')
    print(f'BLEU: {bleu_score:.2f}')
    print('=============================')


if __name__ == '__main__':
    run_evaluate()

