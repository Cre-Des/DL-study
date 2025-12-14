import torch

from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationEncoder, TranslationDecoder
import config

def predict_one_batch(input_tensor, encoder, decoder, en_tokenizer, device):
    """
    预测一个 batch 的数据

    param: input_tensor 输入张量
    param: encoder 编码器
    param: decoder 解码器
    param: en_tokenizer 英文分词器
    param: device 设备
    """
    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        encoder_output, hidden = encoder(input_tensor)

        context_vector = torch.cat([hidden[-2], hidden[-1]], dim=1)

        batch_size = input_tensor.shape[0]
        decoder_input = torch.full((batch_size, 1), en_tokenizer.sos_token_idx, device =  device)
        decoder_hidden = context_vector.unsqueeze(0)

        generated = [[] for _ in range(batch_size)]
        finished = [False for _ in range(batch_size)]

        for step in range(1, config.SEQ_LEN):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            predict_indexes = decoder_output.argmax(dim=-1)
            for i in range(batch_size):
                if finished[i]:
                    continue
                token_id = predict_indexes[i].item()
                if token_id == en_tokenizer.eos_token_idx:
                    finished[i] = True
                    continue
                generated[i].append(token_id)

            if all(finished):
                 break

            decoder_input = predict_indexes

        return generated

def predict(zh_sentences, encoder, decoder, en_tokenizer, zh_tokenizer, device):
    """
    预测

    param: input_sentences 输入句子列表
    param: encoder 编码器
    param: decoder 解码器
    param: en_tokenizer 英文分词器
    param: zh_tokenizer 中文分词器
    param: device 设备
    """
    input_ids = zh_tokenizer.encode(zh_sentences, seq_len=config.SEQ_LEN, add_sos_eos=False)
    input_tensor = torch.tensor([input_ids], device=device)

    generated = predict_one_batch(input_tensor, encoder, decoder, en_tokenizer, device)
    en_indexes = generated[0]
    en_sentence = en_tokenizer.decode(en_indexes)

    return en_sentence

def run_predict():
    """
    启动交互程序
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    en_tokenizer = EnglishTokenizer.load_vocab(config.MODELS_PATH / 'en_vocab.txt')
    zh_tokenizer = ChineseTokenizer.load_vocab(config.MODELS_PATH / 'zh_vocab.txt')

    encoder = TranslationEncoder(vocab_size=zh_tokenizer.vocab_size, padding_idx=zh_tokenizer.pad_token_idx).to(device)
    decoder = TranslationDecoder(vocab_size=en_tokenizer.vocab_size, padding_idx=en_tokenizer.pad_token_idx).to(device)

    encoder.load_state_dict(torch.load(config.MODELS_PATH / 'encoder.pt'))
    decoder.load_state_dict(torch.load(config.MODELS_PATH / 'decoder.pt'))

    print('欢迎使用翻译系统，请输入中文句子：（输入 q 或 quit 退出）')
    while True:
        user_input = input('中文：')
        if user_input in ['q', 'quit']:
            print('谢谢使用，再见！')
            break
        if not user_input:
            print('请输入内容')
            continue

        result = predict(user_input, encoder, decoder, en_tokenizer, zh_tokenizer,  device)
        print(f'英文：{result}')

if __name__ == '__main__':
    run_predict()
