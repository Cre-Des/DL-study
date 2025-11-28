from multiprocessing.pool import worker

import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
import config
from tqdm import tqdm

def build_dataset(sentences, word2index):
    indexed_sentences =[[word2index.get(token, word2index['<unk>']) for token in jieba.lcut(sentence)] for sentence in sentences]

    dataset = []
    for sentence  in tqdm(indexed_sentences, desc='构建数据集'):
        for i in range(len(sentence) - config.SEQ_LEN ):
            input_tensor = sentence[i:i+config.SEQ_LEN]
            target_tensor = sentence[i+config.SEQ_LEN]
            dataset.append({'input': input_tensor, 'target': target_tensor})
    return dataset


def process_data():
    print('开始处理数据')
    # 读取数据
    df = pd.read_json(config.RAW_DATA_DIR / 'synthesized_.jsonl',orient='records', lines=True)

    # 获取所有句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])
    print(f'句子总数：{len(sentences)}')

    # 划分数据集
    train_sentences, test_sentences =train_test_split(sentences, test_size=0.2)

    # 构建词表
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc='构建词表'):
        vocab_set.update(jieba.lcut(sentence))

    vocab_list = ['<unk>'] + list(vocab_set)
    print(f'词表大小：{len(vocab_list)}')

    # 保存词表
    with open(config.MODELS_DIR/'vocab.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab_list))

    # 构建训练集
    word2index =  {word: index for index, word in enumerate(vocab_list)}
    train_dataset = build_dataset(train_sentences, word2index)

    # 保存训练集
    pd.DataFrame(train_dataset).to_json(config.PROCESSED_DATA_DIR/'train.jsonl', orient='records', lines=True)

    # 构建测试集
    text_dataset = build_dataset(test_sentences, word2index)

    # 保存测试集
    pd.DataFrame(text_dataset).to_json(config.PROCESSED_DATA_DIR/'test.jsonl', orient='records', lines=True)

    print('数据处理完成')


if __name__ == '__main__':
    process_data()
