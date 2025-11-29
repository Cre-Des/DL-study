import pandas as pd
from sklearn.model_selection import train_test_split
import config
from tokenizer import Tokenizer

def process():
    """
    数据处理
    """
    df = pd.read_csv(
        config.RAW_DATA_DIR / 'online_shopping_10_cats.csv',
        encoding='utf-8',
        usecols=['review', 'label']
    )

    df = df.dropna()
    df = df[df['review'].str.strip().ne('')]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    Tokenizer.build_vocab(
        train_df['review'].tolist(),
        config.MODELS_DIR / 'vocab.txt'
        )

    tokenizer = Tokenizer.load_vocab(config.MODELS_DIR / 'vocab.txt')

    print('构建训练集')
    train_df['review'] = train_df['review'].apply(lambda x: tokenizer.encode(x, seq_len =config.SEQ_LEN))
    train_df.to_json(config.PROCESSED_DATA_DIR / 'indexed_train.json', orient='records', lines=True)

    print('构建测试集')
    test_df['review'] = test_df['review'].apply(lambda x: tokenizer.encode(x, seq_len =config.SEQ_LEN))
    test_df.to_json(config.PROCESSED_DATA_DIR / 'indexed_test.json', orient='records', lines=True)

    print('数据处理完成')

if __name__ == '__main__':
    process()