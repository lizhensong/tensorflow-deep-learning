# 将文本标记为单词（1-gram和2-gram）
# 使用tf-idf编码进行向量化
# 丢弃只出现俩次的标记并使用f_classif计算标记的重要性，从标记中选择前2w个
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from work2_Nvs1RNN.imdb_classification.data_analysis import load_imdb_sentiment_analysis_dataset
import pickle

# 设定n-gram的大小
NGRAM_RANGE = (1, 2)
# 仅使用前2w个特征向量
TOP_K = 5000
# n-gram是字符级还是词级，俩个属性'word', 'char'.
TOKEN_MODE = 'word'
# 出现频率少于2的特征（标记）将被删除
MIN_DOCUMENT_FREQUENCY = 2


def ngram_vectorize(train_texts, train_labels, val_texts, test_texts):
    kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        # 'dtype': 'int32',
        # fit_transform（）或transform（）返回的矩阵的类型。
        'strip_accents': 'unicode',
        # 在预处理步骤中删除重音并执行其他字符规范化，‘ascii’（只适用ascii）, ‘unicode’（适用所有字符）, None（默认）
        'decode_error': 'replace',
        # 如果给定的字符集有编码不对的字符处理方法‘strict’(默认，保错), ‘ignore’, ‘replace’
        'analyzer': TOKEN_MODE,  # 按词分割
        'min_df': MIN_DOCUMENT_FREQUENCY,  # 小于2个被删
    }
    vectorizer = TfidfVectorizer(**kwargs)
    # 将传入文档变为tf-idf向量表达，（词汇存在vectorizer中）
    x_train = vectorizer.fit_transform(train_texts)
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    # 选择分数最高的k个特征，第一个参数默认为f_classif。
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    # 在（X，y）上运行得分函数并获得适当的特征。
    selector.fit(x_train, train_labels)
    # 将传入进行特征消减
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    x_test = selector.transform(x_test).astype('float32')

    return x_train, x_val, x_test


def save_to_file(self, filename):
    with open(filename, 'wb') as f:
        pickle.dump(self, f)


a = load_imdb_sentiment_analysis_dataset('D:\Python_Work_Space\learning-data\movie-imbd-Initially')
a_train, a_val, a_test = ngram_vectorize(a[0][0], a[0][1], a[1][0], a[2][0])
save_to_file((a_train, a[0][1]), './data/train_data_tf_idf')
save_to_file((a_val, a[1][1]), './data/val_data_tf_idf')
save_to_file((a_test, a[2][1]), './data/test_data_tf_idf')
