# 将文本标记为单词
# 使用前20,000个令牌创建词汇表
# 将标记转换为序列向量
# 将序列填充到固定的序列长度
from tensorflow import keras
import pickle
from work2_Nvs1RNN.imdb_classification.data_analysis import load_imdb_sentiment_analysis_dataset

# 仅使用前2w个特征向量
TOP_K = 20000
# 限制句子的最大长度大于500将被截掉
MAX_SEQUENCE_LENGTH = 500


def sequence_vectorize(train_texts, val_texts, test_texts):
    # 需要保留的最大词数，基于词频
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=TOP_K)
    # 根据文本列表更新内部词汇表
    tokenizer.fit_on_texts(train_texts)
    # 将文本中的每个文本转换为整数序列（0为保留数据不会安排）
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)
    x_test = tokenizer.texts_to_sequences(test_texts)

    # max返回x_train中长度最长的，len再返回长度
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH
    # 长度大于max_length的截断，长度小于max_length的填充（默认是前端操作，填充为0）
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_length, padding='post', truncating='post')
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
    return x_train, x_val, x_test, (tokenizer.word_index, tokenizer.index_word)


def save_to_file(self, filename):
    with open(filename, 'wb') as f:
        pickle.dump(self, f)


a = load_imdb_sentiment_analysis_dataset('D:\Python_Work_Space\learning-data\movie-imbd-Initially')
a_train, a_val, a_test, token = sequence_vectorize(a[0][0], a[1][0], a[2][0])
save_to_file((a_train, a[0][1]), './data/train_data_one_hot')
save_to_file((a_val, a[1][1]), './data/val_data_one_hot')
save_to_file((a_test, a[2][1]), './data/test_data_one_hot')
save_to_file(token, './data/index_word_one_hot')
