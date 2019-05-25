import pickle

train_path = '../data/train_data_one_hot'
val_path = '../data/val_data_one_hot'
test_path = '../data/test_data_one_hot'
one_hot_path = '../data/index_word_one_hot'


def file_to_read(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def data_train(path=train_path):
    return file_to_read(path)


def data_val(path=val_path):
    return file_to_read(path)


def data_test(path=test_path):
    return file_to_read(path)


def index_word(path=one_hot_path):
    return file_to_read(path)
