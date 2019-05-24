import pickle

train_path = '../data/train_data_tf_idf'
val_path = '../data/val_data_tf_idf'
test_path = '../data/test_data_tf_idf'


def file_to_read(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def data_train(path=train_path):
    return file_to_read(path)


def data_val(path=val_path):
    return file_to_read(path)


def data_test(path=test_path):
    return file_to_read(path)
