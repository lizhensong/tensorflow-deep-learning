import numpy as np
import pickle

raw_path = './data/shakespeare.txt'
id_char_path = './data/shakespeare_id_char.txt'
int_text_path = './data/shakespeare_int.txt'


def save_to_file(self, filename):
    with open(filename, 'wb') as f:
        pickle.dump(self, f)


def file_op(filename=raw_path):
    with open(filename, 'r') as rf:
        text = rf.read()
    vocab = sorted(set(text))  # 获取所有字符
    idx2char = np.array(vocab)  # array类型可以下标连取
    save_to_file(idx2char, id_char_path)
    char2idx = {u: i for i, u in enumerate(vocab)}
    text_as_int = np.array([char2idx[c] for c in text])
    save_to_file(text_as_int, int_text_path)


if __name__ == '__main__':
    file_op()
