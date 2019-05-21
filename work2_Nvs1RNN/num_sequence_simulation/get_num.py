import random
import tensorflow as tf


def num(n_samples=10000, max_seq_len=20, min_seq_len=3, max_value=10000):
    data = []
    labels = []
    seq_len = []
    for i in range(n_samples):
        # 序列的长度是随机的，在min_seq_len和max_seq_len之间。
        # seq_len用于存储所有的序列。
        length = random.randint(min_seq_len, max_seq_len)
        seq_len.append(length)
        # 以50%的概率，随机添加一个线性或随机的训练
        if random.random() < .5:
            # 生成一个线性序列
            rand_start = random.randint(0, max_value - length)
            s = [[i] for i in range(rand_start, rand_start + length)]
            # 长度不足max_seq_len的需要补0
            s += [[0] for _ in range(max_seq_len - length)]
            data.append(s)
            # 线性序列的label是[1, 0]（因为我们一共只有两类）
            labels.append(1)
        else:
            # 生成一个随机序列
            s = [[random.randint(0, max_value)] for _ in range(length)]
            # 长度不足max_seq_len的需要补0
            s += [[0] for _ in range(max_seq_len - length)]
            data.append(s)
            labels.append(0)
    return data, labels


def get_train_num():
    train_data = num()
    return tf.data.Dataset.from_tensor_slices(train_data).shuffle(1000).batch(128, drop_remainder=True).repeat()


def get_test_num():
    test_data = num(100)
    return tf.constant(test_data[0]),tf.constant(test_data[1])
