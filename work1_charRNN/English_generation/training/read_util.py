import pickle
import tensorflow as tf

id_char_path = '../data/shakespeare_id_char.txt'
int_text_path = '../data/shakespeare_int.txt'


def file_to_read(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f, encoding='bytes')


# filename：data_int数据地址
# batch_size：批量长度
# seq_length=100：时序长度
# buffer_size=10000：多少个打乱顺序
def data(filename=int_text_path, batch_size=64, seq_length=100, buffer_size=10000):
    text_as_int = file_to_read(filename)
    # 将获取的字符每101个合在一起
    char_data_set = tf.data.Dataset.from_tensor_slices(text_as_int).batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    # 将101字符集变为100输入100输出，每batch_size个字符集合为一个
    return char_data_set.map(split_input_target).shuffle(buffer_size).batch(batch_size, drop_remainder=True).repeat()


def id_char(filename=id_char_path):
    return file_to_read(filename)
