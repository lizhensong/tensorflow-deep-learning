import tensorflow as tf
import numpy as np
from work1_charRNN.English_generation.training.model import build_model
from work1_charRNN.English_generation.training import read_util

id_char = read_util.id_char('../data/shakespeare_id_char.txt')
char_id = {u: i for i, u in enumerate(id_char)}
embedding_dim = 256
batch_size = 64

model = build_model(len(id_char), embedding_dim, batch_size=1)
model.load_weights(tf.train.latest_checkpoint('../checkpoints'))
model.build(tf.TensorShape([1, None]))
model.summary()


def generate_text(model_test, start_string):
    num_generate = 100
    input_eval = [char_id[s] for s in start_string]
    # 添加了batch的维度
    input_eval = tf.cast(tf.expand_dims(input_eval, 0), tf.int64)
    all_output = []
    model_test.reset_states()
    for i in range(num_generate):
        predictions = model_test(input_eval)
        predicted_id = tf.math.argmax(predictions, 2)
        if i == 0:
            # 删除单个维度，将batch维度删除
            predicted_id = tf.squeeze(predicted_id, 0)
            partition_id = np.zeros(predicted_id.shape.as_list())
            partition_id[-1] = 1
            input_id = tf.dynamic_partition(predicted_id, partition_id, 2)[1]
            input_eval = tf.expand_dims(input_id, 0)
        else:
            input_eval = predicted_id
        all_output.append(tf.squeeze(input_eval, 0))
        print(i)
    return all_output


start_string = u"COMINIUS:"
out_put = generate_text(model, start_string=start_string)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with open('../data/test.txt', 'w') as f:
        print('writing...')
        f.write(start_string)
        for char_tensor in out_put:
            char_id = sess.run(char_tensor)
            out_put_char = id_char[char_id]
            print(out_put_char)
            f.write(str(out_put_char))
