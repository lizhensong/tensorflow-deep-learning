import tensorflow as tf
from work1_charRNN.English_generation.training.model import build_model
from work1_charRNN.English_generation.training import read_util

id_char = read_util.id_char('../data/poetry_id_char.txt')
char_id = {u: i for i, u in enumerate(id_char)}
embedding_dim = 256
batch_size = 64

model = build_model(len(id_char), embedding_dim, batch_size=1)
model.load_weights(tf.train.latest_checkpoint('../checkpoints'))
model.build(tf.TensorShape([1, None]))
model.summary()


def generate_text(model_test, start_string):
    with open('../data/test.txt', 'w') as f:
        f.write(start_string)
        num_generate = 1000
        input_eval = [char_id[s] for s in start_string]
        # 添加了batch的维度
        input_eval = tf.expand_dims(input_eval, 0)
        model_test.reset_states()
        with tf.Session() as sess:
            for i in range(num_generate):
                predictions = model_test(input_eval)
                # 删除单个维度，将batch维度删除
                predictions = tf.squeeze(predictions, 0)
                predicted_id = tf.math.argmax(predictions, 1)
                sess.run(tf.global_variables_initializer())
                b = sess.run(predicted_id)
                input_eval = tf.expand_dims(b, 0)
                f.write(id_char[b[-1]])


generate_text(model, start_string=u"圆光低月殿，碎影乱风筠。")
