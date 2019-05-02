import tensorflow as tf
from work1_charRNN.English_generation.training.model import build_model
from work1_charRNN.English_generation.training import read_util

id_char = read_util.id_char('../data/poetry_id_char.txt')
embedding_dim = 256
batch_size = 64
model = build_model(
    vocab_size=len(id_char),
    embedding_dim=embedding_dim,
    batch_size=batch_size)
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy)
train_data = read_util.data('../data/shakespeare_int.txt', batch_size=batch_size)
history = model.fit(train_data, steps_per_epoch=1000)
model.save_weights('../checkpoints/my_checkpoint')
