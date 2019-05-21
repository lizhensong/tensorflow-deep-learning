import tensorflow as tf
from work2_Nvs1RNN.num_sequence_simulation.training.model import build_model
from work2_Nvs1RNN.num_sequence_simulation import get_num

model = build_model(batch_size=128)
model.load_weights(tf.train.latest_checkpoint('../checkpoints'))
model.build(tf.TensorShape([1, None]))
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])
train_data = get_num.get_train_num()
history = model.fit(train_data, steps_per_epoch=10000)
# a_1, a_2 = get_num.get_test_num()
# test_acc = model.evaluate(a_1, a_2, steps=1)
# print(test_acc)
