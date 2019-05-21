import tensorflow as tf
from work2_Nvs1RNN.num_sequence_simulation.training.model import build_model
from work2_Nvs1RNN.num_sequence_simulation import get_num

model = build_model(128)
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])
train_data = get_num.get_train_num()
history = model.fit(train_data, steps_per_epoch=10000)
model.save_weights('../checkpoints/my_checkpoint')
