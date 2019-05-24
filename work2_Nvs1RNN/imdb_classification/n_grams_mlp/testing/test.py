from tensorflow import keras
from work2_Nvs1RNN.imdb_classification.n_grams_mlp import data_use_read

test_data, test_labels = data_use_read.data_test()
new_model = keras.models.load_model('../training/IMDb_mlp_model.h5')
new_model.summary()
new_model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
loss, acc = new_model.evaluate(test_data, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
