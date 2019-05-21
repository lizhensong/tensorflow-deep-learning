from tensorflow import keras


def build_model(batch_size):
    inputs = keras.Input(shape=(None, 1), batch_size=batch_size)
    layer_1 = keras.layers.Masking(mask_value=0)(inputs)
    layer_2 = keras.layers.Bidirectional(keras.layers.LSTM(64))(layer_1)
    layer_3 = keras.layers.Dense(64, activation='relu')(layer_2)
    layer_4 = keras.layers.Dense(1, activation='sigmoid')(layer_3)
    model = keras.Model(inputs=inputs, outputs=layer_4)
    return model
