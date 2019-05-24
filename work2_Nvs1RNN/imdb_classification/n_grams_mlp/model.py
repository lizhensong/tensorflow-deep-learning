from tensorflow import keras
from work2_Nvs1RNN.imdb_classification.output_layer import _get_last_layer_units_and_activation


def mlp_model(layers, units, rate, input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    layer_add = keras.layers.Dropout(rate=rate)(inputs)
    for _ in range(layers - 1):
        layer_add = keras.layers.Dense(units=units, activation='relu')(layer_add)
        layer_add = keras.layers.Dropout(rate=rate)(layer_add)
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    outputs = keras.layers.Dense(units=op_units, activation=op_activation)(layer_add)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
