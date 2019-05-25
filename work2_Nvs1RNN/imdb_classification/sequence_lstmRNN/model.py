from tensorflow import keras
from work2_Nvs1RNN.imdb_classification.output_layer import _get_last_layer_units_and_activation


def lstmrnn_model(embedding_dim,  # int，嵌入向量的维度。
                  dropout_rate,  # float，Dropout图层下降的输入百分比。
                  input_shape,  # tuple，模型输入的形状。
                  num_classes,  # int，输出类的数量。
                  num_features,  # int，单词数（嵌入输入维度）。
                  use_pretrained_embedding=False,  # bool，如果启用了预先训练的嵌入，则为true。
                  is_embedding_trainable=False,  # bool，如果嵌入层是可训练的，则为true。
                  embedding_matrix=None):  # embedding_matrix：dict，具有嵌入系数的字典。

    inputs = keras.Input(shape=input_shape)
    if use_pretrained_embedding:
        layer_1 = keras.layers.Embedding(input_dim=num_features,
                                         output_dim=embedding_dim,
                                         weights=[embedding_matrix],
                                         trainable=is_embedding_trainable)(inputs)
    else:
        layer_1 = keras.layers.Embedding(input_dim=num_features,
                                         output_dim=embedding_dim)(inputs)
    layer_2 = keras.layers.BatchNormalization()(keras.layers.Dropout(rate=dropout_rate)(layer_1))
    layer_3 = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(layer_2)
    layer_4 = keras.layers.Bidirectional(keras.layers.LSTM(32))(layer_3)
    layer_5 = keras.layers.BatchNormalization()(keras.layers.Dropout(rate=dropout_rate)(layer_4))
    layer_6 = keras.layers.Dense(64, activation='relu')(layer_5)
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    outputs = keras.layers.Dense(units=op_units, activation=op_activation)(layer_6)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
