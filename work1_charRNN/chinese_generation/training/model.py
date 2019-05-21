from tensorflow import keras


def build_model(vocab_size, embedding_dim, batch_size):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(1024,
                          stateful=True,
                          return_sequences=True,  # 每个时间点都有返回
                          recurrent_initializer='glorot_uniform'),  # 参数使用均匀分布初始化器
        keras.layers.Dense(vocab_size, activation='softmax')
    ])
    return model
