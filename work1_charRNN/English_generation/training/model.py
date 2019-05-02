from tensorflow import keras


def build_model(vocab_size, embedding_dim, batch_size):
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]),
        keras.layers.LSTM(1024,
                          return_sequences=True,  # 每个时间点都有返回
                          stateful=True,  # 批次中索引i处的每个样本的最后状态将用作后续批次中索引i的样本的初始状态
                          recurrent_initializer='glorot_uniform'),  # 参数使用均匀分布初始化器
        keras.layers.LSTM(516,
                          return_sequences=True,  # 每个时间点都有返回
                          stateful=True,  # 批次中索引i处的每个样本的最后状态将用作后续批次中索引i的样本的初始状态
                          recurrent_initializer='glorot_uniform'),  # 参数使用均匀分布初始化器
        keras.layers.LSTM(256,
                          return_sequences=True,  # 每个时间点都有返回
                          stateful=True,  # 批次中索引i处的每个样本的最后状态将用作后续批次中索引i的样本的初始状态
                          recurrent_initializer='glorot_uniform'),  # 参数使用均匀分布初始化器
        keras.layers.Dense(vocab_size, activation='softmax')
    ])
    return model
