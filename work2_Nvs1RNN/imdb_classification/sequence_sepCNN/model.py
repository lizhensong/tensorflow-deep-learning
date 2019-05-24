# 四层stepCNN模型构建（序列模型分类最好）
from tensorflow import keras
from work2_Nvs1RNN.imdb_classification.output_layer import _get_last_layer_units_and_activation


def sepcnn_model(blocks,  # int sepCNN对和池块再模型中的数量
                 filters,  # int 输出通道数（深度可分离卷积网络第二个卷积核的数量）
                 kernel_size,  # int 卷积窗口的长度。（深度可分离卷积网络第一个卷积核的边长）
                 embedding_dim,  # int，嵌入向量的维度。
                 dropout_rate,  # float，Dropout图层下降的输入百分比。
                 pool_size,  # int，在MaxPooling层下​​缩放输入的因子（多少数量求个最大值）。
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
    layer_add = keras.layers.BatchNormalization()(keras.layers.Dropout(rate=dropout_rate)(layer_1))
    for _ in range(blocks - 1):
        layer_add_1 = keras.layers.SeparableConv1D(filters=filters,
                                                   kernel_size=kernel_size,
                                                   activation='relu',
                                                   bias_initializer='random_uniform',
                                                   depthwise_initializer='random_uniform',  # 深度卷积核的初始化
                                                   padding='same')(layer_add)
        layer_add_2 = keras.layers.SeparableConv1D(filters=filters,
                                                   kernel_size=kernel_size,
                                                   activation='relu',
                                                   bias_initializer='random_uniform',
                                                   depthwise_initializer='random_uniform',
                                                   padding='same')(layer_add_1)
        layer_add_3 = keras.layers.MaxPooling1D(pool_size=pool_size)(layer_add_2)
        layer_add = keras.layers.BatchNormalization()(keras.layers.Dropout(rate=dropout_rate)(layer_add_3))
    layer_3 = keras.layers.SeparableConv1D(filters=filters * 2,
                                           kernel_size=kernel_size,
                                           activation='relu',
                                           # bias_initializer='random_uniform',
                                           # depthwise_initializer='random_uniform',
                                           padding='same')(layer_add)
    layer_4 = keras.layers.SeparableConv1D(filters=filters * 2,
                                           kernel_size=kernel_size,
                                           activation='relu',
                                           bias_initializer='random_uniform',
                                           depthwise_initializer='random_uniform',
                                           padding='same')(layer_3)
    layer_5 = keras.layers.GlobalAveragePooling1D()(layer_4)
    # 对于时序数据的全局平均池化。
    # 如果 data_format='channels_last'，输入为 3D 张量，尺寸为：(batch_size, steps, features)
    # 如果data_format='channels_first'，输入为 3D 张量，尺寸为：(batch_size, features, steps)
    # 输出：尺寸是 (batch_size, features) 的 2D 张量。
    layer_6 = keras.layers.BatchNormalization()(keras.layers.Dropout(rate=dropout_rate)(layer_5))
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    outputs = keras.layers.Dense(units=op_units, activation=op_activation)(layer_6)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
