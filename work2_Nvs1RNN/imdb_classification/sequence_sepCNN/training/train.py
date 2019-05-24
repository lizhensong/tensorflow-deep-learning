from tensorflow import keras
from work2_Nvs1RNN.imdb_classification.sequence_sepCNN import data_use_read
from work2_Nvs1RNN.imdb_classification.sequence_sepCNN.model import sepcnn_model


def train_sepcnn_model(data,
                       num_features,
                       learning_rate=1e-3,
                       epochs=1000,
                       batch_size=128,
                       layers=2,
                       units=64,
                       rate=0.5):
    # 获取数据
    (train_texts, train_labels), (val_texts, val_labels) = data

    # 获取分类个数
    num_classes = max(train_labels) + 1

    # 创建模型
    model = sepcnn_model(blocks=layers, filters=units, kernel_size=3, embedding_dim=32, dropout_rate=rate, pool_size=2,
                         input_shape=train_texts.shape[1:], num_classes=num_classes, num_features=num_features)
    model.summary()
    # Compile model with learning parameters.
    if num_classes == 2:
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.sparse_categorical_crossentropy
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # monitor: 被监测的数据。
    # patience: 没有进步的训练轮数，在这之后训练就会被停止。
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]

    # Train and validate model.
    history = model.fit(
        train_texts,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(val_texts, val_labels),
        # verbose=2,  # Logs once per epoch.
        batch_size=batch_size
    )

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('./IMDb_mlp_model.h5')
    return history['val_acc'][-1], history['val_loss'][-1]


a = data_use_read.data_train()
b = data_use_read.data_val()
train_sepcnn_model((a, b), 20000)
