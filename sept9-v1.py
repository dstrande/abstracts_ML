import tensorflow as tf
from tensorflow.keras import optimizers
import numpy as np
import h5py
import multiprocessing
from tensorboard.plugins.hparams import api as hp
import gc


n = 2  # Use every n index to conserve memory. n=2 uses half the points.
with h5py.File('sept1/200k.hdf5', 'r') as f:  # e.g. sept1/200k
    X_train = np.array(f.get('X_train2')[::n])
    X_test = np.array(f.get('X_test2')[::n])
    y_train = np.array(f.get('y_train')[::n])
    y_test = np.array(f.get('y_test')[::n])

print(X_train.shape, y_train.shape)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=5, dtype='float32')
y_test = tf.keras.utils.to_categorical(y_test, num_classes=5, dtype='float32')


def modeling(params):

    optimize = [optimizers.Nadam, optimizers.Adam, optimizers.SGD,
                optimizers.Adadelta, optimizers.Adagrad, optimizers.Adamax]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(768,), name='input'),
        tf.keras.layers.Dense(768, activation=params['activation'], name='dense1'),
        tf.keras.layers.Dense(params['num_units'], activation=params['activation'], name='dense2'),
        tf.keras.layers.Dropout(params['dropout2']),
        tf.keras.layers.Dense(5, activation='softmax', name='output')
    ])

    opt = optimize[params['optimizer']](learning_rate=params['learning_rate'])  # 0.0005 was best so far.
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[es], batch_size=1024)
    _, accuracy = model.evaluate(X_test, y_test, verbose=2)

    tf.keras.backend.clear_session()
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return accuracy


def run(hparams, run_dir):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        print(hparams)
        accuracy = modeling(hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0009]))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['gelu']))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete([5]))
HP_DROPOUT2 = hp.HParam('dropout2', hp.Discrete([0.5]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([512]))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/sept9-v1').as_default():
    hp.hparams_config(
        hparams=[HP_LEARNING_RATE, HP_ACTIVATION, HP_OPTIMIZER, HP_DROPOUT2, HP_NUM_UNITS],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

session_num = 0
hparams_list = []
run_dirs = []

for learning_rate in HP_LEARNING_RATE.domain.values:
    for activation in HP_ACTIVATION.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            for dropout2 in HP_DROPOUT2.domain.values:
                for num_units in HP_NUM_UNITS.domain.values:
                    hparams = {
                        'learning_rate': learning_rate,
                        'activation': activation,
                        'optimizer': optimizer,
                        'dropout2': dropout2,
                        'num_units': num_units
                    }
                    run_dirs.append(f'logs/sept9-v1/run-{session_num:03}')
                    # print('--- Starting trial: %s' % run_name)
                    # print({h.name: hparams[h] for h in hparams})
                    hparams_list.append(hparams)
                    session_num += 1


if __name__ == "__main__":
    print(len(hparams_list))
    for i in range(len(hparams_list)):
        print('\n', i, '\n')
        process_eval = multiprocessing.Process(target=run, args=(hparams_list[i], run_dirs[i]))
        process_eval.start()
        process_eval.join()
