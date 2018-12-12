import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.models import Sequential


def build_keras_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


def main():
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    num_classes = 10

    keras_model = build_keras_model(input_shape, num_classes)

    model_dir = 'experiments/'
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    save_checkpoints_steps = 1000
    max_checkpoint_without_decrease = 2
    config = tf.estimator.RunConfig(
            save_checkpoints_steps=save_checkpoints_steps,
            save_summary_steps=100,
            keep_checkpoint_max=max_checkpoint_without_decrease + 2
    )

    estimator = keras.estimator.model_to_estimator(
        keras_model,
        model_dir=model_dir,
        config=config
    )
    print(estimator.eval_dir())

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[:1000, :, :]
    y_train = y_train[:1000]
    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator,
        metric_name='loss',
        max_steps_without_decrease=save_checkpoints_steps * max_checkpoint_without_decrease,
        run_every_secs=None,
        run_every_steps=save_checkpoints_steps,
        min_steps=0)
    train_spec = tf.estimator.TrainSpec(
        tf.estimator.inputs.numpy_input_fn(x_train, y_train, shuffle=True, num_epochs=100, batch_size=10),
        hooks=[early_stopping]
    )
    eval_spec = tf.estimator.EvalSpec(
        tf.estimator.inputs.numpy_input_fn(x_test, y_test, shuffle=True, num_epochs=1),
        steps=save_checkpoints_steps,
        start_delay_secs=0,
        throttle_secs=0
    )

    tf.estimator.train_and_evaluate(
        estimator,
        train_spec,
        eval_spec
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    main()
