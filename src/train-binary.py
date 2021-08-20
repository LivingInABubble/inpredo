import os
import sys

from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator


# If you like to speed up training process with GPU, first install PlaidML and then uncomment the following line.
# Otherwise it will fallback to tensorflow.
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

# BEFORE STARTING TRAINING YOU NEED TO MANUALLY TAKE 20 PERCENENT OF THE TRAINING DATA AND PUT IT INTO VALIDATION FOLDER
# I was too lazy to do it in the code.


def main():
    if len(sys.argv) > 1 and (sys.argv[1] == "--development" or sys.argv[1] == "-d"):
        epochs = 10
    else:
        epochs = 1000

    train_data_dir = '../data/train/'
    validation_data_dir = '../data/val/'

    # Input the size of your sample images
    img_width, img_height = 150, 150
    # Enter the number of samples, training + validation
    nb_train_samples, nb_validation_samples = 13204, 1412
    nb_filters1, nb_filters2, nb_filters3 = 32, 32, 64
    conv1_size, conv2_size, conv3_size = 3, 2, 5
    pool_size = 2
    # We have 2 classes, buy and sell
    classes_num = 2
    batch_size = 128
    # lr = 0.001
    # chanDim = 3

    model = Sequential()
    model.add(
        Conv2D(nb_filters1, (conv1_size, conv1_size), input_shape=(img_height, img_width, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Conv2D(nb_filters2, (conv2_size, conv2_size), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format='channels_first'))

    model.add(Conv2D(nb_filters3, (conv3_size, conv3_size), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format='channels_first'))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.rmsprop(),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        # rescale=1. / 255,
        horizontal_flip=False)

    test_datagen = ImageDataGenerator(
        # rescale=1. / 255,
        horizontal_flip=False)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        # shuffle=True,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        # shuffle=True,
        class_mode='categorical')

    """
    Tensorboard log
    """
    # target_dir = f"./models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    target_dir = './models/'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    checkpoint = ModelCheckpoint(target_dir, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        shuffle=True,
        validation_data=validation_generator,
        callbacks=callbacks_list,
        validation_steps=nb_validation_samples // batch_size)

    model.save('../src/models/model.h5')
    model.save_weights('../src/models/weights.h5')


if __name__ == '__main__':
    main()
