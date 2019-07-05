from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Reshape,
    UpSampling2D,
    Conv2D,
    BatchNormalization,
    Activation,
    Input,
    Dropout,
    ZeroPadding2D,
    Flatten
)
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

def build_discriminator(img_shape):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))

    model.add(ZeroPadding2D(padding=((0,1),(0,1))))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization(momentum=0.8))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)
