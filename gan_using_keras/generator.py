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


def build_generator(channels, latent_dim):
    
    model = Sequential()
    
    # Dimension of output space = 128 * 7 * 7
    model.add(Dense(128 * 7 * 7, activation='relu', input_dim=latent_dim))

    # Reshapes an output to a 7 * 7 * 128
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D())


    model.add(Conv2D(128, kernel_size=3, padding='same'))

    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Activation('relu'))
    
    model.add(UpSampling2D())
    
    model.add(Conv2D(64, kernel_size=3, padding='same'))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation('relu'))

    model.add(Conv2D(channels, kernel_size=1, padding='same'))

    model.add(Activation('tanh'))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)
