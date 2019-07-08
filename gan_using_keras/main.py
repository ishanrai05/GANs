# Create a model using Generator and Degenerator
# Generator creates fake images
# Degenerator real images


import argparse

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input

from discriminator import build_discriminator
from generator import build_generator
from utils import img_rows, img_cols, channels, latent_dim, img_shape, save_images
from train import train

parser = argparse.ArgumentParser(description='GANs using keras')
parser.add_argument('--num_epochs', type=int, default=4000, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='train the model')

opt = parser.parse_args()

epochs = opt.num_epochs

if opt.train:

    print (f'Training for {epochs} with Adam optimizer and binary crossentropy loss on MNIST dataset.')
    print (f'Images will be saved in image folder')

    optimizer = Adam(0.0002, 0.5)

    # Build and compile the discriminator
    discriminator = build_discriminator(img_shape)
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    generator = build_generator(channels, latent_dim)

    # The generator will take noise as input and generate images
    z = Input(shape=(latent_dim,))
    img = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    valid = discriminator(img)

    # The combined model (attacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    train(epochs, latent_dim, generator, discriminator, combined, batch_size=32, save_interval=50)
