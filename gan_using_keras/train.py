from keras.datasets import mnist

import numpy as np

from utils import save_images

def train(epochs, latent_dim, generator, discriminator, combined, batch_size=128, save_interval=50):
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    
    # Rescale -1 to 1
    
    X_train = X_train / 127.5 - 1
    X_train = np.expand_dims(X_train, axis=3)
    
    # Adversial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        
        # Train discriminator
        
        # Select random half of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        
        # Sample noise and generate batch of new images
        
        noise = np.random.normal(0,1,(batch_size,latent_dim))
        gen_imgs = generator.predict(noise)
        
        # Train the discriminitor
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss  = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        
        # Train generator
        g_loss = combined.train_on_batch(noise, valid)
        
        print (f"{epoch} [D loss: {d_loss[0]}, acc: {100*d_loss[1]} ] [G loss: {g_loss}]")
        
        if epoch % save_interval == 0:
            save_images(epoch, latent_dim, generator)