import numpy as np
import matplotlib.pyplot as plt

img_rows = 28
img_cols = 28
channels = 1
latent_dim = 100
img_shape = (img_rows, img_cols, channels)

def save_images(epochs, latent_dim, generator):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r*c, latent_dim))
    gen_imgs = generator.predict(noise)
    
    # Rescale Images 0-1
    gen_imgs = 0.5 * gen_imgs + 0.5
    
    fig, axs = plt.subplots(r,c)
    
    cnt = 0
    for i in range (r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt+=1
    fig.savefig(f'images/mnist_{epochs}.png')
    plt.close()