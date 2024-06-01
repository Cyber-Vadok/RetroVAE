import numpy as np
import os 
import matplotlib.pyplot as plt

def load_tile_set():
    tile_set_path = 'gray_tile'
    tile_size = 16
    tile_set = []
    for filename in os.listdir(tile_set_path):
        tile = np.load(os.path.join(tile_set_path, filename))
        tile_set.append(tile)
    return tile_set

def convert_embedding_to_image(embedding, tile_set, tile_size):
    num_tiles_h, num_tiles_w, _ = embedding.shape
    # Initialize the image with the appropriate dimensions
    reconstructed_image = np.zeros((num_tiles_h * tile_size, num_tiles_w * tile_size), dtype=tile_set[0].dtype)
    
    # Loop through each position in the embedding array
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            # Find the index of the tile (one-hot encoding)
            tile_index = np.argmax(embedding[i, j])
            # Retrieve the tile from the tile set
            tile = tile_set[tile_index]
            # Place the tile in the correct position in the reconstructed image
            reconstructed_image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile

    return reconstructed_image

# Define a function to plot original and reconstructed images
def plot_images(original, reconstructed, tile_set, tile_size):
    num_images = min(original.size(0), 5)  # Plot up to 5 images
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 3*num_images))
    
    for i in range(num_images):
        # Plot original image
        img1 = convert_embedding_to_image(original[i].numpy(), tile_set, tile_size)
        axes[i, 0].imshow(img1.squeeze(), cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        img2 = convert_embedding_to_image(reconstructed[i].numpy(), tile_set, tile_size)
        axes[i, 1].imshow(img2.squeeze(), cmap='gray')
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
        