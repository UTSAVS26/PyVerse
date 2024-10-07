import random
import numpy
from PIL import Image

def derandomize(img, seed, original_shape):
    random.seed(seed)
    
    # Recreate the shuffled indices for both x and y
    new_y = list(range(original_shape[0]))
    new_x = list(range(original_shape[1]))
    random.shuffle(new_y)
    random.shuffle(new_x)

    # Create an empty array to store the reconstructed image
    reconstructed = numpy.empty_like(img)
    
    # Reverse the shuffling by mapping randomized positions back to the original positions
    for i, y in enumerate(new_y):
        for j, x in enumerate(new_x):
            reconstructed[y][x] = img[i][j]
    
    return numpy.array(reconstructed)

if __name__ == "__main__":
    # Open the scrambled (encrypted) image
    scrambled_img = Image.open("encrypted.png")
    scrambled_array = numpy.array(scrambled_img)
    
    original_shape = scrambled_array.shape
    
    # Define a range of potential seed values (assuming timestamp or a fixed range)
    start_seed = int(input("Enter the starting seed to brute-force: "))
    end_seed = int(input("Enter the ending seed to brute-force: "))

    for seed in range(start_seed, end_seed):
        reconstructed_img = derandomize(scrambled_array, seed, original_shape)
        
        # Save the reconstructed image
        image = Image.fromarray(reconstructed_img)
        image.save(f"decrypted_{seed}.png")
        
        # Optionally, display the reconstructed image to see if it's correct
        image.show()
        print(f"Tried seed: {seed}")

