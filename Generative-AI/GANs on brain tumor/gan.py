import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set GPU ID to 0

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def load_and_preprocess_images(folder_path, image_size=(128, 128)):
    """
    Loads images from subfolders of folder_path, resizes them, converts to grayscale,
    and normalizes them to the range [-1, 1].
    
    folder_path: path to the dataset folder, which contains subfolders for each class.
    image_size: desired size of the output images (default is 128x128).
    """
    images = []
    labels = []
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Folder names as class labels
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.exists(class_folder):
            print(f"Warning: folder {class_folder} does not exist!")
            continue

        for filename in os.listdir(class_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_folder, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(image_size)
                img_array = np.array(img)
                images.append(img_array)
                labels.append(class_mapping[class_name])  # Add the corresponding class label

    images = np.array(images)
    labels = np.array(labels)

    # Normalize images to the range [-1, 1]
    images = (images - 127.5) / 127.5

    # Return images reshaped and corresponding labels
    return images.reshape(images.shape + (1,)), labels

# Replace with your actual training folder path
train_folder = 'brain-tumor-mri-dataset/Training'

# Load and preprocess the training images and labels
x_train, y_train = load_and_preprocess_images(train_folder)

print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

# Define the Generator model
def build_generator():
    model = keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),  # Start with 8x8 feature maps
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),  # Reshape to (8, 8, 256)

        # Upsample to 16x16
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upsample to 32x32
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upsample to 64x64
        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Upsample to 128x128
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    
    return model

# Define the Discriminator model
def build_discriminator():
    model = keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

# Define loss functions and optimizers
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50

def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Create a progress bar for each batch in the dataset
        progress_bar = tqdm(dataset, desc=f"Training Epoch {epoch+1}/{epochs}", unit="batch")
        
        for image_batch in progress_bar:
            gen_loss, disc_loss = train_step(image_batch)
            
            # Update progress bar description with loss information
            progress_bar.set_postfix({"Gen Loss": gen_loss.numpy(), "Disc Loss": disc_loss.numpy()})

        print(f"Epoch {epoch+1} completed. Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}\n")
        
# Prepare the dataset
buffer_size = max(1000, x_train.shape[0])  # Avoid zero buffer size
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(BATCH_SIZE)

# Train the GAN
train(train_dataset, EPOCHS)

# Generate new images using the trained generator
def generate_images(num_images):
    noise = tf.random.normal([num_images, 100])
    generated_images = generator(noise, training=False)
    generated_images = (generated_images * 127.5 + 127.5).numpy().astype(np.uint8)
    return generated_images

# Generate as many new images as in the original training set
num_augmented = x_train.shape[0]
augmented_images = generate_images(num_augmented)

print(f"Generated {num_augmented} new images")

# Denormalize original images
original_images = (x_train * 127.5 + 127.5).astype(np.uint8)

# Combine original and generated images
augmented_dataset = np.concatenate([original_images, augmented_images], axis=0)

print(f"Original dataset size: {len(original_images)}")
print(f"Augmented dataset size: {len(augmented_dataset)}")

# Visualize some augmented images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(augmented_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()

# Save the augmented images
output_folder = 'augmented_mri_images'
os.makedirs(output_folder, exist_ok=True)

for i, img in enumerate(augmented_dataset):
    img_pil = Image.fromarray(img[:, :, 0])
    img_pil.save(os.path.join(output_folder, f'augmented_mri_{i}.png'))

print(f"Augmented images saved in folder: {output_folder}")