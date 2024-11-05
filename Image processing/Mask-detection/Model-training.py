import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np

# Function to parse XML annotations
def parse_annotations(annotations_path):
    data = []

    for xml_file in os.listdir(annotations_path):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(annotations_path, xml_file))
            root = tree.getroot()
            image_file = root.find('filename').text  # Get image filename without path
            
            # For each object in the annotation
            objects = root.findall('object')
            for obj in objects:
                label = obj.find('name').text  # Get label
                # Map labels to binary (1 for with_mask, 0 for without_mask)
                data.append((image_file, 1 if label == 'with_mask' else 0))

    return data

# Paths
annotations_path = 'dataset/annotations'
images_path = 'dataset/images'

# Parse annotations to get image paths and labels
data = parse_annotations(annotations_path)

# Create a function to load images and their labels
def load_image_and_label(image_name):
    image_path = os.path.join(images_path, image_name)
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Image {image_name} not found.")
        return None, None

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to MobileNetV2 input size
    image = image / 255.0  # Normalize image
    return image

# Prepare the dataset
images = []
labels = []

for image_name, label in data:
    img = load_image_and_label(image_name)
    if img is not None:  # Only add valid images
        images.append(img)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Create a training-validation split
split_index = int(0.8 * len(images))
train_images = images[:split_index]
train_labels = labels[:split_index]
val_images = images[split_index:]
val_labels = labels[split_index:]

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator()

# Create generators
train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
val_generator = val_datagen.flow(val_images, val_labels, batch_size=32)

# Load MobileNetV2
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation="sigmoid")(x)  # Sigmoid for binary classification

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_images) // 32,
    validation_data=val_generator,
    validation_steps=len(val_images) // 32,
    epochs=epochs
)

# Save model
model.save("mask_detector_mobilenetv2.h5")
print("Model saved as mask_detector_mobilenetv2.h5")
