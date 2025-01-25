import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load base model (ResNet50) and add custom layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Grad-CAM function
def get_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), predictions.numpy()

# Function to visualize Grad-CAM and additional insights
def apply_gradcam(image_path, model, target_size=(128, 128), opacity=0.4):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array_expanded)

    # Specify the last conv layer for ResNet50
    last_conv_layer_name = 'conv5_block3_out'
    heatmap, predictions = get_gradcam_heatmap(model, img_array_preprocessed, last_conv_layer_name)

    # Prediction probabilities
    class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']  # Replace with actual class names
    confidences = {class_names[i]: predictions[0][i] for i in range(len(class_names))}

    # Create visualizations
    plt.figure(figsize=(16, 6))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # Heatmap
    plt.subplot(1, 4, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    # Superimposed Image
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    heatmap_img = np.uint8(255 * heatmap)
    
    # Use jet colormap
    jet = plt.cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_img]
    
    # Resize and convert
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(target_size)
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose
    superimposed = jet_heatmap * opacity + img_array
    superimposed = tf.keras.preprocessing.image.array_to_img(superimposed)

    plt.subplot(1, 4, 3)
    plt.imshow(superimposed)
    plt.title('Superimposed Image')
    plt.axis('off')

    # Bar chart for confidence scores
    plt.subplot(1, 4, 4)
    plt.barh(list(confidences.keys()), list(confidences.values()), color='skyblue')
    plt.title('Prediction Confidence')
    plt.xlabel('Probability')
    plt.tight_layout()

    # Show and optionally save results
    plt.show()

    save_results(img, heatmap, superimposed)

# Function to save results
def save_results(original_img, heatmap, superimposed_img, output_dir="output/"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save images
    original_img.save(f"{output_dir}original_image.png")
    plt.imsave(f"{output_dir}heatmap.png", heatmap, cmap='jet')
    superimposed_img.save(f"{output_dir}superimposed_image.png")
    print(f"Results saved in {output_dir}")

jet = plt.colormaps.get_cmap('jet') 

# Example usage
if __name__ == "__main__":
    # Apply Grad-CAM
    image_path = "C:/Users/Medha Agarwal/Desktop/brainAI/brain-tumor-mri-dataset/Testing/glioma/Te-gl_0037.jpg"  # Update with the actual image path
    apply_gradcam(image_path=image_path, model=model, opacity=0.5)
