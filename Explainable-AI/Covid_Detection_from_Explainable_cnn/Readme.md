# COVID-19 Chest X-Ray Classification using Depthwise Separable Convolutional Neural Networks (CNN)

This project focuses on building a convolutional neural network (CNN) model to classify Chest X-Ray (CXR) images into three categories: **COVID-19**, **Viral Infection**, and **Normal**. The model uses image enhancement techniques like **White Balance** and **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for better image processing before classification.

## Dataset

The dataset used in this project consists of Chest X-Ray (CXR) images classified into three categories:
- **COVID-19**: Images of chest X-rays from COVID-19 patients.
- **Viral Infection**: Images of chest X-rays from patients with other viral infections.
- **Normal**: Images of chest X-rays from healthy individuals.

The dataset is stored in the following directories:
- `/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/covid/` for COVID-19 images.
- `/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/normal/` for normal images.
- `/kaggle/input/covid-cxr-image-dataset-research/COVID_IEEE/virus/` for viral infection images.

## Project Setup

1. **Dependencies**: The project requires several Python libraries such as `numpy`, `cv2`, `matplotlib`, `PIL`, `keras`, and `sklearn`. These can be installed via pip:
    ```bash
    pip install numpy opencv-python matplotlib Pillow scikit-learn tensorflow
    ```

2. **Image Enhancement**: The dataset images are enhanced using **White Balance** and **CLAHE** to improve the clarity of the Chest X-rays for better feature extraction.

3. **Model Architecture**: The model uses **Depthwise Separable Convolutional Neural Networks (CNN)**, which reduces the number of parameters while maintaining high performance for image classification tasks.

## Data Preprocessing

1. **White Balance**: Each image channel is processed by adjusting the intensity to standardize the color across images.
    ```python
    def wb(channel, perc=0.05):
        mi, ma = (np.percentile(channel, perc), np.percentile(channel, 100.0-perc))
        channel = np.uint8(np.clip((channel - mi) * 255.0 / (ma - mi), 0, 255))
        return channel
    ```

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: This technique is applied to enhance the contrast in images, especially useful in medical imaging.
    ```python
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img_clahe1 = clahe.apply(gray_image)
    ```

3. **Resizing**: All images are resized to a fixed size of `224x224` pixels to ensure uniformity when feeding them into the model.

4. **Normalization**: The pixel values of images are normalized by dividing by 255.0 to scale the values between 0 and 1.

## Model Architecture

The model architecture includes several convolutional layers, both traditional and depthwise separable, followed by dense layers for final classification:
- **Conv2D Layers**: Initial convolution layers to extract features.
- **MaxPooling Layers**: To reduce the spatial dimensions.
- **SeparableConv2D Layers**: Depthwise separable convolutions to reduce parameters.
- **BatchNormalization**: To normalize activations and improve training stability.
- **Dropout**: To prevent overfitting.
- **Dense Layers**: For classification of the images into three categories.

```python
inputs = Input(shape=(224, 224, 3))
x = Conv2D(filters=4, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = MaxPool2D(pool_size=(2, 2))(x)
...
output = Dense(units=3, activation='softmax')(x)
model = Model(inputs=inputs, outputs=output)
```

## Training and Evaluation

1. **Model Compilation**: The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function.
    ```python
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    ```

2. **Callbacks**: A `ModelCheckpoint` callback is used to save the best model based on validation loss during training.

3. **Train-Test Split**: The data is split into training and testing sets using `train_test_split` from `sklearn`.

4. **Data Augmentation**: An augmentation generator is used to artificially increase the size of the training data by applying random transformations.
    ```python
    trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
    ```

## Results and Observations

- **Normal CXR Images**: Clear lung patterns.
- **Viral Infection CXR Images**: Slight congestion in the lungs.
- **COVID-19 CXR Images**: Serious lung congestion.

## Conclusion

This project demonstrates how CNNs, particularly Depthwise Separable CNNs, can be used effectively to classify Chest X-ray images into categories such as COVID-19, Viral Infection, and Normal. The image enhancement techniques of White Balance and CLAHE significantly improve the quality of the input images, contributing to better model performance.

## Future Work

- Fine-tuning hyperparameters.
- Exploring more advanced CNN architectures.
- Integrating the model into a real-world application for automated CXR analysis.


