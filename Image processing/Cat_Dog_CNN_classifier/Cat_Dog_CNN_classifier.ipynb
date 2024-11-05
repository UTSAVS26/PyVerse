{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "6282f39b",
   "metadata": {
    "papermill": {
     "duration": 0.02303,
     "end_time": "2021-12-09T16:40:52.177750",
     "exception": false,
     "start_time": "2021-12-09T16:40:52.154720",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **Introduction** \n",
    "\n",
    "![](https://miro.medium.com/max/1400/1*oB3S5yHHhvougJkPXuc8og.gif)\n",
    "\n",
    "## **What is Image classification?**\n",
    "**Image classification is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rules. The categorization law can be devised using one or more spectral or textural characteristics.**\n",
    "\n",
    "##  **Different image classification techniques:-**\n",
    "![](https://raw.githubusercontent.com/everydaycodings/files-for-multiplethings/master/1.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e766d1d7",
   "metadata": {
    "papermill": {
     "duration": 0.021852,
     "end_time": "2021-12-09T16:40:52.223259",
     "exception": false,
     "start_time": "2021-12-09T16:40:52.201407",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Importing libraries**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "5005caf9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:40:52.277989Z",
     "iopub.status.busy": "2021-12-09T16:40:52.277265Z",
     "iopub.status.idle": "2021-12-09T16:40:56.308366Z",
     "shell.execute_reply": "2021-12-09T16:40:56.307349Z",
     "shell.execute_reply.started": "2021-12-09T16:12:49.375038Z"
    },
    "papermill": {
     "duration": 4.063465,
     "end_time": "2021-12-09T16:40:56.308532",
     "exception": false,
     "start_time": "2021-12-09T16:40:52.245067",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Importing the basic libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import layers, models\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a09a22b6",
   "metadata": {
    "papermill": {
     "duration": 0.02381,
     "end_time": "2021-12-09T16:40:58.505057",
     "exception": false,
     "start_time": "2021-12-09T16:40:58.481247",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Preprocessing the training Data using ImageDataGenerator**\n",
    "**One of the methods to prevent overfitting is to have more data. By this, our model will be exposed to more aspects of data and thus will generalize better. To get more data, either you manually collect data or generate data from the existing data by applying some transformations. The latter method is known as Data Augmentation.**\n",
    "\n",
    " - **rescale:** rescaling factor. If None or 0, no rescaling is applied, otherwise we multiply the data by the value provided.\n",
    " - **shear_range:** This is the shear angle in the counter-clockwise direction in degrees.\n",
    " - **zoom_range:** This zooms the image.\n",
    " - **horizontal_flip:** Randomly flips the input image in the horizontal direction.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7b25bb3c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:40:58.555137Z",
     "iopub.status.busy": "2021-12-09T16:40:58.554344Z",
     "iopub.status.idle": "2021-12-09T16:41:01.112941Z",
     "shell.execute_reply": "2021-12-09T16:41:01.113516Z",
     "shell.execute_reply.started": "2021-12-09T16:12:49.401944Z"
    },
    "papermill": {
     "duration": 2.586195,
     "end_time": "2021-12-09T16:41:01.113744",
     "exception": false,
     "start_time": "2021-12-09T16:40:58.527549",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 8005 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)\n",
    "\n",
    "training_set = datagen.flow_from_directory(\n",
    "        \"./archive/training_set/training_set/\",\n",
    "        target_size=(64, 64),\n",
    "        batch_size=32,\n",
    "        class_mode=\"binary\"\n",
    "      )\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "178346cd",
   "metadata": {
    "papermill": {
     "duration": 0.034913,
     "end_time": "2021-12-09T16:41:01.184943",
     "exception": false,
     "start_time": "2021-12-09T16:41:01.150030",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### **Preprocessing the test Data using ImageDataGenerator**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "33356abb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:01.261890Z",
     "iopub.status.busy": "2021-12-09T16:41:01.261094Z",
     "iopub.status.idle": "2021-12-09T16:41:01.488030Z",
     "shell.execute_reply": "2021-12-09T16:41:01.488803Z",
     "shell.execute_reply.started": "2021-12-09T16:12:50.933152Z"
    },
    "papermill": {
     "duration": 0.269435,
     "end_time": "2021-12-09T16:41:01.489070",
     "exception": false,
     "start_time": "2021-12-09T16:41:01.219635",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 2023 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "datagen1 = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "test_set = datagen1.flow_from_directory(\n",
    "        \"./archive/test_set/test_set\",\n",
    "        target_size=(64, 64),\n",
    "        batch_size=32,\n",
    "        class_mode=\"binary\"\n",
    "      )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ac28e91",
   "metadata": {
    "papermill": {
     "duration": 0.037747,
     "end_time": "2021-12-09T16:41:01.640320",
     "exception": false,
     "start_time": "2021-12-09T16:41:01.602573",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# **Creating the Model**\n",
    "### Importing useful models for CNN Layers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "17f5510d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:01.723446Z",
     "iopub.status.busy": "2021-12-09T16:41:01.722577Z",
     "iopub.status.idle": "2021-12-09T16:41:01.724800Z",
     "shell.execute_reply": "2021-12-09T16:41:01.724153Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.049994Z"
    },
    "papermill": {
     "duration": 0.047156,
     "end_time": "2021-12-09T16:41:01.724965",
     "exception": false,
     "start_time": "2021-12-09T16:41:01.677809",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.layers import Conv2D\n",
    "from tensorflow.keras.layers import Dense\n",
    "from tensorflow.keras.regularizers import l2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d3060ab",
   "metadata": {
    "papermill": {
     "duration": 0.036972,
     "end_time": "2021-12-09T16:41:01.884935",
     "exception": false,
     "start_time": "2021-12-09T16:41:01.847963",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### When to use a Sequential model\n",
    "\n",
    "A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.\n",
    "\n",
    "A Sequential model is **not appropriate** when:\n",
    "- Your model has multiple inputs or multiple outputs\n",
    "- Any of your layers has multiple inputs or multiple outputs\n",
    "- You need to do layer sharing\n",
    "- You want non-linear topology (e.g. a residual connection, a multi-branch model)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "47928ef6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:01.948751Z",
     "iopub.status.busy": "2021-12-09T16:41:01.948155Z",
     "iopub.status.idle": "2021-12-09T16:41:01.987460Z",
     "shell.execute_reply": "2021-12-09T16:41:01.987920Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.066484Z"
    },
    "papermill": {
     "duration": 0.066833,
     "end_time": "2021-12-09T16:41:01.988055",
     "exception": false,
     "start_time": "2021-12-09T16:41:01.921222",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From c:\\Users\\yashk\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "cnn = tf.keras.models.Sequential()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "53029d6b",
   "metadata": {
    "papermill": {
     "duration": 0.0235,
     "end_time": "2021-12-09T16:41:02.035218",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.011718",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Step 1 - **Convolution**\n",
    "This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.\n",
    "\n",
    "When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers or None, does not include the sample axis), e.g. `input_shape=(64, 64, 3)` for 64x64 RGB pictures in `data_format=\"channels_last\"`. You can use None when a dimension has variable size.\n",
    "\n",
    "**Arguments Used:**\n",
    "- **filters:** Integer, the dimensionality of the output space.\n",
    "- **padding:** one of \"valid\" or \"same\". \"valid\" means no padding. \"same\" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.\n",
    "- **kernel_size:** An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.\n",
    "- **activation:** Activation function to use. If you don't specify anything, no activation is applied.\n",
    "- **strides:** An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "606c2e76",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:02.094073Z",
     "iopub.status.busy": "2021-12-09T16:41:02.091109Z",
     "iopub.status.idle": "2021-12-09T16:41:02.127810Z",
     "shell.execute_reply": "2021-12-09T16:41:02.127413Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.078905Z"
    },
    "papermill": {
     "duration": 0.069426,
     "end_time": "2021-12-09T16:41:02.127954",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.058528",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cnn.add(tf.keras.layers.Conv2D(filters=32,padding=\"same\",kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06a345cc",
   "metadata": {
    "papermill": {
     "duration": 0.023143,
     "end_time": "2021-12-09T16:41:02.174544",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.151401",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Step 2 - **Pooling**\n",
    "Downsamples the input along its spatial dimensions (height and width) by taking the maximum value over an input window (of size defined by pool_size) for each channel of the input. The window is shifted by strides along each dimension."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d6abdcf7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:02.234084Z",
     "iopub.status.busy": "2021-12-09T16:41:02.231455Z",
     "iopub.status.idle": "2021-12-09T16:41:02.239047Z",
     "shell.execute_reply": "2021-12-09T16:41:02.238539Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.096071Z"
    },
    "papermill": {
     "duration": 0.039407,
     "end_time": "2021-12-09T16:41:02.239164",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.199757",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From c:\\Users\\yashk\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\layers\\pooling\\max_pooling2d.py:161: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e1ea4f85",
   "metadata": {
    "papermill": {
     "duration": 0.025212,
     "end_time": "2021-12-09T16:41:02.290614",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.265402",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### **Adding a second convolutional layer**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5e83d557",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:02.347715Z",
     "iopub.status.busy": "2021-12-09T16:41:02.346896Z",
     "iopub.status.idle": "2021-12-09T16:41:02.357173Z",
     "shell.execute_reply": "2021-12-09T16:41:02.356718Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.106671Z"
    },
    "papermill": {
     "duration": 0.041315,
     "end_time": "2021-12-09T16:41:02.357277",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.315962",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',kernel_size=3, activation='relu'))\n",
    "cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "058cba68",
   "metadata": {
    "papermill": {
     "duration": 0.023688,
     "end_time": "2021-12-09T16:41:02.404906",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.381218",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Step 3 - **Flattening**\n",
    "**Flattens the input. Does not affect the batch size.**\n",
    "\n",
    "**Note:** If inputs are shaped (batch,) without a feature axis, then flattening adds an extra channel dimension and output shape is (batch, 1)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2739732c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:02.456520Z",
     "iopub.status.busy": "2021-12-09T16:41:02.456002Z",
     "iopub.status.idle": "2021-12-09T16:41:02.461973Z",
     "shell.execute_reply": "2021-12-09T16:41:02.461543Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.124106Z"
    },
    "papermill": {
     "duration": 0.033814,
     "end_time": "2021-12-09T16:41:02.462102",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.428288",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cnn.add(tf.keras.layers.Flatten())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cf70e899",
   "metadata": {
    "papermill": {
     "duration": 0.031548,
     "end_time": "2021-12-09T16:41:02.517104",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.485556",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Step 4 - **Full Connection**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b242f827",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:02.573498Z",
     "iopub.status.busy": "2021-12-09T16:41:02.572682Z",
     "iopub.status.idle": "2021-12-09T16:41:02.577713Z",
     "shell.execute_reply": "2021-12-09T16:41:02.577327Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.136721Z"
    },
    "papermill": {
     "duration": 0.037421,
     "end_time": "2021-12-09T16:41:02.577814",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.540393",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c72375b1",
   "metadata": {
    "papermill": {
     "duration": 0.023366,
     "end_time": "2021-12-09T16:41:02.624556",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.601190",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### Step 5 - **Output Layer**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2cd64d7c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:02.678489Z",
     "iopub.status.busy": "2021-12-09T16:41:02.674631Z",
     "iopub.status.idle": "2021-12-09T16:41:02.685467Z",
     "shell.execute_reply": "2021-12-09T16:41:02.685061Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.151023Z"
    },
    "papermill": {
     "duration": 0.037286,
     "end_time": "2021-12-09T16:41:02.685571",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.648285",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cnn.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation\n",
    "             ='linear'))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d6310fd9",
   "metadata": {
    "papermill": {
     "duration": 0.023194,
     "end_time": "2021-12-09T16:41:02.732058",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.708864",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### **Printing out the summary of the Layers**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "afedf5ac",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:02.784390Z",
     "iopub.status.busy": "2021-12-09T16:41:02.783255Z",
     "iopub.status.idle": "2021-12-09T16:41:02.789302Z",
     "shell.execute_reply": "2021-12-09T16:41:02.789952Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.165547Z"
    },
    "papermill": {
     "duration": 0.034184,
     "end_time": "2021-12-09T16:41:02.790106",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.755922",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv2d (Conv2D)             (None, 32, 32, 32)        896       \n",
      "                                                                 \n",
      " max_pooling2d (MaxPooling2  (None, 16, 16, 32)        0         \n",
      " D)                                                              \n",
      "                                                                 \n",
      " conv2d_1 (Conv2D)           (None, 16, 16, 32)        9248      \n",
      "                                                                 \n",
      " max_pooling2d_1 (MaxPoolin  (None, 8, 8, 32)          0         \n",
      " g2D)                                                            \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 2048)              0         \n",
      "                                                                 \n",
      " dense (Dense)               (None, 128)               262272    \n",
      "                                                                 \n",
      " dense_1 (Dense)             (None, 1)                 129       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 272545 (1.04 MB)\n",
      "Trainable params: 272545 (1.04 MB)\n",
      "Non-trainable params: 0 (0.00 Byte)\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "cnn.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "feaa7562",
   "metadata": {
    "papermill": {
     "duration": 0.023612,
     "end_time": "2021-12-09T16:41:02.886758",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.863146",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Training the CNN**\n",
    "### Compiling the CNN\n",
    "#### Attributes:\n",
    "- **optimizer:-** String (name of optimizer) or optimizer instance.\n",
    "- **loss:-** Loss function.\n",
    "- **metrics:-** List of metrics to be evaluated by the model during training and testing. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "0562c757",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:02.943095Z",
     "iopub.status.busy": "2021-12-09T16:41:02.939713Z",
     "iopub.status.idle": "2021-12-09T16:41:02.949428Z",
     "shell.execute_reply": "2021-12-09T16:41:02.948969Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.177472Z"
    },
    "papermill": {
     "duration": 0.038279,
     "end_time": "2021-12-09T16:41:02.949535",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.911256",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From c:\\Users\\yashk\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\optimizers\\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c5040c40",
   "metadata": {
    "papermill": {
     "duration": 0.024229,
     "end_time": "2021-12-09T16:41:02.997551",
     "exception": false,
     "start_time": "2021-12-09T16:41:02.973322",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### **Training the CNN on the Training set and evaluating it on the Test set**\n",
    "#### Attributes:-\n",
    "- **x:-** Input data\n",
    "- **validation_data:-** Data on which to evaluate the loss and any model metrics at the end of each epoch.\n",
    "- **epochs:-** Integer. Number of epochs to train the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a5c20958",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:41:03.048316Z",
     "iopub.status.busy": "2021-12-09T16:41:03.047651Z",
     "iopub.status.idle": "2021-12-09T16:50:57.545418Z",
     "shell.execute_reply": "2021-12-09T16:50:57.544964Z",
     "shell.execute_reply.started": "2021-12-09T16:12:51.189191Z"
    },
    "papermill": {
     "duration": 594.524275,
     "end_time": "2021-12-09T16:50:57.545558",
     "exception": false,
     "start_time": "2021-12-09T16:41:03.021283",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/15\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From c:\\Users\\yashk\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\utils\\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.\n",
      "\n",
      "WARNING:tensorflow:From c:\\Users\\yashk\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\keras\\src\\engine\\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.\n",
      "\n",
      "251/251 [==============================] - 49s 185ms/step - loss: 0.9034 - accuracy: 0.5671 - val_loss: 0.7784 - val_accuracy: 0.5902\n",
      "Epoch 2/15\n",
      "251/251 [==============================] - 21s 82ms/step - loss: 0.7385 - accuracy: 0.6521 - val_loss: 0.7342 - val_accuracy: 0.7044\n",
      "Epoch 3/15\n",
      "251/251 [==============================] - 19s 75ms/step - loss: 0.6716 - accuracy: 0.6848 - val_loss: 0.6157 - val_accuracy: 0.7128\n",
      "Epoch 4/15\n",
      "251/251 [==============================] - 19s 77ms/step - loss: 0.6312 - accuracy: 0.7023 - val_loss: 0.5893 - val_accuracy: 0.7400\n",
      "Epoch 5/15\n",
      "251/251 [==============================] - 23s 90ms/step - loss: 0.6055 - accuracy: 0.7129 - val_loss: 0.5652 - val_accuracy: 0.7514\n",
      "Epoch 6/15\n",
      "251/251 [==============================] - 23s 92ms/step - loss: 0.5903 - accuracy: 0.7217 - val_loss: 0.5801 - val_accuracy: 0.7608\n",
      "Epoch 7/15\n",
      "251/251 [==============================] - 20s 80ms/step - loss: 0.5660 - accuracy: 0.7379 - val_loss: 0.5755 - val_accuracy: 0.7588\n",
      "Epoch 8/15\n",
      "251/251 [==============================] - 20s 80ms/step - loss: 0.5519 - accuracy: 0.7405 - val_loss: 0.5543 - val_accuracy: 0.7721\n",
      "Epoch 9/15\n",
      "251/251 [==============================] - 25s 99ms/step - loss: 0.5502 - accuracy: 0.7440 - val_loss: 0.5706 - val_accuracy: 0.7751\n",
      "Epoch 10/15\n",
      "251/251 [==============================] - 22s 87ms/step - loss: 0.5377 - accuracy: 0.7513 - val_loss: 0.6298 - val_accuracy: 0.7543\n",
      "Epoch 11/15\n",
      "251/251 [==============================] - 25s 102ms/step - loss: 0.5131 - accuracy: 0.7631 - val_loss: 0.5230 - val_accuracy: 0.7528\n",
      "Epoch 12/15\n",
      "251/251 [==============================] - 20s 81ms/step - loss: 0.5085 - accuracy: 0.7629 - val_loss: 0.5079 - val_accuracy: 0.7909\n",
      "Epoch 13/15\n",
      "251/251 [==============================] - 21s 82ms/step - loss: 0.4956 - accuracy: 0.7725 - val_loss: 0.4848 - val_accuracy: 0.7874\n",
      "Epoch 14/15\n",
      "251/251 [==============================] - 21s 85ms/step - loss: 0.4825 - accuracy: 0.7725 - val_loss: 0.5786 - val_accuracy: 0.7810\n",
      "Epoch 15/15\n",
      "251/251 [==============================] - 19s 74ms/step - loss: 0.4785 - accuracy: 0.7830 - val_loss: 0.5446 - val_accuracy: 0.7217\n"
     ]
    }
   ],
   "source": [
    "r=cnn.fit(x = training_set, validation_data = test_set, epochs = 15)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74cc779f",
   "metadata": {
    "papermill": {
     "duration": 0.975733,
     "end_time": "2021-12-09T16:50:59.495484",
     "exception": false,
     "start_time": "2021-12-09T16:50:58.519751",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## **Ploting the Train loss,val loss and train acc, val acc**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "55120a1a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:51:01.730148Z",
     "iopub.status.busy": "2021-12-09T16:51:01.729056Z",
     "iopub.status.idle": "2021-12-09T16:51:02.110801Z",
     "shell.execute_reply": "2021-12-09T16:51:02.110380Z",
     "shell.execute_reply.started": "2021-12-09T16:21:59.058626Z"
    },
    "papermill": {
     "duration": 1.362963,
     "end_time": "2021-12-09T16:51:02.110951",
     "exception": false,
     "start_time": "2021-12-09T16:51:00.747988",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAiMAAAGdCAYAAADAAnMpAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAABhE0lEQVR4nO3dd1iV9f/H8efhsKcgCoIobnHkFkdbTLPM0TA1tV1m0/Yw2/Zr2E7b9W1pQ9PKNDVnuRI1Fbe4BVxsmef8/rgBpRQZ53AzXo/r4vKM+9z3+6ByXnymxW632xERERExiYvZBYiIiEjtpjAiIiIiplIYEREREVMpjIiIiIipFEZERETEVAojIiIiYiqFERERETGVwoiIiIiYytXsAkrDZrNx6NAh/Pz8sFgsZpcjIiIipWC320lLSyMsLAwXl7O3f1SLMHLo0CEiIiLMLkNERETKYf/+/TRs2PCsz1eLMOLn5wcYb8bf39/kakRERKQ0UlNTiYiIKPocP5tqEUYKu2b8/f0VRkRERKqZcw2x0ABWERERMZXCiIiIiJhKYURERERMVS3GjIiISM1lt9vJy8sjPz/f7FKkjKxWK66urhVedkNhRERETJOTk8Phw4fJzMw0uxQpJ29vbxo0aIC7u3u5z6EwIiIiprDZbMTHx2O1WgkLC8Pd3V0LW1YjdrudnJwcjhw5Qnx8PC1atChxYbOSKIyIiIgpcnJysNlsRERE4O3tbXY5Ug5eXl64ubmxd+9ecnJy8PT0LNd5NIBVRERMVd7fpqVqcMTfn/4FiIiIiKkURkRERMRUCiMiIiImi4yM5M033zT9HGbRAFYREZEyuvjii+nYsaPDPvzXrFmDj4+PQ85VHZWrZeS9994jMjIST09PoqOjWb169VmPzc3N5bnnnqNZs2Z4enrSoUMH5s6dW+6CHembVfu4+5tYklKzzC5FRERqmMLF3EqjXr16tXpGUZnDyPTp0xk/fjwTJ04kNjaWDh060K9fP5KSks54/FNPPcUHH3zAO++8Q1xcHHfeeSdDhgxh3bp1FS6+or5ZvZdf/jnMyvjjZpciIiIYH+CZOXmmfNnt9lLVeOONN7JkyRLeeustLBYLFouFPXv2sHjxYiwWC7/99htdunTBw8OD5cuXs2vXLgYNGkRISAi+vr5069aNBQsWFDvnv7tYLBYLH3/8MUOGDMHb25sWLVowe/bsMn0v9+3bx6BBg/D19cXf35/rrruOxMTEouc3bNjAJZdcgp+fH/7+/nTp0oW///4bgL179zJw4EACAwPx8fGhbdu2zJkzp0zXL4syd9NMnjyZ2267jZtuugmAqVOn8uuvv/Lpp5/y2GOP/ef4L7/8kieffJIBAwYAMHbsWBYsWMDrr7/OV199VcHyKya6SV02HUxl1e5jXNUhzNRaREQETubm0+bpeaZcO+65fni7n/tj8a233mL79u20a9eO5557DjBaNvbs2QPAY489xmuvvUbTpk0JDAxk//79DBgwgBdffBEPDw/+97//MXDgQLZt20ajRo3Oep1nn32WV155hVdffZV33nmHkSNHsnfvXoKCgs5Zo81mKwoiS5YsIS8vj3HjxjFs2DAWL14MwMiRI+nUqRNTpkzBarWyfv163NzcABg3bhw5OTksXboUHx8f4uLi8PX1Ped1y6tMYSQnJ4e1a9fy+OOPFz3m4uJCTEwMK1asOONrsrOz/7MIipeXF8uXLz/rdbKzs8nOzi66n5qaWpYySy26SRCfLI9n5e5jTjm/iIjUPAEBAbi7u+Pt7U1oaOh/nn/uuefo27dv0f2goCA6dOhQdP/5559n5syZzJ49m7vvvvus17nxxhsZPnw4AC+99BJvv/02q1evpn///uesceHChWzcuJH4+HgiIiIA+N///kfbtm1Zs2YN3bp1Y9++fTz88MO0bt0agBYtWhS9ft++fVx99dW0b98egKZNm57zmhVRpjBy9OhR8vPzCQkJKfZ4SEgIW7duPeNr+vXrx+TJk7nwwgtp1qwZCxcuZMaMGSVuiDRp0iSeffbZspRWLt2bBGGxwK4jGRxJy6aen4fTrykiImfn5WYl7rl+pl3bEbp27Vrsfnp6Os888wy//vorhw8fJi8vj5MnT7Jv374Sz3PeeecV3fbx8cHf3/+sQyL+bcuWLURERBQFEYA2bdpQp04dtmzZQrdu3Rg/fjy33norX375JTExMVx77bU0a9YMgHvvvZexY8fy+++/ExMTw9VXX12sHkdz+tTet956ixYtWtC6dWvc3d25++67uemmm0pcse3xxx8nJSWl6Gv//v1Oqa2OtzutQ/0BWK1xIyIiprNYLHi7u5ry5ah9cf49K+ahhx5i5syZvPTSSyxbtoz169fTvn17cnJySjxPYZfJ6d8bm83mkBoBnnnmGTZv3swVV1zBH3/8QZs2bZg5cyYAt956K7t372bUqFFs3LiRrl278s477zjs2v9WpjASHByM1WotNgAGIDEx8YxNVWD0o/30009kZGSwd+9etm7diq+vb4lNPh4eHvj7+xf7cpboJkbf26p4ddWIiEjpuLu7l9jCf7o///yTG2+8kSFDhtC+fXtCQ0OLxpc4S1RUFPv37y/2y3xcXBzJycm0adOm6LGWLVvywAMP8PvvvzN06FA+++yzouciIiK48847mTFjBg8++CAfffSR0+otUxhxd3enS5cuLFy4sOgxm83GwoUL6dmzZ4mv9fT0JDw8nLy8PH788UcGDRpUvoodrEfTgjCyWy0jIiJSOpGRkaxatYo9e/Zw9OjRElssWrRowYwZM1i/fj0bNmxgxIgRDm3hOJOYmBjat2/PyJEjiY2NZfXq1YwePZqLLrqIrl27cvLkSe6++24WL17M3r17+fPPP1mzZg1RUVEA3H///cybN4/4+HhiY2NZtGhR0XPOUOZumvHjx/PRRx/xxRdfsGXLFsaOHUtGRkbR7JrRo0cXG+C6atUqZsyYwe7du1m2bBn9+/fHZrPxyCOPOO5dVED3JnUB2JaYxvGMkpvMREREwOh6sVqttGnThnr16pU4/mPy5MkEBgbSq1cvBg4cSL9+/ejcubNT67NYLMyaNYvAwEAuvPBCYmJiaNq0KdOnTwfAarVy7NgxRo8eTcuWLbnuuuu4/PLLi8Zr5ufnM27cOKKioujfvz8tW7bk/fffd1699tJOrD7Nu+++y6uvvkpCQgIdO3bk7bffJjo6GjBWpYuMjOTzzz8HYMmSJYwdO5bdu3fj6+vLgAEDePnllwkLK/1U2tTUVAICAkhJSXFKl02/N5ayLTGNqTd0pn+7Bg4/v4iI/FdWVhbx8fE0adKk3FvPi/lK+nss7ed3uZaDv/vuu886Halw/nKhiy66iLi4uPJcptJENw1iW2IaK3cfVxgRERGpZNooD2PxM4BVmlEjIiJS6RRGMNYbAdiakEpypsaNiIiIVCaFEaCenwfN6vlgt2u9ERERkcqmMFKgR1N11YiIiJhBYaRAdFEY0eJnIiIilUlhpECPgnEjmw+lknIy1+RqREREag+FkQL1/T1pEmyMG/l7j7pqREREKovCyGlO7VOjMCIiIs4VGRnJm2++edbnb7zxRgYPHlxp9ZhJYeQ0RYNYd2vciIiISGVRGDlNdMGmeZsOpZKWpXEjIiIilUFh5DQNArxoFORNvs3O33tPmF2OiIhUQR9++CFhYWH/2Xl30KBB3HzzzQDs2rWLQYMGERISgq+vL926dWPBggUVum52djb33nsv9evXx9PTk/PPP581a9YUPX/ixAlGjhxJvXr18PLyokWLFnz22WcA5OTkcPfdd9OgQQM8PT1p3LgxkyZNqlA9jqQw8i9F40Z2a9yIiEils9shJ8Ocr1LuG3vttddy7NgxFi1aVPTY8ePHmTt3LiNHjgQgPT2dAQMGsHDhQtatW0f//v0ZOHBgibv7nssjjzzCjz/+yBdffEFsbCzNmzenX79+HD9ufF5NmDCBuLg4fvvtN7Zs2cKUKVMIDg4G4O2332b27Nl89913bNu2ja+//prIyMhy1+Jo5doorybr0bQu3689oPVGRETMkJsJL5V+V3eHeuIQuPuc87DAwEAuv/xyvvnmG/r06QPADz/8QHBwMJdccgkAHTp0oEOHDkWvef7555k5cyazZ88+60azJcnIyGDKlCl8/vnnXH755QB89NFHzJ8/n08++YSHH36Yffv20alTJ7p27QpQLGzs27ePFi1acP7552OxWGjcuHGZa3AmtYz8S+G4kY0HUsjIzjO5GhERqYpGjhzJjz/+SHZ2NgBff/01119/PS4uxsdqeno6Dz30EFFRUdSpUwdfX1+2bNlS7paRXbt2kZubS+/evYsec3Nzo3v37mzZsgWAsWPHMm3aNDp27MgjjzzCX3/9VXTsjTfeyPr162nVqhX33nsvv//+e3nfulOoZeRfGgZ6E17Hi4PJJ1m79wQXtqxndkkiIrWHm7fRQmHWtUtp4MCB2O12fv31V7p168ayZct44403ip5/6KGHmD9/Pq+99hrNmzfHy8uLa665hpwc523Gevnll7N3717mzJnD/Pnz6dOnD+PGjeO1116jc+fOxMfH89tvv7FgwQKuu+46YmJi+OGHH5xWT1kojJxBdNMgZsQeZFX8MYUREZHKZLGUqqvEbJ6engwdOpSvv/6anTt30qpVKzp37lz0/J9//smNN97IkCFDAKOlZM+ePeW+XrNmzXB3d+fPP/8s6mLJzc1lzZo13H///UXH1atXjzFjxjBmzBguuOACHn74YV577TUA/P39GTZsGMOGDeOaa66hf//+HD9+nKCgoHLX5SgKI2fQo0ldI4xoEKuIiJzFyJEjufLKK9m8eTM33HBDsedatGjBjBkzGDhwIBaLhQkTJvxn9k1Z+Pj4MHbsWB5++GGCgoJo1KgRr7zyCpmZmdxyyy0APP3003Tp0oW2bduSnZ3NL7/8QlRUFACTJ0+mQYMGdOrUCRcXF77//ntCQ0OpU6dOuWtyJIWRMyhc/GzDgWRO5uTj5W41uSIREalqLr30UoKCgti2bRsjRowo9tzkyZO5+eab6dWrF8HBwTz66KOkpqZW6Hovv/wyNpuNUaNGkZaWRteuXZk3bx6BgYEAuLu78/jjj7Nnzx68vLy44IILmDZtGgB+fn688sor7NixA6vVSrdu3ZgzZ07RGBezWez2Us5lMlFqaioBAQGkpKTg7+/v9OvZ7XZ6vfwHh1Oy+ObWaHo1D3b6NUVEapusrCzi4+Np0qQJnp6eZpcj5VTS32NpP7+rRiSqYiwWS9F6Iyu1NLyIiIhTKYycRXRBV81KbZonIiLiVAojZ1E4bmT9/mSycvNNrkZERKTmUhg5i8i63tT38yAnz8b6/clmlyMiIlJjKYychcViOdVVo3EjIiIiTqMwUgJtmici4nzVYFKnlMARf38KIyXoUbBPTey+E2TnadyIiIgjubm5AZCZmWlyJVIRhX9/hX+f5aFFz0rQrJ4vwb7uHE3P4Z8DKXSLNH/JXBGRmsJqtVKnTh2SkpIA8Pb2xmKxmFyVlJbdbiczM5OkpCTq1KmD1Vr+BUIVRkpgrDdSl183HmbV7mMKIyIiDhYaGgpQFEik+qlTp07R32N5KYycQ3TTIH7deJiVu49z96VmVyMiUrNYLBYaNGhA/fr1yc3NNbscKSM3N7cKtYgUUhg5h+gmxoyatXtPkJtvw82qYTYiIo5mtVod8qEm1ZM+Wc+hRX1fgnzcOZmbzz8HUswuR0REpMZRGDkHFxcL3QvGiqyK13ojIiIijqYwUgrRTQs3zdN6IyIiIo6mMFIKReNG9hwnL99mcjUiIiI1i8JIKbQO9SPAy42MnHw2HUo1uxwREZEaRWGkFFxcLHQvWhpe40ZEREQcSWGklIr2qYnXuBERERFHUhgppR4FO/iuiT9Ovk2bOomIiDiKwkgpRTXwx8/TlbTsPOI0bkRERMRhFEZKyar1RkRERJxCYaQMtN6IiIiI4ymMlEHheiOr449p3IiIiIiDKIyUQdswf3w9XEnNymNrgsaNiIiIOILCSBm4Wl3o0jgQgFXqqhEREXEIhZEyKpziq0GsIiIijqEwUkaFg1hXxR/HpnEjIiIiFaYwUkbtwwPwdreSnJnL9qQ0s8sRERGp9hRGyshN40ZEREQcSmGkHDRuRERExHEURsqhaNO83cex2zVuREREpCIURsrhvIZ18HRz4VhGDjuT0s0uR0REpFpTGCkHd1cXOjcyxo2sjNe4ERERkYpQGCmnwqXhV+3WuBEREZGKUBgppx6nrTeicSMiIiLlpzBSTh0i6uDu6sKRtGx2H80wuxwREZFqS2GknDzdrHSKqANovREREZGKUBipgGitNyIiIlJhCiMV0EPrjYiIiFSYwkgFdGoUiLvVhYTULPYdzzS7HBERkWpJYaQCvNytdIgIAGClpviKiIiUi8JIBZ1ab0SDWEVERMpDYaSCok9bb0RERETKTmGkgro0DsTVxcLB5JPs17gRERGRMlMYqSBvd1fOa6hxIyIiIuWlMOIAp9YbUVeNiIhIWSmMOEB04XojWvxMRESkzBRGHKBrZBBWFwv7j5/kUPJJs8sRERGpVhRGHMDXw5V24ca4EbWOiIiIlE25wsh7771HZGQknp6eREdHs3r16hKPf/PNN2nVqhVeXl5ERETwwAMPkJWVVa6Cq6rCpeFX7tK4ERERkbIocxiZPn0648ePZ+LEicTGxtKhQwf69etHUlLSGY//5ptveOyxx5g4cSJbtmzhk08+Yfr06TzxxBMVLr4qObXeiFpGREREyqLMYWTy5Mncdttt3HTTTbRp04apU6fi7e3Np59+esbj//rrL3r37s2IESOIjIzksssuY/jw4edsTaluukYG4WKBPccySUytWa0+IiIizlSmMJKTk8PatWuJiYk5dQIXF2JiYlixYsUZX9OrVy/Wrl1bFD52797NnDlzGDBgwFmvk52dTWpqarGvqs7f0422YVpvREREpKzKFEaOHj1Kfn4+ISEhxR4PCQkhISHhjK8ZMWIEzz33HOeffz5ubm40a9aMiy++uMRumkmTJhEQEFD0FRERUZYyTVM4xXel9qkREREpNafPplm8eDEvvfQS77//PrGxscyYMYNff/2V559//qyvefzxx0lJSSn62r9/v7PLdIhTi5+pZURERKS0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ZrPZWLhwIT179jSxsspht9u5++67mTlzJn/88QdNmjQxu6RK1adPHzZu3Mj69euLvrp27crIkSNZv349VqvV7BKdrnfv3v+Zzr19+3YaN25sUkWVKzMzExeX4j/+rFYrNpvNpIrM1aRJE0JDQ4v9TExNTWXVqlW14mdiocIgsmPHDhYsWEDdunXNLqnSjBo1in/++afYz8WwsDAefvhh5s2b5/Tr19pumvHjxzNmzBi6du1K9+7defPNN8nIyOCmm24yuzSnGzduHN988w2zZs3Cz8+vqE84ICAALy8vk6tzPj8/v/+Mj/Hx8aFu3bq1ZtzMAw88QK9evXjppZe47rrrWL16NR9++CEffvih2aVVioEDB/Liiy/SqFEj2rZty7p165g8eTI333yz2aU5TXp6Ojt37iy6Hx8fz/r16wkKCqJRo0bcf//9vPDCC7Ro0YImTZowYcIEwsLCGDx4sHlFO1hJ34MGDRpwzTXXEBsbyy+//EJ+fn7Rz8agoCDc3d3NKtthzvVv4N/hy83NjdDQUFq1auX84iplzk4V9c4779gbNWpkd3d3t3fv3t2+cuVKs0uqFMAZvz777DOzSzNNbZvaa7fb7T///LO9Xbt2dg8PD3vr1q3tH374odklVZrU1FT7fffdZ2/UqJHd09PT3rRpU/uTTz5pz87ONrs0p1m0aNEZ/9+PGTPGbrcb03snTJhgDwkJsXt4eNj79Olj37Ztm7lFO1hJ34P4+Piz/mxctGiR2aU7xLn+DfxbZU7ttdjtNXjJQREREanyauWYEREREak6FEZERETEVAojIiIiYiqFERERETGVwoiIiIiYSmFERERETKUwIiIiIqZSGBERERFTKYyIiIiIqRRGRERExFQKIyIiImIqhREREREx1f8DJiDFR/veAd4AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot the loss\n",
    "import matplotlib.pyplot as plt\n",
    "plt.plot(r.history['loss'], label='train loss')\n",
    "plt.plot(r.history['val_loss'], label='val loss')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "# plot the accuracy\n",
    "plt.plot(r.history['accuracy'], label='train acc')\n",
    "plt.plot(r.history['val_accuracy'], label='val acc')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38a02d8a",
   "metadata": {
    "papermill": {
     "duration": 1.030247,
     "end_time": "2021-12-09T16:51:04.145472",
     "exception": false,
     "start_time": "2021-12-09T16:51:03.115225",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### **Saving the trained model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "29b453a2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:51:06.121683Z",
     "iopub.status.busy": "2021-12-09T16:51:06.120881Z",
     "iopub.status.idle": "2021-12-09T16:51:06.154749Z",
     "shell.execute_reply": "2021-12-09T16:51:06.155181Z",
     "shell.execute_reply.started": "2021-12-09T16:21:59.442276Z"
    },
    "papermill": {
     "duration": 1.009299,
     "end_time": "2021-12-09T16:51:06.155329",
     "exception": false,
     "start_time": "2021-12-09T16:51:05.146030",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "cnn.save('./classification.h5')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4cfe44ad",
   "metadata": {
    "papermill": {
     "duration": 0.981247,
     "end_time": "2021-12-09T16:51:08.117173",
     "exception": false,
     "start_time": "2021-12-09T16:51:07.135926",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "### **Taking the sample image converting the image to an array and predicting the result**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "a83444f5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:51:10.073824Z",
     "iopub.status.busy": "2021-12-09T16:51:10.073310Z",
     "iopub.status.idle": "2021-12-09T16:51:10.204002Z",
     "shell.execute_reply": "2021-12-09T16:51:10.203517Z",
     "shell.execute_reply.started": "2021-12-09T16:21:59.599041Z"
    },
    "papermill": {
     "duration": 1.111535,
     "end_time": "2021-12-09T16:51:10.204136",
     "exception": false,
     "start_time": "2021-12-09T16:51:09.092601",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 76ms/step\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing import image\n",
    "test_image = image.load_img('archive/training_set/training_set/cats/cat.1028.jpg', target_size = (64,64))\n",
    "test_image = image.img_to_array(test_image)\n",
    "test_image=test_image/255\n",
    "test_image = np.expand_dims(test_image, axis = 0)\n",
    "result = cnn.predict(test_image)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "740cd2d7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The image classified is cat\n"
     ]
    }
   ],
   "source": [
    "if result[0]<0:\n",
    "    print(\"The image classified is cat\")\n",
    "else:\n",
    "    print(\"The image classified is dog\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "27f4cef3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 103ms/step\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing import image\n",
    "test_image = image.load_img('archive/training_set/training_set/dogs/dog.1077.jpg', target_size = (64,64))\n",
    "test_image = image.img_to_array(test_image)\n",
    "test_image=test_image/255\n",
    "test_image = np.expand_dims(test_image, axis = 0)\n",
    "result = cnn.predict(test_image)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "7de03a0c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2021-12-09T16:51:12.440582Z",
     "iopub.status.busy": "2021-12-09T16:51:12.438791Z",
     "iopub.status.idle": "2021-12-09T16:51:12.443143Z",
     "shell.execute_reply": "2021-12-09T16:51:12.442499Z",
     "shell.execute_reply.started": "2021-12-09T16:21:59.695901Z"
    },
    "papermill": {
     "duration": 0.982728,
     "end_time": "2021-12-09T16:51:12.443295",
     "exception": false,
     "start_time": "2021-12-09T16:51:11.460567",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The image classified is dog\n"
     ]
    }
   ],
   "source": [
    "if result[0]<0:\n",
    "    print(\"The image classified is cat\")\n",
    "else:\n",
    "    print(\"The image classified is dog\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56eba700",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 635.069725,
   "end_time": "2021-12-09T16:51:18.907455",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2021-12-09T16:40:43.837730",
   "version": "2.3.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
