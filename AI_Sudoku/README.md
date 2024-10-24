# AI_Sudoku

GUI Smart Sudoku Solver that tries to extract a sudoku puzzle from a photo and solve it.

## Installation

1. Download and install Python3 from [here](https://www.python.org/downloads/)
2. I recommend using [virtualenv](https://virtualenv.pypa.io/en/latest/). Download virtualenv by opening a terminal and typing:
    ```bash
    pip install virtualenv
    ```
3. Create a virtual environment with the name sudokuenv.

   * Windows
   ```bash
   virtualenv sudokuenv
   cd sudokuenv/Scripts
   activate
   ```
   * Linux:
   ```bash
   source sudokuenv/bin/activate
    ```
4. Clone this repository, extract it if you downloaded a .zip or .tar file and cd into the cloned repository.

    * For Example:
    ```bash
    cd A:\AI_Sudoku-master
    ```
5. Install the required packages by typing:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
* Before running the application, know that you can set the **modeltype** variable in **Run.py** to either "CNN" or "KNN" to choose the Convolutional Neural Network or the K Nearest Neighbours Algorithm for Recognition. By default it is set to "KNN" and I got a way higher accuracy using KNN itself, so I would recommend that you don't change it.
    ```Python
    '''Run this file to run the application'''
    from MainUI import MainUI
    from CNN import CNN
    from KNN import KNN
    import os
    # Change the model type variable value to "CNN" to use the Convolutional Neural Network
    # Change the model type variable value to "KNN" to use the K Nearest Neighbours Classifier
    modeltype = "KNN"
    ```
* Type the below command to run the Application. You *need* to be connected to the Internet and it might take 5-10 minutes to create the **knn.sav** file so please wait patiently. This delay is only during the first run as once created, the application will use the local file
    ```bash
    python Run.py
    ```
* The GUI Homepage that opens up as soon as you run the application.


* You need to select an image of a Sudoku Puzzle through the GUI Home Page.


* Once you press **Next**, a number of stages of image processing take place which are displayed by the GUI leading up to recognition. Here are snapshots of two of the stages:


* For recognition, a **CNN** or **KNN** can be used. This option can be toggled as mentioned in the first point. Once recognized, the board is displayed and you can rectify any wrongly recognized entries in the board.


* Finally click on **reveal solution** to display the solution.


## Working

### Image Preprocessing

* **Gaussian Blurring** Blurring using a Gaussian function. This is to reduce noise and detail.


* **Adaptive Gaussian Thresholding** Adaptive thresholding with a Gaussian Function to account for different illuminations in different parts of the image.


* **Inverting** to make the digits and lines white while making the background black.


* **Dilation** with a plus shaped 3X3 Kernel to fill out any cracks in the board lines and thicken the board lines.


* **Flood Filling** Since the board will probably be the largest blob a.k.a connected component with the largest area, floodfilling from different seed points and finding all connected components followed by finding the largest floodfilled area region will give the board. 


* The **largest blob** a.k.a the board is found after the previous step. Let's call this the outerbox


* **Eroding** the grid a bit to undo the effects of the dilation on the outerbox that we did earlier.


* **Hough Line Transform** to find all the lines in the detected outerbox.


* **Merging** related lines. The lines found by the Hough Transform that are close to each other are fused together.

* **Finding the Extreme lines** . We find the border lines by choosing the nearest line from the top with slope almost 0 as the upper edge, the nearest line from the bottom with slope almost 0 as the lower edge, the nearest line from the left with slope almost infinity as the left edge and the nearest line from the right with slope almost infinity as the right edge.


* **Finding the four intersection points**. The four intersection points of these lines are found and plotted along with the lines.


* **Warping perspective**. We find the perspective matrix using the end points, correct the perspective and crop the board out of the original image.


* **Thresholding and Inverting the grid**. The cropped image from the previous step is adaptive thresholded and inverted.


* **Slicing** the grid into 81 slices to get images of each cell of the Sudoku board.


* **Blackfilling and centering the number**. Any white patches other than the number are removed by floodfilling with black from the outer layer points as seeds. Then the approximate bounding box of the number is found and centered in the image.


### Recognition

#### Convolutional Neural Network

Read about CNNs [here](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
* **Layers** A Convolution Layer, a Max Pooling layer flattened into a hidden layer followed by some Dropout Regularization, another hidden layer and finally the output layer. Each of the inner layer uses *ReLu* while the output layer uses *softmax*.
* **Compilation** *Adam* optimizer and *sparse categorical cross entropy* loss.
* **Training** The model is trained on the **MNIST** handwritten digits dataset which has around 70,000 28X28 images.
* **Accuracy** Around 98 percent accuracy on the test set.

#### K Nearest Neighbours

Read about KNN [here](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)
* **K** value used is 3.
* **Training** Trained on the **MNIST** handwritten digits dataset which has around 70,000 28X28 images.
* **Accuracy** Around 97 percent accuracy on the test set.
    
## ToDo

* Improve Accuracy.
* Resolve Bugs/Issues if any found.
* Optimize Code to make it faster.

## Contributing

Contributions are welcome :smile:

### Pull requests

Just a few guidelines:
* Write clean code with appropriate comments and add suitable error handling.
* Test the application and make sure no bugs/ issues come up.
* Open a pull request and I will be happy to acknowledge your contribution after some checking from my side.

### Issues

If you find any bugs/issues, raise an issue.








