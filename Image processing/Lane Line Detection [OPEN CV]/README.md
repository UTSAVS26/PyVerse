## **Lane-Line-Detection**

### 🎯 **Goal**

The main goal of this project is to detect lane lines in images or video streams using computer vision techniques. The purpose is to provide a solution that can be used in autonomous driving systems to ensure vehicles can stay within the correct lane by identifying lane markings effectively.

### 🧵 **Dataset**

This project does not use a specific pre-labeled dataset. Instead, it processes video files or images provided by the user to detect lane lines in real-time. You can use any video or image containing road lane markings for testing the lane detection functionality.

### 🧾 **Description**

Lane-Line-Detection is a computer vision project implemented in Python using OpenCV. It detects and highlights lane lines in a video or image by applying various image processing techniques such as grayscale conversion, Gaussian blur, Canny edge detection, region of interest selection, and Hough Transform to identify lane boundaries.

### 🧮 **What I had done!**

1. Preprocessed the input video/image by converting it to grayscale and applying Gaussian blur to reduce noise.
2. Detected edges using Canny edge detection.
3. Defined a region of interest (ROI) to focus on the lane area.
4. Applied Hough Transform to detect lines within the ROI.
5. Drew the lane lines onto the original image/video using the detected lines.
6. Displayed the processed image/video with lane lines highlighted.

### 🚀 **Models Implemented**

- **Canny Edge Detection**: Used for detecting edges in the image based on gradients. It helps identify the potential boundaries of lane lines.
- **Hough Line Transform**: Applied to detect straight lines within the image's region of interest. This is ideal for lane line detection as lane markings are often straight.
  
These techniques are chosen for their effectiveness in edge and line detection, crucial for identifying lane boundaries.

### 📚 **Libraries Needed**

- OpenCV
- NumPy
- Matplotlib (for visualization)

### 📊 **Exploratory Data Analysis Results**

**Visualizations**:

- **Original Image**:
  ![Original Image](./testimg.jpg)

- **Processed Image**:
  ![Processed Image](./testimageresult.png)

- **Lane Line Detection in Action (GIF)**:
  ![Lane Line Detection GIF](./finalresult.gif)

These visualizations show how lane lines are detected and highlighted in both images and videos.

### 📈 **Performance of the Models based on the Accuracy Scores**

Since this is a computer vision project based on image processing techniques, accuracy is more qualitative rather than quantitative. The effectiveness is evaluated visually by observing how well the lane lines are detected in different lighting conditions, road structures, and image/video qualities.

### 📢 **Conclusion**

The lane detection project effectively identifies lane lines in both images and video streams. By using edge detection and line finding techniques, it demonstrates good performance in standard road scenarios. The use of Canny edge detection and Hough Line Transform provides reliable detection, making the system suitable for real-time applications in autonomous vehicles.

### ✒️ **Your Signature**

Aviral Garg  
[GitHub](https://github.com/aviralgarg05) | [LinkedIn](https://linkedin.com/in/aviralgarg05)
