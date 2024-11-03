# Video Frame Extraction and Entropy Analysis

This project extracts frames from a video, saves both colored and grayscale versions, calculates entropy for each grayscale frame, and analyzes histogram differences between frames. This workflow can be helpful in video analysis tasks like detecting scene changes, frame selection based on entropy, and other types of frame-by-frame video analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Plotting Frame Entropies](#plotting-frame-entropies)
- [Reducing Frames](#reducing-frames)
- [License](#license)

## Project Overview

The primary objectives of this project are:
1. Extract frames from a video file and save both the colored and grayscale versions.
2. Compute entropy for each grayscale frame to analyze its complexity.
3. Compare consecutive frames using histogram differences and select frames based on those changes.
4. Visualize the entropy across selected frames to understand variation over time.

## Features
- Extracts frames from a video file.
- Saves extracted frames as both color and grayscale images.
- Computes entropy for grayscale images.
- Compares consecutive frames based on histogram differences.
- Selects and plots frames with the highest entropy changes.

## Installation
To run this code, make sure you have the following dependencies installed:

```bash
pip install opencv-python numpy matplotlib
```

## Usage
- Place your video file (e.g., `Sample.mkv`) in the project directory.
- Run the code to extract frames and save them in two folders: `vout1` (colored frames) and `grayvout1` (grayscale frames).
- Entropy values will be calculated for each frame and plotted. You can also plot the histogram difference-based frame selection.
``` bash
python main.py
```
## Project Structure

``` bash
project-folder/
│
├── Sample.mkv             # Sample video file (place your video file here)
├── vout1/                 # Directory to store colored frames
├── grayvout1/             # Directory to store grayscale frames
├── main.py                # Main Python script
├── README.md              # Project description file
```
## Plotting Frame Entropies
The entropy of each selected frame is calculated and plotted to visualize changes over time.

``` bash
# Plotting the entropy values for selected frames
plt.plot(x, listed, color='r', label='Entropy')
plt.legend()
plt.show()
```

## Reducing Frames
To reduce the number of frames analyzed, only the frames with the most significant histogram differences are selected. This helps in analyzing key frames, such as those with scene changes.

``` bash
selected_frames = select_frames(frame_changes, num_frames=100)
```
