# Tower of Hanoi Game

An interactive Tower of Hanoi game built with Python using `pygame` for the graphical interface and `colorama` for colorful console messages. This classic puzzle game challenges players to move all disks from one rod to another while following specific rules.

## Game Overview

In the Tower of Hanoi puzzle:
- You have three rods and several disks of different sizes.
- The goal is to move all the disks from the first rod to the third rod.
- Disks must be stacked in decreasing order of size on the destination rod.
- You can only move one disk at a time, and a larger disk cannot be placed on a smaller disk.

This game implements these rules and includes visual feedback, vibrant colors, and drag-and-drop controls.

## Features

- **Interactive gameplay**: Click and drag disks between rods to move them.
- **Colorful Console Instructions**: Using `colorama`, the game provides color-coded messages in the console, guiding players and alerting them of invalid moves.
- **Customizable settings**: Adjust the number of disks for different levels of difficulty.

## Setup Instructions

### Requirements

- Python 3.x
- `pygame` library for graphics
- `colorama` library for colored console feedback

### Installation

1. Install dependencies:

    ```bash
    pip install pygame colorama
    ```

2. Run the game:

    ```bash
    python game.py
    ```

## How to Play

1. Launch the game by running `game.py`.
2. Follow the console instructions to understand the game rules:
   - Move one disk at a time.
   - Only place smaller disks on larger ones.
3. Click on a disk to pick it up, then click on another rod to drop it there.
4. Try to move all disks from the first rod to the third rod with the fewest moves.

## Console Guide

The console provides feedback:
- **Instructions**: Game rules and objectives.
- **Invalid Move Alert**: If you try to place a larger disk on a smaller disk, you’ll see an error message in red.

## Customization

You can adjust the following settings in `game.py`:
- **Number of disks**: Change `num_disks` for different difficulty levels.
- **Colors**: Modify `DISK_COLORS` and `BACKGROUND_COLOR` to customize the look and feel.


## Acknowledgments

Thanks to the `pygame` and `colorama` communities for their libraries, which make this interactive Python game possible!
