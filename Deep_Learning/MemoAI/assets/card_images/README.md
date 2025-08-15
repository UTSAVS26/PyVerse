# Card Images Directory

This directory contains card face images for the memory game.

## Structure

- `README.md` - This file
- Card images will be added here for different themes

## Usage

The game currently uses emoji-based card values, but you can extend it to use custom images by:

1. Adding image files to this directory
2. Modifying the `_generate_card_values()` method in `game/memory_game.py`
3. Updating the UI to display images instead of emojis

## Supported Formats

- PNG
- JPG/JPEG
- SVG (for scalable graphics)

## Naming Convention

Use descriptive names like:
- `animal_dog.png`
- `animal_cat.png`
- `fruit_apple.png`
- `fruit_banana.png` 