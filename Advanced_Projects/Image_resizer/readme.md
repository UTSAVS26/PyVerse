# 🖼️ Image Resizer (GUI)

A **lightweight Python desktop app** to resize images (JPG/PNG/BMP/GIF) with ease.
Built using **Tkinter** for GUI and **Pillow** for image processing.

---

## ✨ Features

- 📂 Select one or more images.
- 📏 Resize by **custom width & height** (in pixels).
- ➗ Resize by **percentage scale** (e.g., 50%).
- 💾 Save resized images in an **output folder** with `_resized` suffix.
- ⚡ Lightweight, dependency-minimal, and beginner-friendly.
- 🌑 Modern dark-themed interface.

---

## 📂 Project Structure

```
image_resizer/
│── app.py           # Full application (GUI + logic)
└── output/          # Resized images are saved here
```

---

## 🛠️ Requirements

- Python 3.8+
- [Pillow](https://pypi.org/project/pillow/) (PIL fork, for image processing)

Install dependency:

```bash
pip install pillow
```

---

## 📖 Usage

1. Click **“📂 Select Images”** → choose one or more images.
2. Enter either:

   - **Width + Height** (pixels), or
   - **Scale (%)** (e.g., 50 = half size).

3. (Optional) Choose a custom output folder.
4. Click **“⚡ Resize”** → resized images will be saved with `_resized` suffix.

---

## 🖼️ Supported Formats

- JPG
- PNG
- BMP
- GIF

---
