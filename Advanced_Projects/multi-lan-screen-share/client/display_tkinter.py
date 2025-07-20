import tkinter as tk
from PIL import Image, ImageTk
import io

def display_loop(frame_queue, headless=False):
    if frame_queue is None:
        raise ValueError("frame_queue cannot be None")
    if not hasattr(frame_queue, 'pop'):
        raise TypeError("frame_queue must support pop() method")
    # …rest of the function…
    root = tk.Tk()
    root.title('Screen Share - Tkinter')
    label = tk.Label(root)
    label.pack()

    def update():
        if frame_queue:
            frame_data = frame_queue.pop(0)
            try:
                img = Image.open(io.BytesIO(frame_data))
                tk_img = ImageTk.PhotoImage(img)
                label.config(image=tk_img)
                label.image = tk_img
            except Exception as e:
                print(f"[Tkinter] Frame error: {e}")
        if not headless:
            root.after(10, update)

    update()
    if not headless:
        root.mainloop()
    else:
        root.destroy() 