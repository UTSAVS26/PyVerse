
import numpy as np
from numba import cuda

@cuda.jit
def xor_encrypt_kernel(frame, key, out):
    i = cuda.grid(1)
    if i < frame.size:
        out[i] = frame[i] ^ key[i % len(key)]

def gpu_encrypt_frame(frame, key):
    frame = frame.flatten()
    d_frame = cuda.to_device(frame)
    d_key = cuda.to_device(np.frombuffer(key, dtype=np.uint8))
    d_out = cuda.device_array_like(d_frame)
    threads = 256
    blocks = (len(frame) + (threads - 1)) // threads
    xor_encrypt_kernel[blocks, threads](d_frame, d_key, d_out)
    return d_out.copy_to_host().reshape(-1)
