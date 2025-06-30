# GPU-Accelerated Large-Scale Sorting with CuPy

This project implements a high-performance sorting pipeline for very large arrays (e.g., 100 million integers) **entirely on the GPU** using Python and CuPy. It leverages:

- **CuPy’s built-in GPU sorting** for fast batch sorting.
- A **custom CUDA merge kernel** (via `cupy.RawKernel`) for efficient GPU-based merging of sorted batches.
- Recursive merging of multiple batches to scale sorting beyond GPU memory limits.

---

## Features

- **Full GPU pipeline:** sorting and merging performed entirely on GPU, minimizing data transfer overhead.
- **Custom CUDA kernel:** for merging two sorted arrays in parallel on GPU.
- **Scalable:** processes huge arrays by splitting into manageable batches and recursively merging.
- **Benchmarked on 100 million integers** with multi-second execution times.
- **Pure Python implementation** leveraging CuPy and CUDA.

---

## Requirements

- Python 3.7+
- [CuPy](https://docs.cupy.dev/en/stable/install.html) matching your CUDA version (e.g. `cupy-cuda11x`)
- NVIDIA GPU with CUDA support

Install CuPy via pip:

```bash
pip install cupy-cuda11x
```

Replace `11x` with your CUDA version, e.g., `112`, `113`, `120`.

---

## Usage

Run the benchmark script:

```bash
python bitonic_sort_gpu.py
```

This will:

1. Generate 100 million random integers on the CPU.
2. Sort them in batches on the GPU.
3. Recursively merge sorted batches using the custom CUDA kernel.
4. Copy the fully sorted array back to CPU memory.
5. Output timing and correctness validation.

---

## How It Works

- The array is split into batches (default 50 million elements).
- Each batch is sorted using CuPy’s `cp.sort()` on GPU.
- A **custom CUDA kernel** merges two sorted arrays on GPU.
- Recursive pairwise merging reduces batches to a single fully sorted array.
- Minimal CPU-GPU memory transfers maximize throughput.

---

## Performance

On a modern NVIDIA GPU, sorting 100 million integers completes in approximately 4–6 seconds.

---

## File Overview

- `bitonic_sort_gpu.py` — main script containing:
  - Raw CUDA merge kernel
  - Batch sorting and recursive merging logic
  - Benchmark function

---

## Notes

- Requires sufficient GPU memory to hold batches and temporary merge buffers.
- Adjust batch size for your hardware.
- Can be extended to sort other numeric data types.

## Author

Your Name — @SK8-infi

---
