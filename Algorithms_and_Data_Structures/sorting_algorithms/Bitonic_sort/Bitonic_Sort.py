import cupy as cp
import numpy as np
import time
import math

# -------------------------------
# Raw CUDA GPU Merge Kernel Code
# -------------------------------
merge_kernel_code = r'''
extern "C" __global__
void merge_kernel(const int* A, const int* B, int* C, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m + n;
    if (i >= total) return;

    int low = max(0, i - n);
    int high = min(i, m);
    while (low < high) {
        int mid = (low + high) / 2;
        if (A[mid] <= B[i - mid - 1])
            low = mid + 1;
        else
            high = mid;
    }
    int a_idx = low;
    int b_idx = i - low;

    if (a_idx < m && (b_idx >= n || A[a_idx] <= B[b_idx]))
        C[i] = A[a_idx];
    else
        C[i] = B[b_idx];
}
'''

# Compile the raw CUDA kernel
merge_kernel = cp.RawKernel(merge_kernel_code, 'merge_kernel')

# -------------------------------
# Merge Two Sorted Batches on GPU
# -------------------------------
def merge_sorted_batches_gpu(arr1, arr2):
    if arr1.dtype != cp.int32 or arr2.dtype != cp.int32:
        raise TypeError("merge_kernel expects int32 arrays; got "
                        f"{arr1.dtype} and {arr2.dtype}")
    m, n = arr1.size, arr2.size
    total = m + n
    C = cp.empty(total, dtype=arr1.dtype)
    # … rest of implementation …
    threads_per_block = 256
    blocks = (total + threads_per_block - 1) // threads_per_block

    merge_kernel((blocks,), (threads_per_block,),
                 (arr1, arr2, C, np.int32(m), np.int32(n)))

    cp.cuda.get_current_device().synchronize()
    return C

# -------------------------------
# Recursive Merge of All Batches
# -------------------------------
def recursive_gpu_merge(batches):
    while len(batches) > 1:
        new_batches = []
        for i in range(0, len(batches), 2):
            if i + 1 < len(batches):
                merged = merge_sorted_batches_gpu(batches[i], batches[i + 1])
            else:
                merged = batches[i]
            new_batches.append(merged)
        batches = new_batches
    return batches[0]

# -------------------------------
# Full Pipeline: Sort + Merge
# -------------------------------
def full_gpu_sort_large_array(arr_np, batch_size=50_000_000):
    # Ensure input is int32 for compatibility with merge kernel
    if arr_np.dtype != np.int32:
        arr_np = arr_np.astype(np.int32)
    total_size = arr_np.size
    num_batches = math.ceil(total_size / batch_size)

    sorted_batches_gpu = []
    print(f"🚀 Sorting {total_size} elements in {num_batches} GPU batches...")

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, total_size)
        print(f"🔢 Sorting batch {i+1}/{num_batches} [{start}:{end}]...")
        batch = cp.asarray(arr_np[start:end])
        sorted_batch = cp.sort(batch)
        # …
        sorted_batches_gpu.append(sorted_batch)

    print("🔁 Merging all GPU batches recursively...")
    merged_gpu_array = recursive_gpu_merge(sorted_batches_gpu)

    print("📤 Copying final sorted array back to CPU...")
    return cp.asnumpy(merged_gpu_array)

# -------------------------------
# Benchmark
# -------------------------------
def benchmark_gpu_merge_sort():
    N = 100_000_000  # 100 million
    print(f"🧪 Generating {N} random integers...")
    arr = np.random.randint(0, 1_000_000, size=N).astype(np.int32)

    start = time.time()
    sorted_result = full_gpu_sort_large_array(arr)
    duration = time.time() - start

    print(f"\n✅ Sorted {N} integers on GPU in {duration:.2f} seconds.")
    print("🔍 First 10 elements:", sorted_result[:10])
    print("✅ Is sorted?", np.all(sorted_result[:-1] <= sorted_result[1:]))

if __name__ == "__main__":
    benchmark_gpu_merge_sort()
