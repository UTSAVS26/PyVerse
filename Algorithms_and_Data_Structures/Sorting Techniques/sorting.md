# Sorting Algorithms

Sorting algorithms are essential for organizing data in a specific order, typically ascending or descending. This README provides an overview of common sorting algorithms, their time complexities, and use cases, making it beginner-friendly.

## Table of Contents
- [1. Bubble Sort](#1-bubble-sort)
- [2. Selection Sort](#2-selection-sort)
- [3. Insertion Sort](#3-insertion-sort)
- [4. Merge Sort](#4-merge-sort)
- [5. Quick Sort](#5-quick-sort)
- [6. Heap Sort](#6-heap-sort)
- [7. Radix Sort](#7-radix-sort)
- [8. Bucket Sort](#8-bucket-sort)

## 1. Bubble Sort

### Description
Bubble Sort is the simplest sorting algorithm. It repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. This process is repeated until the list is sorted.

### Time Complexity
- **Best Case**: O(n) — when the list is already sorted.
- **Average Case**: O(n²)
- **Worst Case**: O(n²)

### Use Cases
- Good for small datasets or when teaching the basic concept of sorting algorithms.

## 2. Selection Sort

### Description
Selection Sort divides the input list into two parts: a sorted and an unsorted part. It repeatedly selects the smallest (or largest) element from the unsorted part and moves it to the end of the sorted part.

### Time Complexity
- **Best Case**: O(n²)
- **Average Case**: O(n²)
- **Worst Case**: O(n²)

### Use Cases
- Useful for small datasets where memory writes are costly.

## 3. Insertion Sort

### Description
Insertion Sort builds a sorted array one element at a time. It takes each element and inserts it into its correct position in the already sorted portion of the array.

### Time Complexity
- **Best Case**: O(n) — when the array is already sorted.
- **Average Case**: O(n²)
- **Worst Case**: O(n²)

### Use Cases
- Efficient for small datasets or nearly sorted arrays.

## 4. Merge Sort

### Description
Merge Sort is a divide-and-conquer algorithm that splits the array into halves, sorts each half, and merges them back together. This algorithm is stable, meaning that it maintains the relative order of records with equal keys.

### Time Complexity
- **Best Case**: O(n log n)
- **Average Case**: O(n log n)
- **Worst Case**: O(n log n)

### Use Cases
- Suitable for large datasets and when stable sorting is required.

## 5. Quick Sort

### Description
Quick Sort is a highly efficient sorting algorithm that uses a divide-and-conquer approach. It selects a "pivot" element from the array and partitions the other elements into two sub-arrays according to whether they are less than or greater than the pivot. It then recursively sorts the sub-arrays.

### Time Complexity
- **Best Case**: O(n log n)
- **Average Case**: O(n log n)
- **Worst Case**: O(n²) — occurs when the smallest or largest element is always chosen as the pivot.

### Use Cases
- Often faster than other O(n log n) algorithms due to better cache performance and is suitable for large datasets.

## 6. Heap Sort

### Description
Heap Sort is a comparison-based sorting technique based on a binary heap data structure. It divides the input into a sorted and an unsorted region and uses a heap to find the largest (or smallest) element in the unsorted region efficiently.

### Time Complexity
- **Best Case**: O(n log n)
- **Average Case**: O(n log n)
- **Worst Case**: O(n log n)

### Use Cases
- Useful for large datasets and when memory efficiency is crucial, as it sorts in-place.

## 7. Radix Sort

### Description
Radix Sort is a non-comparison-based sorting algorithm that sorts integers by processing individual digits. It groups numbers by their individual digits, starting from the least significant digit to the most significant digit.

### Time Complexity
- **Best Case**: O(nk) — where k is the number of digits in the maximum number.
- **Average Case**: O(nk)
- **Worst Case**: O(nk)

### Use Cases
- Effective for sorting large sets of integers or strings when the range of values is known.

## 8. Bucket Sort

### Description
Bucket Sort is a distribution-based sorting algorithm. It divides the input array into a finite number of buckets and sorts each bucket individually, either using a different sorting algorithm or recursively applying the bucket sort.

### Time Complexity
- **Best Case**: O(n + k) — where k is the number of buckets.
- **Average Case**: O(n + k)
- **Worst Case**: O(n²) — occurs when all elements are placed into a single bucket and a comparison sort is used.

### Use Cases
- Efficient when the input is uniformly distributed and works well for sorting floating-point numbers.

## Conclusion
Understanding sorting algorithms is crucial for programming and computer science. Each algorithm has its strengths and weaknesses, and knowing which one to use in a specific situation is key to optimizing performance. This guide provides a basic understanding to get you started with sorting techniques.
