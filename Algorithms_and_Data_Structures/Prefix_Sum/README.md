# 📌 Prefix Sum Algorithm

This script demonstrates the use of a **Prefix Sum Array** to efficiently compute the sum of elements in a given range of an array in constant time (O(1)) after linear time preprocessing (O(n)).

---

## 📋 What is Prefix Sum?

The **Prefix Sum** of an array is a second array where each element at index `i` is the sum of all elements from the original array from index `0` to `i-1`.

This allows for **fast range sum queries**:  
Sum from index `l` to `r` = `prefix[r+1] - prefix[l]`

---

## 🧠 Use Case

This is commonly used in:
- Range sum problems
- Competitive programming
- Optimization of nested loops

---

## 🛠️ How It Works

1. A prefix sum array is constructed from the original array.
2. To get the sum of a subarray `[l, r]`, subtract `prefix[l]` from `prefix[r + 1]`.

---

## 🧪 Example Run

```python
Input Array: [10, 2, 3, 4, 14, 15, 2, 22]
Prefix Sum Array: [0, 10, 12, 15, 19, 33, 48, 50, 72]
Sum from index 2 to 5 is: 36
