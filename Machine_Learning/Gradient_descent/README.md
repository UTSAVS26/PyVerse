# GRADIENT DESCENT

There are mainly three types of gradient descent:

1. **Batch Gradient Descent (BGD)**
2. **Stochastic Gradient Descent (SGD)**
3. **Mini-Batch Gradient Descent**

### 1. Batch Gradient Descent (BGD):
Batch Gradient Descent computes the gradient of the cost function with respect to the parameters for the entire training dataset. This means that every update of the model's parameters is based on the complete dataset.

**Characteristics**:
- **Update Rule**: The weights and bias are updated after processing the entire dataset.
- **Speed**: It can be slow when dealing with large datasets, as it requires computing the gradient for every data point before updating.
- **Convergence**: The updates are more stable, leading to smoother convergence towards the global minimum. However, it may be slow for very large datasets.
- **Memory Usage**: Requires a lot of memory because it needs to load the entire dataset into memory to compute the gradients.

**Pros**:
- Provides a more stable and accurate estimate of the gradient.
- Generally converges in a smoother fashion compared to other methods.

**Cons**:
- Slow when dealing with large datasets, as every iteration requires going through all the data points.
- High memory usage.

---

### 2. Stochastic Gradient Descent (SGD):
In Stochastic Gradient Descent, the model's parameters are updated for each training example, rather than for the entire dataset. This means it computes the gradient and updates the weights for each sample one at a time.

**Characteristics**:
- **Update Rule**: The weights and bias are updated after processing each data point (sample).
- **Speed**: Faster than Batch Gradient Descent because updates are made more frequently, but less stable due to the noisy nature of single-sample updates.
- **Convergence**: Due to the frequent updates, the convergence path is much noisier, which can help the algorithm escape local minima but may also cause erratic behavior.
- **Memory Usage**: Requires less memory because it only needs to store one data point at a time.

**Pros**:
- Faster updates and convergence for large datasets since it updates with every data point.
- Can escape local minima due to the stochastic nature of updates.

**Cons**:
- The updates can be noisy, leading to a jagged convergence path.
- May overshoot the minimum and converge less accurately.

---

### 3. Mini-Batch Gradient Descent:
Mini-Batch Gradient Descent strikes a balance between Batch and Stochastic Gradient Descent. It divides the training dataset into small batches (mini-batches) and updates the model’s parameters for each batch.

**Characteristics**:
- **Update Rule**: The weights and bias are updated after processing a mini-batch (a small group of data points).
- **Speed**: Faster than Batch Gradient Descent and more stable than Stochastic Gradient Descent. It reduces the variance of parameter updates and can make more efficient use of the computational resources.
- **Convergence**: It provides a smoother convergence than SGD and is more efficient for large datasets.
- **Memory Usage**: Uses less memory than Batch Gradient Descent but more than SGD, as it needs to store a mini-batch at a time.

**Pros**:
- Combines the benefits of both Batch and Stochastic Gradient Descent.
- Faster convergence than Batch Gradient Descent, and smoother updates compared to Stochastic Gradient Descent.
- Can be parallelized to take advantage of modern hardware.

**Cons**:
- Still requires choosing an appropriate mini-batch size, which can impact performance.
- Not as fast as SGD in terms of frequent updates but more stable in convergence.

---

### Summary:
- **Batch Gradient Descent**: Uses the entire dataset for each update; accurate but slow and memory-intensive.
- **Stochastic Gradient Descent**: Updates for each individual data point; fast but with high variability in updates.
- **Mini-Batch Gradient Descent**: Updates for small batches of data; offers a balance between the speed of SGD and the stability of BGD.
