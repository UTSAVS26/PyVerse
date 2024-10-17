# Visualizing Tools

# 🎯 Goal

The primary objective of this project is to explain the implementation, visualization, and analysis of various queue data structures, including Dequeue, Linear Queue, Priority Queue, and Queue in Two Stacks. The project focuses on providing an interactive experience where users can input data and observe the behavior of these data structures through visual representation.

# 🧵 Dataset / Input

This project does not rely on a predefined dataset. Instead, it takes user input in the following formats:
1. **File Upload:** Users can upload a file containing the input data. Ensure that the input file format is described below.
2. **Manual Entry:** Users can manually input data.

### File Format Requirements:
- **File1:** Accepts input in CSV format, containing fields such as `operation`, `value`, and so on.
- **File2:** Requires JSON format, with key fields such as `operation` and `priority`.
- **File3:** Describes the operations in the format `<operation>: <value>`.
- **File4:** Takes file types or formats and expects the following fields: `operation_type`, `data`.

**Example Input:**
- **Manual Input Format Example:**
    ```plaintext
    operation: enqueue
    value: 10
    ```
  
- **File Input Example:**
    ```csv
    operation,value
    enqueue,10
    dequeue,
    enqueue,20
    ```

# 🧾 Description
This project provides a web-based platform developed using Streamlit that allows users to interact with the system through manual input or file uploads. The tool implements specific functionalities for Dequeue, Linear Queue, Priority Queue, and Queue in Two Stacks, enabling users to visualize operations in real-time.

### Features:
- **Real-time visualization** of queue operations based on user input.
- **Multiple input modes:** manual and file-based.
- **Dynamic operations** that are applied step-by-step to the input data, allowing users to see how elements are added or removed from the queues.

# 🧮 What I Had Done!
1. Developed an interactive web interface using Streamlit.
2. Implemented features for file uploading and manual input handling for various queue operations.
3. Visualized the queue operations (enqueue, dequeue, etc.) in real-time.
4. Provided feedback on every step of the process, illustrating how each queue type behaves with different operations.
5. Output examples based on the input data.

### Sample Output (based on the project):
- After processing the input, the system generates visualizations or outputs relevant to the queue operations. For example:
    - **Queue Visualization Output:**
    ```plaintext
    [10] -> [20] (for a linear queue)
    ```
    - **Processed Data Output:**
    ```plaintext
    operation: dequeue
    result: 10
    ```

# 📚 Libraries Needed
To run this project, install the following libraries:
- **streamlit:** for building the web interface.
- **matplotlib** (or other visualization libraries as per your needs).
- **Any other dependencies needed by your project.**

Install them using:
```bash
pip install -r requirements.txt
```

### Requirements File (requirements.txt):
```plaintext
streamlit
matplotlib
```

# 📊 Exploratory Data Analysis Results
No EDA is required as the project is based on user inputs. However, real-time graphical outputs or visual representations of the queue operations provide insights into how each queue type operates based on user inputs.

### 📈 Performance of the Algorithms
Since the project focuses on various queue algorithms, no accuracy metrics are required. Performance can be verified based on the correct execution of:
- Enqueue and dequeue operations for each queue type.
- Visualization of the current state of each queue after operations are performed.

# 📢 Conclusion
The project effectively demonstrates how to process user inputs and visualize queue operations in real-time. The system allows for dynamic interaction through both manual entry and file upload, offering flexibility in how users interact with different queue data structures.

# ✒️ Your Signature
Benak Deepak
