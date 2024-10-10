# **EdgeLock-X**

### 🎯 **Goal**

The primary goal of **EdgeLock-X** is to address the security challenges of deploying machine learning models for face authentication in a browser context. The project ensures that the ML model is safeguarded from reverse engineering and tampering while maintaining efficiency in size and performance to deliver a smooth user experience.

### 🧵 **Dataset**

**EdgeLock-X** does not rely on a pre-existing dataset. Instead, it processes real-time facial data captured via the user's webcam. The system uses encrypted data to authenticate users while protecting the integrity of the model and data.

### 🧾 **Description**

**EdgeLock-X** provides a proof of concept for face authentication in a browser using **TensorFlow.js** and **FaceMesh**. The application securely handles facial data with AES-256 encryption and integrates **Trusted Execution Environment (TEE)** for protecting the ML model. Obfuscation techniques and snapshotting further optimize the model's size without compromising security.

### 🧮 **What I Had Done!**

- Implemented a **face detection feature** using **TensorFlow.js** and **FaceMesh** to capture real-time facial data from the user's webcam.
- Integrated **AES-256 encryption** to ensure secure data transmission and storage.
- Developed a **TEE-based model handling system**, leveraging **obfuscation** and **snapshotting** to protect the model from reverse engineering.
- Set up **real-time face authentication**, providing feedback to users via the web interface.

### 🚀 **Models Implemented**

- **FaceMesh (TensorFlow.js)**: Used for real-time face detection in the browser.
- **AES-256 Encryption**: Protects data both in transit and during authentication.
- **TEE Integration**: Ensures secure decryption and execution of the ML model.

### 📚 **Technologies, Libraries, and Tools**

- **Frontend**:
  - HTML
  - JavaScript
  - TensorFlow.js
  - FaceMesh

- **Backend**:
  - Flask
  - MongoDB (optional for data storage)

- **Security**:
  - AES-256 Encryption
  - Trusted Execution Environment (TEE)
  - Obfuscation
  - Snapshotting

### 📊 **Exploratory Data Analysis Results**

Since **EdgeLock-X** deals with real-time input from users, traditional data analysis is not performed. However, the system provides real-time insights based on successful or failed face authentication attempts, ensuring the process is secure and efficient.

### 📈 **Performance Metrics**

The performance of the system is assessed through:
- **Authentication accuracy**: How reliably the system detects and authenticates faces.
- **Model security**: How effectively the obfuscation and TEE protect the model.
- **User experience**: Minimal latency and load times, with seamless face detection and authentication.

### 💻 **How to Run**

To get started with **EdgeLock-X**, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Install Python dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure environment variables**:

    Rename `.env-sample` to `.env` and add the required keys and settings.

4. **Start the TEE server**:

    ```bash
    python tee_server.py
    ```

5. **Start the web server**:

    ```bash
    python web_server.py
    ```

6. **Access the application**:

    Open a web browser and navigate to `http://localhost:5000`. You will see the registration and login pages.

7. **Test the application**:
    - **Register**: Enter a username and click "Register" to save the face image.
    - **Login**: Enter the username and click "Login" to authenticate with the captured face image. Successful login will redirect to the home page.

### 📢 **Conclusion**

**EdgeLock-X** is an innovative tool for implementing face authentication in a browser while safeguarding the underlying ML model. The system combines security techniques like TEE, encryption, and obfuscation to prevent reverse engineering and tampering, ensuring a secure and user-friendly authentication experience.

### ✒️ **Signature**

**[J B Mugundh]**  
GitHub: [Github](https://github.com/J-B-Mugundh)  
LinkedIn: [LinkedIn](https://www.linkedin.com/in/mugundhjb/)
