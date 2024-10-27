## **Web Application Firewall**

### 🎯 **Goal**

The goal of this code is to create a Flask-based web application designed to process user input securely by preventing common web security threats.

### 🧵 **Dataset**

N/A

### 🧾 **Description**

This Python code is a Flask-based web application designed to process user input securely by preventing common web security threats like SQL Injection and Cross-Site Scripting (XSS). It uses pattern matching with regular expressions to detect these threats in the input. Additionally, it has a rate-limiting feature, limiting the number of requests from a single user to prevent abuse (DoS attacks).

### 🧮 **What I had done!**

1. Threat Detection: Implemented pattern matching using regular expressions to detect potential SQL injection and XSS attacks in user input.
2. Input Sanitization: By using the html.escape() function, ensured that any harmful HTML characters are escaped, preventing malicious scripts from being executed in the browser.
3. Rate Limiting: Integrated Flask-Limiter to impose rate limits (e.g., 10 requests per minute) on incoming requests, protecting the app from abuse such as denial of service (DoS) attacks.
4. Logging: You set up logging to record any detected security events, such as SQL injection attempts or XSS attacks. This helps with monitoring and auditing the security of the application.

### 🚀 **Models Implemented**

N/A

### 📚 **Libraries Needed**

1. `Flask` : It is a lightweight web framework used to handle HTTP requests and responses, providing routes like /submit for user interaction.
2. `Flask-Limiter` : Provides rate-limiting to protect the application from abuse, such as a Denial of Service (DoS) attack by limiting the number of requests a user can make.

### 📊 **Exploratory Data Analysis Results**

N/A. 

### 📈 **Performance of the Models based on the Accuracy Scores**

N/A. 

### 📢 **Conclusion**

This Flask-based web application effectively implements basic cybersecurity protections by detecting and preventing common threats like SQL Injection and Cross-Site Scripting (XSS) through pattern matching and input sanitization. The addition of rate-limiting safeguards the application from excessive requests, helping to mitigate DoS attacks. The app also incorporates robust logging to track security events, making it a simple yet effective solution for securing user input and enhancing web application security.

**Deanne Vaz**  
[GitHub](https://github.com/djv554) | | [LinkedIn](https://www.linkedin.com/in/deanne-vaz/)
