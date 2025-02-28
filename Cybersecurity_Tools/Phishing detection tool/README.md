## **Phishing detection tool**

### 🎯 **Goal**

The goal of this code is to create a simple phishing detection tool. 

### 🧵 **Dataset**

N/A

### 🧾 **Description**

This project is a phishing detection tool that analyzes user-provided email information (sender's email address and email body) to detect potential phishing attempts. It works by checking for suspicious patterns like keywords, unusual domains, and unsafe URLs.

### 🧮 **What I had done!**

1. Checking for Phishing Keywords: Implemented a function to identify common phishing phrases (like "verify your account" or "urgent") in the email body.
2. Identifying Suspicious URLs: Implemented URL scanning to flag links with suspicious domains or links using punycode or IP addresses instead of domain names.
3. User Input: Modified the code to take input from users, allowing them to provide the sender's email address and the email content for dynamic analysis.

### 🚀 **Models Implemented**

N/A

### 📚 **Libraries Needed**

1. `re`: This is the regular expressions library, used to search for URLs in the email body.
2. `urllib.parse` : Specifically, the urlparse function is used to parse URLs to check for suspicious domains and patterns.
3. `Levenshtein` : This library is used to calculate the Levenshtein distance between two domain names, which helps identify domains that are similar but slightly altered.

### 📊 **Exploratory Data Analysis Results**

N/A. 

### 📈 **Performance of the Models based on the Accuracy Scores**

N/A. 

### 📢 **Conclusion**

This phishing detection tool provides a straightforward way to analyze email content for phishing indicators by checking for suspicious keywords, untrusted or similar-looking domains, and risky URLs. By allowing users to input email details, it offers a flexible, user-friendly solution for identifying potential phishing attempts. With the integration of the Levenshtein distance for domain validation and regular expressions for URL scanning, it enhances the accuracy of detecting subtle phishing tactics.

**Deanne Vaz**  
[GitHub](https://github.com/djv554) | | [LinkedIn](https://www.linkedin.com/in/deanne-vaz/)
