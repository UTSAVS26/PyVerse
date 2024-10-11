# **Email Automation System**

This project automates the process of sending various types of emails (e.g., welcome emails, password recovery emails, and notifications) using Python's `smtplib` library. The aim is to streamline email communication, reduce manual effort, and enhance user engagement through timely notifications.

## **Project Aim**
- **Automate Email Communication**: Streamline the process of sending emails automatically based on user actions or schedules.
- **Reduce Manual Effort**: Eliminate repetitive email tasks to save time and resources.
- **Enhance User Engagement**: Send relevant and timely emails to users to maintain interaction and improve the user experience.

## **Features**
- **Welcome Emails**: Automatically sends a personalized welcome email to new users.
- **Password Recovery Emails**: Sends a secure link for users to reset their passwords.
- **Notification Emails**: Allows sending customized notifications to users.
- **Email Validation**: Ensures email addresses are valid before sending.
- **HTML and Text Emails**: Supports sending both plain text and HTML-formatted emails.
  
## **Requirements**
- **Python 3.7+**
- **Required Libraries**:
  - `smtplib`: For sending emails using the SMTP protocol (built-in).
  - `email-validator`: For validating email addresses.

### **Install the required library:**
```bash
pip install email-validator
