import smtplib
from email.message import EmailMessage
from email_validator import validate_email, EmailNotValidError

# Configurations
SMTP_SERVER = 'smtp.gmail.com'  # Use your email provider's SMTP server
SMTP_PORT = 587
SENDER_EMAIL = 'your_email@gmail.com'
SENDER_PASSWORD = 'your_app_password'  # Gmail App Password for security

# Email automation system
class EmailAutomation:
    def __init__(self):
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT
        self.sender_email = SENDER_EMAIL
        self.sender_password = SENDER_PASSWORD

    def validate_email_address(self, email_address):
        try:
            valid = validate_email(email_address)
            return valid.email
        except EmailNotValidError as e:
            print(f"Invalid email address: {e}")
            return None

    def send_email(self, recipient_email, subject, body, is_html=False):
        # Create the email content
        msg = EmailMessage()
        msg['From'] = self.sender_email
        msg['To'] = self.validate_email_address(recipient_email)
        msg['Subject'] = subject

        if is_html:
            msg.add_alternative(body, subtype='html')
        else:
            msg.set_content(body)

        if not msg['To']:
            print("Invalid recipient email. Aborting email send.")
            return

        # Send the email
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
                print(f"Email sent successfully to {recipient_email}")
        except Exception as e:
            print(f"Error sending email: {e}")

    def send_welcome_email(self, recipient_email, username):
        subject = "Welcome to Our Service!"
        body = f"""
        Hi {username},
        Welcome to our platform! We are excited to have you on board.
        If you have any questions, feel free to reach out to us.
        Best regards,
        The Team
        """
        self.send_email(recipient_email, subject, body)

    def send_password_recovery_email(self, recipient_email, reset_link):
        subject = "Password Recovery"
        body = f"""
        Hi,
        We received a request to reset your password. Click the link below to reset it:
        {reset_link}
        If you did not request a password reset, please ignore this email.
        """
        self.send_email(recipient_email, subject, body)

    def send_notification_email(self, recipient_email, message):
        subject = "Notification"
        body = message
        self.send_email(recipient_email, subject, body)


# Example usage
if __name__ == "__main__":
    email_auto = EmailAutomation()

    # Send welcome email
    email_auto.send_welcome_email("recipient@example.com", "John Doe")

    # Send password recovery email
    reset_link = "https://example.com/reset-password?token=abc123"
    email_auto.send_password_recovery_email("recipient@example.com", reset_link)

    # Send a general notification
    email_auto.send_notification_email("recipient@example.com", "This is a test notification.")
