class NotifyAgent:
    @staticmethod
    async def send(user: str, response: str):
        # Replace with SendGrid/Twilio API
        print(f"[Notify] Sent to {user}: {response}")
