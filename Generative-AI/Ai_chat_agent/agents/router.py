class RoutingAgent:
    @staticmethod
    async def route(intent: str, user: str, message: str):
        if intent == "faq":
            return await support_faq.FAQAgent.handle(user, message)
        elif intent == "complaint":
            return await support_ticket.TicketAgent.handle(user, message)
        elif intent == "account":
            return await support_account.AccountAgent.handle(user, message)
        else:
            return "Sorry, I couldn't understand your request."
