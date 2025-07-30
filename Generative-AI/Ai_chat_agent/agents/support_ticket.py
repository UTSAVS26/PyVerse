from db.database import SessionLocal
from db.models import Ticket

class TicketAgent:
    @staticmethod
    async def handle(user: str, message: str) -> str:
        async with SessionLocal() as session:
            ticket = Ticket(user=user, message=message)
            session.add(ticket)
            await session.commit()
        return f"Ticket has been created for your complaint, {user}."
