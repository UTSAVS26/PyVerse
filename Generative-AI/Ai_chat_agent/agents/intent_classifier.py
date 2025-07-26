INTENT_KEYWORDS = {
    "faq": ["how", "what", "where", "when"],
    "complaint": ["not working", "issue", "problem", "complaint"],
    "account": ["login", "password", "account", "profile"]
}

class IntentClassifierAgent:
    @staticmethod
    def classify(query: str) -> str:
        query_lower = query.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        return "unknown"
