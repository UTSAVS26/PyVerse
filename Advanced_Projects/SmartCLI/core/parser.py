import os
import re
from typing import Dict, Any

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    import openai
except ImportError:
    openai = None

class QueryParser:
    """NLP query parser for SmartCLI."""
    ACTIONS = ['find', 'delete', 'compress', 'list', 'move', 'copy', 'rename']

    def __init__(self, model=None, use_openai=False):
        self.use_openai = use_openai and openai is not None and os.getenv('OPENAI_API_KEY')
        self.model = model
        if not self.use_openai and pipeline is not None:
            self.nlp = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        else:
            self.nlp = None
        self.labels = self.ACTIONS + [
            'file', 'folder', 'image', 'pdf', 'zip', 'older', 'newer', 'large', 'small',
            'Downloads', 'Desktop', 'Documents', 'week', 'month', 'day', 'today', 'yesterday'
        ]

    def extract_entities(self, query: str) -> Dict[str, Any]:
        entities = {}
        # File type (case-insensitive, always lower)
        file_types = re.findall(r'pdf|png|jpg|zip|image|file|folder', query, re.I)
        if file_types:
            entities['file_type'] = [ftype.lower() for ftype in file_types]
        # Location
        locations = re.findall(r'Downloads|Desktop|Documents', query, re.I)
        if locations:
            entities['location'] = [loc.capitalize() for loc in locations]
        # Time (robust, case-insensitive)
        time_patterns = [
            r'last\s+week', r'last\s+month', r'today', r'yesterday',
            r'older than \d+\s*(day|week|month)s?', r'older than \d+\s*weeks?', r'older than \d+\s*months?'
        ]
        time_matches = []
        for pat in time_patterns:
            match = re.findall(pat, query, re.I)
            if match:
                time_matches.extend(match if isinstance(match, list) else [match])
        if time_matches:
            entities['time'] = [str(t).strip().lower() for t in time_matches]
        # Size
        sizes = re.findall(r'large|small', query, re.I)
        if sizes:
            entities['size'] = [s.lower() for s in sizes]
        return entities

    def extract_intent_regex(self, query: str) -> str:
        for act in self.ACTIONS:
            if act in query.lower():
                return act
        return None

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query and return a structured command intent."""
        intent = None
        entities = self.extract_entities(query)
        # Use OpenAI API if enabled
        if self.use_openai:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            prompt = f"Extract the main action (find, delete, compress, list, move, copy, rename), file type, location, and any time or size filters from this query: '{query}'. Respond as JSON."
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150
                )
                import json
                result = json.loads(response.choices[0].message.content)
                action = result.get('action')
                if action in self.ACTIONS:
                    intent = action
                # Merge OpenAI entities with regex entities
                for k, v in result.items():
                    if k != 'action':
                        entities.setdefault(k, v)
            except Exception as e:
                entities['error'] = str(e)
        # Use local transformer zero-shot classification
        elif self.nlp is not None:
            result = self.nlp(query, self.labels)
            top_label = result['labels'][0]
            if top_label in self.ACTIONS:
                intent = top_label
            else:
                # Fallback to regex/keyword extraction for intent
                intent = self.extract_intent_regex(query)
            entities['labels'] = result['labels'][:3]
            regex_entities = self.extract_entities(query)
            for k, v in regex_entities.items():
                entities.setdefault(k, v)
        else:
            intent = self.extract_intent_regex(query)
        return {"intent": intent, "entities": entities, "raw": query} 