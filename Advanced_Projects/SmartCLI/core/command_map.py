import os
from core.utils import get_system_path

class CommandMapper:
    """Maps parsed intents to shell or Python commands."""
    ACTIONS = ['find', 'delete', 'compress', 'list', 'move', 'copy', 'rename']

    def __init__(self):
        pass

    def map_intent(self, parsed_query: dict) -> str:
        """Map a parsed query to a shell command string."""
        intent = parsed_query.get('intent')
        entities = parsed_query.get('entities', {})
        if intent is None or intent not in self.ACTIONS:
            return "echo 'Command mapping not implemented.'"
        location = None
        if 'location' in entities:
            location = get_system_path(entities['location'][0]) if isinstance(entities['location'], list) else get_system_path(entities['location'])
        else:
            location = os.path.expanduser('~')
        file_type = None
        if 'file_type' in entities:
            file_type = entities['file_type'][0].lower() if isinstance(entities['file_type'], list) else entities['file_type'].lower()
        elif 'labels' in entities:
            for label in entities['labels']:
                if label in ['pdf', 'image', 'zip', 'file', 'folder', 'jpg']:
                    file_type = label.lower()
                    break
        ext_map = {'pdf': '*.pdf', 'image': '*.png *.jpg *.jpeg *.gif', 'zip': '*.zip', 'jpg': '*.jpg', 'file': '*', 'folder': '*'}
        file_pattern = ext_map.get(file_type, '*')
        mtime = ''
        if 'time' in entities:
            time_str = entities['time'][0] if isinstance(entities['time'], list) else entities['time']
            if 'last week' in time_str:
                mtime = '-mtime -7'
            elif 'last month' in time_str:
                mtime = '-mtime -30'
            elif 'today' in time_str:
                mtime = '-mtime 0'
            elif 'yesterday' in time_str:
                mtime = '-mtime 1'
            elif 'older than' in time_str:
import os
import re
from core.utils import get_system_path
                match = re.search(r'older than (\d+) (day|week|month)s?', time_str)
                if match:
                    num, unit = match.groups()
                    days = int(num) * (7 if unit.startswith('week') else 30 if unit.startswith('month') else 1)
                    mtime = f'-mtime +{days}'
            elif 'older' in time_str:
                mtime = '-mtime +14'  # fallback for 'older than 2 weeks', etc.
        if intent == 'find' or intent == 'list':
            cmd = f"find {location} -name '{file_pattern}' {mtime}".strip()
        elif intent == 'delete':
            cmd = f"find {location} -name '{file_pattern}' {mtime} -delete".strip()
        elif intent == 'compress':
            out_name = "archive.zip"
            cmd = f"cd {location} && zip -r {out_name} {file_pattern}"
        else:
            cmd = "echo 'Command mapping not implemented.'"
        return cmd 