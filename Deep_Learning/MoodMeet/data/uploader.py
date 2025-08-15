"""
Data Uploader Module for MoodMeet

Handles text input, transcript processing, and data validation.
"""

import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TranscriptEntry:
    """Represents a single entry in a transcript."""
    speaker: str
    text: str
    timestamp: Optional[datetime] = None
    line_number: int = 0


class TranscriptProcessor:
    """Processes and validates transcript data."""
    
    def __init__(self):
        self.speaker_pattern = re.compile(r'^([A-Za-z\s]+):\s*(.+)$')
        self.timestamp_pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]')
    
    def parse_transcript(self, text: str) -> List[TranscriptEntry]:
        """
        Parse transcript text into structured entries.
        
        Args:
            text: Raw transcript text
            
        Returns:
            List of TranscriptEntry objects
        """
        entries = []
        lines = text.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Try to match speaker pattern
            match = self.speaker_pattern.match(line)
            if match:
                speaker = match.group(1).strip()
                text_content = match.group(2).strip()
                
                # Extract timestamp if present
                timestamp = None
                timestamp_match = self.timestamp_pattern.search(line)
                if timestamp_match:
                    try:
                        timestamp_str = timestamp_match.group(1)
                        timestamp = datetime.strptime(timestamp_str, '%H:%M:%S')
                    except ValueError:
                        pass
                
                entry = TranscriptEntry(
                    speaker=speaker,
                    text=text_content,
                    timestamp=timestamp,
                    line_number=i + 1
                )
                entries.append(entry)
            else:
                # If no speaker pattern, treat as continuation of previous entry
                if entries:
                    entries[-1].text += ' ' + line
                else:
                    # Create anonymous speaker entry
                    entry = TranscriptEntry(
                        speaker="Unknown",
                        text=line,
                        line_number=i + 1
                    )
                    entries.append(entry)
        
        return entries
    
    def validate_transcript(self, entries: List[TranscriptEntry]) -> Tuple[bool, List[str]]:
        """
        Validate transcript entries for quality and completeness.
        
        Args:
            entries: List of transcript entries
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not entries:
            errors.append("No transcript entries found")
            return False, errors
        
        # Check for minimum content
        total_text = ' '.join([entry.text for entry in entries])
        if len(total_text.strip()) < 10:
            errors.append("Transcript too short (minimum 10 characters)")
        
        # Check for speaker diversity
        # Check for speaker diversity
        speakers = set([entry.speaker for entry in entries])
        if len(speakers) < 2:
            errors.append("Insufficient speaker diversity (minimum 2 speakers required)")
        
        # Check for empty or very short entries
        for entry in entries:
            if len(entry.text.strip()) < 2:
                errors.append(f"Entry too short at line {entry.line_number}")
        
        return len(errors) == 0, errors
    
    def to_dataframe(self, entries: List[TranscriptEntry]) -> pd.DataFrame:
        """
        Convert transcript entries to pandas DataFrame.
        
        Args:
            entries: List of transcript entries
            
        Returns:
            DataFrame with transcript data
        """
        data = []
        for entry in entries:
            data.append({
                'speaker': entry.speaker,
                'text': entry.text,
                'timestamp': entry.timestamp,
                'line_number': entry.line_number,
                'text_length': len(entry.text)
            })
        
        return pd.DataFrame(data)
    
    def get_speaker_stats(self, entries: List[TranscriptEntry]) -> Dict[str, Dict]:
        """
        Get statistics for each speaker.
        
        Args:
            entries: List of transcript entries
            
        Returns:
            Dictionary with speaker statistics
        """
        speaker_stats = {}
        
        for entry in entries:
            if entry.speaker not in speaker_stats:
                speaker_stats[entry.speaker] = {
                    'message_count': 0,
                    'total_words': 0,
                    'total_chars': 0,
                    'avg_message_length': 0
                }
            
            speaker_stats[entry.speaker]['message_count'] += 1
            speaker_stats[entry.speaker]['total_words'] += len(entry.text.split())
            speaker_stats[entry.speaker]['total_chars'] += len(entry.text)
        
        # Calculate averages
        for speaker in speaker_stats:
            count = speaker_stats[speaker]['message_count']
            if count > 0:
                speaker_stats[speaker]['avg_message_length'] = (
                    speaker_stats[speaker]['total_chars'] / count
                )
        
        return speaker_stats


class TextInputHandler:
    """Handles various text input formats."""
    
    def __init__(self):
        self.processor = TranscriptProcessor()
    
    def process_text_input(self, text: str) -> Tuple[pd.DataFrame, Dict, bool, List[str]]:
        """
        Process text input and return structured data.
        
        Args:
            text: Raw text input
            
        Returns:
            Tuple of (dataframe, speaker_stats, is_valid, errors)
        """
        # Parse transcript
        entries = self.processor.parse_transcript(text)
        
        # Validate
        is_valid, errors = self.processor.validate_transcript(entries)
        
        # Convert to DataFrame
        df = self.processor.to_dataframe(entries)
        
        # Get speaker statistics
        speaker_stats = self.processor.get_speaker_stats(entries)
        
        return df, speaker_stats, is_valid, errors
    
    def process_file_upload(self, file_content: str, filename: str) -> Tuple[pd.DataFrame, Dict, bool, List[str]]:
        """
        Process uploaded file content.
        
        Args:
            file_content: File content as string
            filename: Name of uploaded file
            
        Returns:
            Tuple of (dataframe, speaker_stats, is_valid, errors)
        """
        return self.process_text_input(file_content)


# Example usage and testing
if __name__ == "__main__":
    # Test with sample transcript
    sample_text = """
    Alice: We're falling behind schedule.
    Bob: Let's regroup and finish the draft today.
    Carol: Honestly, I'm feeling a bit burned out.
    David: I think we can make it work if we focus.
    Alice: That sounds like a good plan.
    """
    
    handler = TextInputHandler()
    df, stats, valid, errors = handler.process_text_input(sample_text)
    
    print("DataFrame:")
    print(df)
    print("\nSpeaker Stats:")
    print(stats)
    print(f"\nValid: {valid}")
    print(f"Errors: {errors}") 