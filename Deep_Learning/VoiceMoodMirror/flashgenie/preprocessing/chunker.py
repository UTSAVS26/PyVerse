"""
Text Chunking Module

Splits text into manageable chunks for NLP processing.
"""

import re
from typing import List, Dict, Any, Optional
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into chunks for better NLP processing."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def chunk_by_sentences(self, text: str, max_sentences: int = 10) -> List[str]:
        """
        Split text into chunks based on sentences.
        
        Args:
            text: Text to chunk
            max_sentences: Maximum sentences per chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed limits, start new chunk
            if (len(current_chunk) >= max_sentences or 
                current_length + sentence_length > self.chunk_size):
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into chunks based on paragraphs.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_length = len(paragraph)
            
            # If adding this paragraph would exceed chunk size, start new chunk
            if current_length + paragraph_length > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # If a single paragraph is too long, split it
                if paragraph_length > self.chunk_size:
                    # Split the long paragraph into smaller chunks
                    for i in range(0, paragraph_length, self.chunk_size):
                        chunk_text = paragraph[i:i + self.chunk_size]
                        chunks.append(chunk_text)
                    continue
            
            current_chunk.append(paragraph)
            current_length += paragraph_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks
    
    def chunk_by_words(self, text: str, max_words: int = 200) -> List[str]:
        """
        Split text into chunks based on word count.
        
        Args:
            text: Text to chunk
            max_words: Maximum words per chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Split by whitespace to preserve punctuation
        words = text.split()
        
        chunks = []
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i + max_words]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
        return chunks
    
    def chunk_with_overlap(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of overlapping text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                search_text = text[search_start:end]
                
                # Find last sentence ending
                sentence_endings = ['.', '!', '?', '\n']
                last_ending = -1
                
                for ending in sentence_endings:
                    pos = search_text.rfind(ending)
                    if pos > last_ending:
                        last_ending = pos
                
                if last_ending > 0:
                    end = search_start + last_ending + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(text):
                break
                
        return chunks
    
    def chunk_by_sections(self, text: str, section_patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Split text into sections based on headers.
        
        Args:
            text: Text to chunk
            section_patterns: Patterns to identify section headers
            
        Returns:
            List of dictionaries with section info and content
        """
        if not text:
            return []
            
        if section_patterns is None:
            section_patterns = [
                r'^Chapter\s+\d+',
                r'^Section\s+\d+',
                r'^\d+\.\s+[A-Z]',
                r'^[A-Z][A-Z\s]+$'
            ]
        
        lines = text.split('\n')
        sections = []
        current_section = {
            'title': 'Introduction',
            'content': '',
            'level': 0
        }
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header
            is_header = False
            header_level = 0
            
            for i, pattern in enumerate(section_patterns):
                if re.match(pattern, line, re.IGNORECASE):
                    is_header = True
                    header_level = i + 1
                    break
            
            if is_header:
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                
                # Start new section
                current_section = {
                    'title': line,
                    'content': '',
                    'level': header_level
                }
            else:
                # Add line to current section content
                if current_section['content']:
                    current_section['content'] += '\n' + line
                else:
                    current_section['content'] = line
        
        # Add the last section
        if current_section['content'].strip():
            sections.append(current_section)
            
        return sections
    
    def get_optimal_chunks(self, text: str, method: str = 'sentences') -> List[str]:
        """
        Get optimally sized chunks using the specified method.
        
        Args:
            text: Text to chunk
            method: Chunking method ('sentences', 'paragraphs', 'words', 'overlap')
            
        Returns:
            List of text chunks
        """
        if method == 'sentences':
            return self.chunk_by_sentences(text)
        elif method == 'paragraphs':
            return self.chunk_by_paragraphs(text)
        elif method == 'words':
            return self.chunk_by_words(text)
        elif method == 'overlap':
            return self.chunk_with_overlap(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
