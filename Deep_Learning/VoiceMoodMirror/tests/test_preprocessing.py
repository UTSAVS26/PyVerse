"""
Tests for Preprocessing Modules
"""

import pytest
from flashgenie.preprocessing.clean_text import TextCleaner
from flashgenie.preprocessing.chunker import TextChunker


class TestTextCleaner:
    """Test cases for TextCleaner class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cleaner = TextCleaner()
        
        # Sample text for testing
        self.sample_text = """
        This is a sample document with multiple    spaces.
        
        It contains references and footnotes.
        
        1. This is a footnote.
        a) Another footnote.
        * Yet another footnote.
        
        References:
        Smith, J. (2020). Sample paper.
        Jones, A. (2021). Another paper.
        
        Bibliography:
        Brown, B. (2019). Third paper.
        """
    
    def test_init(self):
        """Test TextCleaner initialization."""
        cleaner = TextCleaner(remove_references=True, remove_footnotes=True)
        assert cleaner.remove_references is True
        assert cleaner.remove_footnotes is True
        
        cleaner = TextCleaner(remove_references=False, remove_footnotes=False)
        assert cleaner.remove_references is False
        assert cleaner.remove_footnotes is False
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "This   has   multiple    spaces."
        result = self.cleaner.clean_text(text)
        assert result == "This has multiple spaces."
    
    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        result = self.cleaner.clean_text("")
        assert result == ""
        
        result = self.cleaner.clean_text(None)
        assert result == ""
    
    def test_remove_extra_whitespace(self):
        """Test removal of extra whitespace."""
        text = "This   has   multiple    spaces.\n\n\nAnd multiple newlines."
        result = self.cleaner._remove_extra_whitespace(text)
        assert "   " not in result
        assert "\n\n\n" not in result
    
    def test_remove_special_characters(self):
        """Test removal of special characters."""
        text = "This has @#$% special characters and page numbers 1"
        result = self.cleaner._remove_special_characters(text)
        assert "@#$%" not in result
    
    def test_remove_references(self):
        """Test removal of reference sections."""
        text = "Main content. References: Smith, J. (2020)."
        result = self.cleaner._remove_references(text)
        assert "References:" not in result
        assert "Main content." in result
    
    def test_remove_footnotes(self):
        """Test removal of footnotes."""
        text = "Main content.\n1. This is a footnote.\nMore content."
        result = self.cleaner._remove_footnotes(text)
        assert "1. This is a footnote." not in result
        assert "Main content." in result
        assert "More content." in result
    
    def test_normalize_text(self):
        """Test text normalization."""
        text = "This is a sentence.Another sentence."
        result = self.cleaner._normalize_text(text)
        assert "sentence. Another" in result
    
    def test_clean_sentences(self):
        """Test cleaning a list of sentences."""
        sentences = [
            "This is a good sentence.",
            "Too short.",
            "This is another good sentence with more content."
        ]
        result = self.cleaner.clean_sentences(sentences)
        assert len(result) == 2  # "Too short" should be filtered out
        assert "This is a good sentence." in result
        assert "This is another good sentence with more content." in result
    
    def test_remove_duplicates(self):
        """Test removal of duplicate lines."""
        text = "Line 1\nLine 2\nLine 1\nLine 3"
        result = self.cleaner.remove_duplicates(text)
        lines = result.split('\n')
        assert len(lines) == 3  # Duplicate "Line 1" should be removed
        assert "Line 1" in lines
        assert "Line 2" in lines
        assert "Line 3" in lines
    
    def test_clean_text_full_pipeline(self):
        """Test the complete cleaning pipeline."""
        result = self.cleaner.clean_text(self.sample_text)
        
        # Should remove footnotes
        assert "1. This is a footnote." not in result
        assert "a) Another footnote." not in result
        assert "* Yet another footnote." not in result
        
        # Should remove references
        assert "References:" not in result
        assert "Bibliography:" not in result
        assert "Smith, J. (2020). Sample paper." not in result
        
        # Should normalize whitespace
        assert "   " not in result  # No multiple spaces
        assert "\n\n\n" not in result  # No multiple newlines


class TestTextChunker:
    """Test cases for TextChunker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TextChunker()
        
        # Sample text for testing
        self.sample_text = """
        This is the first sentence. This is the second sentence. 
        This is the third sentence. This is the fourth sentence.
        
        This is a new paragraph with multiple sentences. 
        It contains more content for testing purposes.
        
        Another paragraph here. With more sentences to chunk.
        """
    
    def test_init(self):
        """Test TextChunker initialization."""
        chunker = TextChunker(chunk_size=500, overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 100
    
    def test_chunk_by_sentences(self):
        """Test chunking by sentences."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = self.chunker.chunk_by_sentences(text, max_sentences=2)
        
        assert len(result) == 2
        assert "Sentence one. Sentence two." in result[0]
        assert "Sentence three. Sentence four." in result[1]
    
    def test_chunk_by_sentences_empty(self):
        """Test chunking empty text by sentences."""
        result = self.chunker.chunk_by_sentences("")
        assert result == []
    
    def test_chunk_by_paragraphs(self):
        """Test chunking by paragraphs."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = self.chunker.chunk_by_paragraphs(text)
        
        assert len(result) == 1  # All paragraphs in one chunk due to size limit
        assert "Paragraph one." in result[0]
        assert "Paragraph two." in result[0]
        assert "Paragraph three." in result[0]
    
    def test_chunk_by_words(self):
        """Test chunking by word count."""
        text = "This is a test sentence with multiple words for chunking."
        result = self.chunker.chunk_by_words(text, max_words=3)
        
        assert len(result) == 4  # Should create 4 chunks of 3 words each
        assert "This is a" in result[0]
        assert "test sentence with" in result[1]
        assert "multiple words for" in result[2]
        assert "chunking." in result[3]
    
    def test_chunk_with_overlap(self):
        """Test chunking with overlap."""
        text = "This is a longer text that needs to be chunked with overlap."
        result = self.chunker.chunk_with_overlap(text)
        
        # Should create at least one chunk
        assert len(result) >= 1
        assert all(len(chunk) <= self.chunker.chunk_size for chunk in result)
    
    def test_chunk_by_sections(self):
        """Test chunking by sections."""
        text = """
        Chapter 1: Introduction
        This is the introduction content.
        
        Section 1.1: Background
        This is the background content.
        
        Chapter 2: Methods
        This is the methods content.
        """
        result = self.chunker.chunk_by_sections(text)
        
        assert len(result) >= 1
        for section in result:
            assert 'title' in section
            assert 'content' in section
            assert 'level' in section
    
    def test_get_optimal_chunks_sentences(self):
        """Test getting optimal chunks using sentences method."""
        text = "Sentence one. Sentence two. Sentence three."
        result = self.chunker.get_optimal_chunks(text, method='sentences')
        assert len(result) >= 1
    
    def test_get_optimal_chunks_paragraphs(self):
        """Test getting optimal chunks using paragraphs method."""
        text = "Paragraph one.\n\nParagraph two."
        result = self.chunker.get_optimal_chunks(text, method='paragraphs')
        assert len(result) >= 1
    
    def test_get_optimal_chunks_words(self):
        """Test getting optimal chunks using words method."""
        text = "This is a test sentence."
        result = self.chunker.get_optimal_chunks(text, method='words')
        assert len(result) >= 1
    
    def test_get_optimal_chunks_overlap(self):
        """Test getting optimal chunks using overlap method."""
        text = "This is a test sentence for overlap chunking."
        result = self.chunker.get_optimal_chunks(text, method='overlap')
        assert len(result) >= 1
    
    def test_get_optimal_chunks_invalid_method(self):
        """Test getting optimal chunks with invalid method."""
        with pytest.raises(ValueError):
            self.chunker.get_optimal_chunks("test", method='invalid')
    
    def test_chunk_by_sentences_with_size_limit(self):
        """Test chunking by sentences with size limit."""
        # Create a long sentence
        long_sentence = "This is a very long sentence. " * 50
        result = self.chunker.chunk_by_sentences(long_sentence, max_sentences=10)
        
        # Should respect both sentence count and size limits
        assert len(result) >= 1
        for chunk in result:
            assert len(chunk) <= self.chunker.chunk_size
    
    def test_chunk_by_paragraphs_with_size_limit(self):
        """Test chunking by paragraphs with size limit."""
        # Create long paragraphs
        long_paragraph = "This is a long paragraph. " * 100
        text = f"{long_paragraph}\n\n{long_paragraph}"
        result = self.chunker.chunk_by_paragraphs(text)
        
        # Should respect size limits
        assert len(result) >= 1
        for chunk in result:
            assert len(chunk) <= self.chunker.chunk_size
