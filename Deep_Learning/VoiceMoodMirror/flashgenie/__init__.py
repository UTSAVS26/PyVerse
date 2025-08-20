"""
FlashGenie: AI Flashcard Generator from PDFs

An intelligent flashcard generator that processes PDFs and creates
educational flashcards using NLP techniques.
"""

__version__ = "1.0.0"
__author__ = "FlashGenie Team"
__email__ = "support@flashgenie.ai"

from .pdf_parser.extract_text import PDFTextExtractor
from .preprocessing.clean_text import TextCleaner
from .preprocessing.chunker import TextChunker
from .nlp.keyword_extractor import KeywordExtractor
from .nlp.question_generator import QuestionGenerator
from .nlp.answer_selector import AnswerSelector
from .flashcard.flashcard_formatter import FlashcardFormatter
from .flashcard.export import FlashcardExporter

__all__ = [
    "PDFTextExtractor",
    "TextCleaner", 
    "TextChunker",
    "KeywordExtractor",
    "QuestionGenerator",
    "AnswerSelector",
    "FlashcardFormatter",
    "FlashcardExporter"
]
