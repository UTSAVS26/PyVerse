"""
Main FlashGenie Application

Orchestrates the entire flashcard generation process from PDF to flashcards.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .pdf_parser.extract_text import PDFTextExtractor
from .preprocessing.clean_text import TextCleaner
from .preprocessing.chunker import TextChunker
from .nlp.keyword_extractor import KeywordExtractor
from .nlp.question_generator import QuestionGenerator
from .nlp.answer_selector import AnswerSelector
from .flashcard.flashcard_formatter import FlashcardFormatter, QuestionType, Flashcard
from .flashcard.export import FlashcardExporter

logger = logging.getLogger(__name__)


class FlashGenie:
    """Main FlashGenie application class."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize FlashGenie with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.pdf_extractor = PDFTextExtractor(use_pymupdf=self.config.get('use_pymupdf', True))
        self.text_cleaner = TextCleaner(
            remove_references=self.config.get('remove_references', True),
            remove_footnotes=self.config.get('remove_footnotes', True)
        )
        self.text_chunker = TextChunker(
            chunk_size=self.config.get('chunk_size', 1000),
            overlap=self.config.get('chunk_overlap', 200)
        )
        self.keyword_extractor = KeywordExtractor(
            use_spacy=self.config.get('use_spacy', True),
            use_keybert=self.config.get('use_keybert', False)
        )
        self.question_generator = QuestionGenerator(
            use_transformers=self.config.get('use_transformers', True)
        )
        self.answer_selector = AnswerSelector(
            use_spacy=self.config.get('use_spacy', True)
        )
        self.flashcard_formatter = FlashcardFormatter()
        self.flashcard_exporter = FlashcardExporter()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'use_pymupdf': True,
            'remove_references': True,
            'remove_footnotes': True,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'use_spacy': True,
            'use_keybert': False,
            'use_transformers': True,
            'num_questions': 10,
            'num_keywords': 20,
            'max_answer_length': 200,
            'min_quality_score': 0.7
        }
    
    def process_pdf(self, pdf_path: str, output_dir: str = "output", 
                   num_questions: int = None, export_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process a PDF file and generate flashcards.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save output files
            num_questions: Number of questions to generate
            export_formats: List of export formats
            
        Returns:
            Dictionary containing results and metadata
        """
        if export_formats is None:
            export_formats = ['csv', 'anki', 'json', 'txt', 'html']
        
        if num_questions is None:
            num_questions = self.config.get('num_questions', 10)
        
        logger.info(f"Starting PDF processing: {pdf_path}")
        
        try:
            # Step 1: Extract text from PDF
            logger.info("Step 1: Extracting text from PDF...")
            text = self.pdf_extractor.extract_text(pdf_path)
            if not text:
                raise ValueError("No text extracted from PDF")
            
            # Step 2: Clean and preprocess text
            logger.info("Step 2: Cleaning and preprocessing text...")
            cleaned_text = self.text_cleaner.clean_text(text)
            
            # Step 3: Chunk text for processing
            logger.info("Step 3: Chunking text...")
            chunks = self.text_chunker.get_optimal_chunks(cleaned_text, method='sentences')
            
            # Step 4: Extract keywords
            logger.info("Step 4: Extracting keywords...")
            keywords_with_scores = self.keyword_extractor.extract_keywords_combined(
                cleaned_text, top_k=self.config.get('num_keywords', 20)
            )
            keywords = [kw for kw, score in keywords_with_scores]
            
            # Step 5: Generate questions
            logger.info("Step 5: Generating questions...")
            questions = self.question_generator.generate_questions_combined(
                cleaned_text, keywords, num_questions
            )
            
            # Step 6: Find answers for questions
            logger.info("Step 6: Finding answers...")
            answers = self.answer_selector.find_answers_for_questions(
                questions, cleaned_text, self.config.get('max_answer_length', 200)
            )
            
            # Step 7: Create flashcards
            logger.info("Step 7: Creating flashcards...")
            qa_pairs = list(zip(questions, answers))
            flashcards = self.flashcard_formatter.create_flashcards_from_qa_pairs(
                qa_pairs, source_text=cleaned_text, tags=['pdf_generated']
            )
            
            # Step 8: Filter by quality
            logger.info("Step 8: Filtering by quality...")
            quality_flashcards = self.flashcard_formatter.filter_flashcards_by_quality(
                flashcards, self.config.get('min_quality_score', 0.7)
            )
            
            # Step 9: Export flashcards
            logger.info("Step 9: Exporting flashcards...")
            base_filename = Path(pdf_path).stem
            export_results = self.flashcard_exporter.export_multiple_formats(
                quality_flashcards, output_dir, base_filename, export_formats
            )
            
            # Create summary
            summary = self.flashcard_exporter.create_export_summary(quality_flashcards, export_results)
            
            # Save summary
            summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            logger.info(f"Processing completed. Generated {len(quality_flashcards)} flashcards.")
            
            return {
                'success': True,
                'num_flashcards': len(quality_flashcards),
                'num_questions_generated': len(questions),
                'num_keywords_extracted': len(keywords),
                'export_results': export_results,
                'summary': summary,
                'output_dir': output_dir,
                'base_filename': base_filename
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def process_text(self, text: str, output_dir: str = "output", 
                    num_questions: int = None, export_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process raw text and generate flashcards.
        
        Args:
            text: Input text
            output_dir: Directory to save output files
            num_questions: Number of questions to generate
            export_formats: List of export formats
            
        Returns:
            Dictionary containing results and metadata
        """
        if export_formats is None:
            export_formats = ['csv', 'anki', 'json', 'txt', 'html']
        
        if num_questions is None:
            num_questions = self.config.get('num_questions', 10)
        
        logger.info("Starting text processing...")
        
        try:
            # Step 1: Clean and preprocess text
            logger.info("Step 1: Cleaning and preprocessing text...")
            cleaned_text = self.text_cleaner.clean_text(text)
            
            # Step 2: Chunk text for processing
            logger.info("Step 2: Chunking text...")
            chunks = self.text_chunker.get_optimal_chunks(cleaned_text, method='sentences')
            
            # Step 3: Extract keywords
            logger.info("Step 3: Extracting keywords...")
            keywords_with_scores = self.keyword_extractor.extract_keywords_combined(
                cleaned_text, top_k=self.config.get('num_keywords', 20)
            )
            keywords = [kw for kw, score in keywords_with_scores]
            
            # Step 4: Generate questions
            logger.info("Step 4: Generating questions...")
            questions = self.question_generator.generate_questions_combined(
                cleaned_text, keywords, num_questions
            )
            
            # Step 5: Find answers for questions
            logger.info("Step 5: Finding answers...")
            answers = self.answer_selector.find_answers_for_questions(
                questions, cleaned_text, self.config.get('max_answer_length', 200)
            )
            
            # Step 6: Create flashcards
            logger.info("Step 6: Creating flashcards...")
            qa_pairs = list(zip(questions, answers))
            flashcards = self.flashcard_formatter.create_flashcards_from_qa_pairs(
                qa_pairs, source_text=cleaned_text, tags=['text_generated']
            )
            
            # Step 7: Filter by quality
            logger.info("Step 7: Filtering by quality...")
            quality_flashcards = self.flashcard_formatter.filter_flashcards_by_quality(
                flashcards, self.config.get('min_quality_score', 0.7)
            )
            
            # Step 8: Export flashcards
            logger.info("Step 8: Exporting flashcards...")
            base_filename = "text_flashcards"
            export_results = self.flashcard_exporter.export_multiple_formats(
                quality_flashcards, output_dir, base_filename, export_formats
            )
            
            # Create summary
            summary = self.flashcard_exporter.create_export_summary(quality_flashcards, export_results)
            
            # Save summary
            summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            logger.info(f"Processing completed. Generated {len(quality_flashcards)} flashcards.")
            
            return {
                'success': True,
                'num_flashcards': len(quality_flashcards),
                'num_questions_generated': len(questions),
                'num_keywords_extracted': len(keywords),
                'export_results': export_results,
                'summary': summary,
                'output_dir': output_dir,
                'base_filename': base_filename
            }
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_processing_stats(self, result: Dict[str, Any]) -> str:
        """
        Get processing statistics as a formatted string.
        
        Args:
            result: Result dictionary from process_pdf or process_text
            
        Returns:
            Formatted statistics string
        """
        if not result.get('success', False):
            return f"Processing failed: {result.get('error', 'Unknown error')}"
        
        stats = f"""
Processing Statistics:
====================
Flashcards generated: {result.get('num_flashcards', 0)}
Questions generated: {result.get('num_questions_generated', 0)}
Keywords extracted: {result.get('num_keywords_extracted', 0)}
Output directory: {result.get('output_dir', 'N/A')}
Base filename: {result.get('base_filename', 'N/A')}

Export Results:
"""
        
        export_results = result.get('export_results', {})
        for format_type, success in export_results.items():
            status = "✓ Success" if success else "✗ Failed"
            stats += f"  {format_type.upper()}: {status}\n"
        
        return stats


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FlashGenie: AI Flashcard Generator")
    parser.add_argument("input", help="Input PDF file or text file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--questions", "-q", type=int, default=10, help="Number of questions to generate")
    parser.add_argument("--formats", "-f", nargs="+", 
                       default=['csv', 'anki', 'json', 'txt', 'html'],
                       help="Export formats")
    parser.add_argument("--config", "-c", help="Configuration file (JSON)")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize FlashGenie
    flashgenie = FlashGenie(config)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.suffix.lower() == '.pdf':
        result = flashgenie.process_pdf(
            str(input_path), 
            args.output, 
            args.questions, 
            args.formats
        )
    else:
        # Assume it's a text file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = flashgenie.process_text(
            text, 
            args.output, 
            args.questions, 
            args.formats
        )
    
    # Print results
    print(flashgenie.get_processing_stats(result))
    
    if result.get('success', False):
        print(f"\nFiles saved to: {result['output_dir']}")
        print(f"Summary saved to: {result['output_dir']}/{result['base_filename']}_summary.txt")


if __name__ == "__main__":
    main()
