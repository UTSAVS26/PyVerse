"""
Flashcard Export Module

Exports flashcards to various formats (CSV, Anki, PDF, etc.).
"""

import os
import csv
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from .flashcard_formatter import Flashcard, FlashcardFormatter

logger = logging.getLogger(__name__)


class FlashcardExporter:
    """Exports flashcards to various formats."""
    
    def __init__(self):
        """Initialize the flashcard exporter."""
        self.formatter = FlashcardFormatter()
    
    def export_to_csv(self, flashcards: List[Flashcard], output_path: str, 
                     include_metadata: bool = True) -> bool:
        """
        Export flashcards to CSV format.
        
        Args:
            flashcards: List of flashcards to export
            output_path: Path to output CSV file
            include_metadata: Whether to include metadata columns
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                if include_metadata:
                    fieldnames = ['Question', 'Answer', 'Type', 'Difficulty', 'Tags', 'Source_Text']
                else:
                    fieldnames = ['Question', 'Answer']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for flashcard in flashcards:
                    row = {
                        'Question': flashcard.question,
                        'Answer': flashcard.answer
                    }
                    
                    if include_metadata:
                        row.update({
                            'Type': flashcard.question_type.value,
                            'Difficulty': flashcard.difficulty,
                            'Tags': ';'.join(flashcard.tags),
                            'Source_Text': flashcard.source_text[:100] + '...' if len(flashcard.source_text) > 100 else flashcard.source_text
                        })
                    
                    writer.writerow(row)
            
            logger.info(f"Exported {len(flashcards)} flashcards to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False
    
    def export_to_anki(self, flashcards: List[Flashcard], output_path: str) -> bool:
        """
        Export flashcards to Anki-compatible format.
        
        Args:
            flashcards: List of flashcards to export
            output_path: Path to output file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for flashcard in flashcards:
                    # Anki format: question\tanswer
                    line = f"{flashcard.question}\t{flashcard.answer}\n"
                    f.write(line)
            
            logger.info(f"Exported {len(flashcards)} flashcards to Anki format: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to Anki format: {e}")
            return False
    
    def export_to_json(self, flashcards: List[Flashcard], output_path: str, 
                      include_metadata: bool = True) -> bool:
        """
        Export flashcards to JSON format.
        
        Args:
            flashcards: List of flashcards to export
            output_path: Path to output JSON file
            include_metadata: Whether to include metadata
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            flashcard_data = []
            
            for flashcard in flashcards:
                data = {
                    'question': flashcard.question,
                    'answer': flashcard.answer
                }
                
                if include_metadata:
                    data.update({
                        'question_type': flashcard.question_type.value,
                        'difficulty': flashcard.difficulty,
                        'tags': flashcard.tags,
                        'source_text': flashcard.source_text,
                        'metadata': flashcard.metadata
                    })
                
                flashcard_data.append(data)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(flashcard_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(flashcards)} flashcards to JSON: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return False
    
    def export_to_txt(self, flashcards: List[Flashcard], output_path: str, 
                     format_type: str = "simple") -> bool:
        """
        Export flashcards to plain text format.
        
        Args:
            flashcards: List of flashcards to export
            output_path: Path to output text file
            format_type: Format type ('simple', 'detailed', 'numbered')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format_type == "numbered":
                    f.write("FlashGenie Flashcards\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, flashcard in enumerate(flashcards, 1):
                        f.write(f"{i}. Q: {flashcard.question}\n")
                        f.write(f"   A: {flashcard.answer}\n\n")
                
                elif format_type == "detailed":
                    f.write("FlashGenie Flashcards - Detailed Format\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, flashcard in enumerate(flashcards, 1):
                        f.write(f"Card {i}:\n")
                        f.write(f"Question: {flashcard.question}\n")
                        f.write(f"Answer: {flashcard.answer}\n")
                        f.write(f"Type: {flashcard.question_type.value}\n")
                        f.write(f"Difficulty: {flashcard.difficulty}\n")
                        f.write(f"Tags: {', '.join(flashcard.tags)}\n")
                        f.write("-" * 30 + "\n\n")
                
                else:  # simple
                    for flashcard in flashcards:
                        f.write(f"Q: {flashcard.question}\n")
                        f.write(f"A: {flashcard.answer}\n\n")
            
            logger.info(f"Exported {len(flashcards)} flashcards to text: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to text: {e}")
            return False
    
    def export_to_html(self, flashcards: List[Flashcard], output_path: str, 
                      title: str = "FlashGenie Flashcards") -> bool:
        """
        Export flashcards to HTML format.
        
        Args:
            flashcards: List of flashcards to export
            output_path: Path to output HTML file
            title: Title for the HTML page
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            color: #333;
        }}
        .flashcard {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .question {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .answer {{
            color: #34495e;
            border-left: 3px solid #3498db;
            padding-left: 15px;
        }}
        .metadata {{
            font-size: 0.8em;
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .tags {{
            display: inline-block;
            background: #ecf0f1;
            padding: 2px 8px;
            border-radius: 12px;
            margin-right: 5px;
            font-size: 0.7em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total cards: {len(flashcards)}</p>
    </div>
"""
            
            for i, flashcard in enumerate(flashcards, 1):
                tags_html = ''.join([f'<span class="tags">{tag}</span>' for tag in flashcard.tags])
                
                html_content += f"""
    <div class="flashcard">
        <div class="question">Q{i}: {flashcard.question}</div>
        <div class="answer">A: {flashcard.answer}</div>
        <div class="metadata">
            Type: {flashcard.question_type.value} | 
            Difficulty: {flashcard.difficulty} | 
            Tags: {tags_html}
        </div>
    </div>
"""
            
            html_content += """
</body>
</html>
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Exported {len(flashcards)} flashcards to HTML: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to HTML: {e}")
            return False
    
    def export_to_pdf(self, flashcards: List[Flashcard], output_path: str, 
                     title: str = "FlashGenie Flashcards") -> bool:
        """
        Export flashcards to PDF format.
        
        Args:
            flashcards: List of flashcards to export
            output_path: Path to output PDF file
            title: Title for the PDF
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # First create HTML, then convert to PDF
            html_path = output_path.replace('.pdf', '.html')
            success = self.export_to_html(flashcards, html_path, title)
            
            if not success:
                return False
            
            # Try to convert HTML to PDF using weasyprint or similar
            try:
                import weasyprint
                weasyprint.HTML(filename=html_path).write_pdf(output_path)
                os.remove(html_path)  # Clean up HTML file
                logger.info(f"Exported {len(flashcards)} flashcards to PDF: {output_path}")
                return True
            except ImportError:
                logger.warning("weasyprint not available. Install with: pip install weasyprint")
                logger.info(f"HTML file created at: {html_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export to PDF: {e}")
            return False
    
    def export_multiple_formats(self, flashcards: List[Flashcard], output_dir: str, 
                               base_filename: str, formats: List[str] = None) -> Dict[str, bool]:
        """
        Export flashcards to multiple formats.
        
        Args:
            flashcards: List of flashcards to export
            output_dir: Directory to save files
            base_filename: Base filename (without extension)
            formats: List of formats to export ('csv', 'anki', 'json', 'txt', 'html', 'pdf')
            
        Returns:
            Dictionary mapping format to success status
        """
        if formats is None:
            formats = ['csv', 'anki', 'json', 'txt', 'html']
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for format_type in formats:
            output_path = os.path.join(output_dir, f"{base_filename}.{format_type}")
            
            if format_type == 'csv':
                results['csv'] = self.export_to_csv(flashcards, output_path)
            elif format_type == 'anki':
                results['anki'] = self.export_to_anki(flashcards, output_path)
            elif format_type == 'json':
                results['json'] = self.export_to_json(flashcards, output_path)
            elif format_type == 'txt':
                results['txt'] = self.export_to_txt(flashcards, output_path)
            elif format_type == 'html':
                results['html'] = self.export_to_html(flashcards, output_path)
            elif format_type == 'pdf':
                results['pdf'] = self.export_to_pdf(flashcards, output_path)
        
        return results
    
    def create_export_summary(self, flashcards: List[Flashcard], export_results: Dict[str, bool]) -> str:
        """
        Create a summary of the export process.
        
        Args:
            flashcards: List of flashcards that were exported
            export_results: Results from export_multiple_formats
            
        Returns:
            Summary string
        """
        summary = f"Export Summary\n{'='*50}\n"
        summary += f"Total flashcards: {len(flashcards)}\n"
        summary += f"Export timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        summary += "Export Results:\n"
        for format_type, success in export_results.items():
            status = "✓ Success" if success else "✗ Failed"
            summary += f"  {format_type.upper()}: {status}\n"
        
        # Add statistics
        if flashcards:
            question_types = {}
            difficulties = {}
            
            for flashcard in flashcards:
                question_types[flashcard.question_type.value] = question_types.get(flashcard.question_type.value, 0) + 1
                difficulties[flashcard.difficulty] = difficulties.get(flashcard.difficulty, 0) + 1
            
            summary += f"\nQuestion Type Distribution:\n"
            for qtype, count in question_types.items():
                summary += f"  {qtype}: {count}\n"
            
            summary += f"\nDifficulty Distribution:\n"
            for difficulty, count in difficulties.items():
                summary += f"  {difficulty}: {count}\n"
        
        return summary
