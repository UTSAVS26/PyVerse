"""
Tests for Flashcard Modules
"""

import pytest
import tempfile
import os
from flashgenie.flashcard.flashcard_formatter import (
    FlashcardFormatter, Flashcard, QuestionType
)
from flashgenie.flashcard.export import FlashcardExporter


class TestFlashcard:
    """Test cases for Flashcard dataclass."""
    
    def test_flashcard_creation(self):
        """Test creating a flashcard."""
        flashcard = Flashcard(
            question="What is mitochondria?",
            answer="The powerhouse of the cell.",
            question_type=QuestionType.FACTUAL,
            difficulty="medium",
            tags=["biology", "cell"],
            source_text="Sample text",
            metadata={"key": "value"}
        )
        
        assert flashcard.question == "What is mitochondria?"
        assert flashcard.answer == "The powerhouse of the cell."
        assert flashcard.question_type == QuestionType.FACTUAL
        assert flashcard.difficulty == "medium"
        assert flashcard.tags == ["biology", "cell"]
        assert flashcard.source_text == "Sample text"
        assert flashcard.metadata == {"key": "value"}
    
    def test_flashcard_defaults(self):
        """Test flashcard creation with defaults."""
        flashcard = Flashcard(
            question="What is DNA?",
            answer="Genetic material.",
            question_type=QuestionType.FACTUAL
        )
        
        assert flashcard.difficulty == "medium"
        assert flashcard.tags == []
        assert flashcard.source_text == ""
        assert flashcard.metadata == {}


class TestFlashcardFormatter:
    """Test cases for FlashcardFormatter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = FlashcardFormatter()
        
        # Sample flashcards for testing
        self.sample_flashcards = [
            Flashcard(
                question="What is mitochondria?",
                answer="The powerhouse of the cell.",
                question_type=QuestionType.FACTUAL,
                tags=["biology"]
            ),
            Flashcard(
                question="How does DNA work?",
                answer="It contains genetic information.",
                question_type=QuestionType.CONCEPTUAL,
                tags=["biology", "genetics"]
            )
        ]
    
    def test_init(self):
        """Test FlashcardFormatter initialization."""
        formatter = FlashcardFormatter()
        assert hasattr(formatter, 'difficulty_keywords')
        assert 'easy' in formatter.difficulty_keywords
        assert 'medium' in formatter.difficulty_keywords
        assert 'hard' in formatter.difficulty_keywords
    
    def test_create_flashcard(self):
        """Test creating a single flashcard."""
        flashcard = self.formatter.create_flashcard(
            question="What is photosynthesis?",
            answer="Process by which plants convert sunlight into energy.",
            question_type=QuestionType.FACTUAL,
            source_text="Biology textbook",
            tags=["biology", "plants"]
        )
        
        assert isinstance(flashcard, Flashcard)
        assert flashcard.question == "What is photosynthesis?"
        assert flashcard.answer == "Process by which plants convert sunlight into energy."
        assert flashcard.question_type == QuestionType.FACTUAL
        assert flashcard.tags == ["biology", "plants"]
        assert flashcard.source_text == "Biology textbook"
    
    def test_create_flashcards_from_qa_pairs(self):
        """Test creating flashcards from Q&A pairs."""
        qa_pairs = [
            ("What is mitochondria?", "The powerhouse of the cell."),
            ("What is DNA?", "Genetic material.")
        ]
        
        flashcards = self.formatter.create_flashcards_from_qa_pairs(
            qa_pairs,
            question_type=QuestionType.FACTUAL,
            source_text="Sample text",
            tags=["biology"]
        )
        
        assert len(flashcards) == 2
        assert all(isinstance(fc, Flashcard) for fc in flashcards)
        assert all(fc.tags == ["biology"] for fc in flashcards)
    
    def test_create_flashcards_from_qa_pairs_empty(self):
        """Test creating flashcards from empty Q&A pairs."""
        flashcards = self.formatter.create_flashcards_from_qa_pairs([])
        assert flashcards == []
    
    def test_create_flashcards_from_qa_pairs_invalid(self):
        """Test creating flashcards from invalid Q&A pairs."""
        qa_pairs = [
            ("", "Valid answer"),  # Empty question
            ("Valid question", ""),  # Empty answer
            ("", "")  # Both empty
        ]
        
        flashcards = self.formatter.create_flashcards_from_qa_pairs(qa_pairs)
        assert len(flashcards) == 0  # All should be filtered out
    
    def test_create_fill_blank_flashcards(self):
        """Test creating fill-in-the-blank flashcards."""
        text = "The mitochondria is the powerhouse of the cell. DNA contains genetic information."
        keywords = ["mitochondria", "DNA"]
        
        flashcards = self.formatter.create_fill_blank_flashcards(
            text, keywords, num_cards=2
        )
        
        assert len(flashcards) <= 2
        assert all(fc.question_type == QuestionType.FILL_BLANK for fc in flashcards)
        assert all("____" in fc.question for fc in flashcards)
        assert all(fc.tags == ["fill_blank"] for fc in flashcards)
    
    def test_create_true_false_flashcards(self):
        """Test creating true/false flashcards."""
        statements = [
            "Mitochondria is the powerhouse of the cell.",
            "DNA contains genetic information."
        ]
        
        flashcards = self.formatter.create_true_false_flashcards(
            statements, num_cards=2
        )
        
        assert len(flashcards) == 2
        assert all(fc.question_type == QuestionType.TRUE_FALSE for fc in flashcards)
        assert all("True or False:" in fc.question for fc in flashcards)
        assert all(fc.answer == "True" for fc in flashcards)
        assert all(fc.tags == ["true_false"] for fc in flashcards)
    
    def test_format_flashcard_for_display_simple(self):
        """Test formatting flashcard for simple display."""
        flashcard = self.sample_flashcards[0]
        result = self.formatter.format_flashcard_for_display(flashcard, "simple")
        
        assert "Q: What is mitochondria?" in result
        assert "A: The powerhouse of the cell." in result
    
    def test_format_flashcard_for_display_detailed(self):
        """Test formatting flashcard for detailed display."""
        flashcard = self.sample_flashcards[0]
        result = self.formatter.format_flashcard_for_display(flashcard, "detailed")
        
        assert "Question: What is mitochondria?" in result
        assert "Answer: The powerhouse of the cell." in result
        assert "Type: factual" in result
        assert "Difficulty: medium" in result
        assert "Tags: biology" in result
    
    def test_format_flashcard_for_display_anki(self):
        """Test formatting flashcard for Anki display."""
        flashcard = self.sample_flashcards[0]
        result = self.formatter.format_flashcard_for_display(flashcard, "anki")
        
        assert result == "What is mitochondria?\tThe powerhouse of the cell."
    
    def test_format_flashcards_for_export_csv(self):
        """Test formatting flashcards for CSV export."""
        result = self.formatter.format_flashcards_for_export(
            self.sample_flashcards, "csv"
        )
        
        assert "Question,Answer,Type,Difficulty,Tags" in result
        assert "What is mitochondria?" in result
        assert "The powerhouse of the cell." in result
        assert "factual" in result
        assert "medium" in result
        assert "biology" in result
    
    def test_format_flashcards_for_export_anki(self):
        """Test formatting flashcards for Anki export."""
        result = self.formatter.format_flashcards_for_export(
            self.sample_flashcards, "anki"
        )
        
        lines = result.strip().split('\n')
        assert len(lines) == 2
        assert "What is mitochondria?\tThe powerhouse of the cell." in lines
        assert "How does DNA work?\tIt contains genetic information." in lines
    
    def test_format_flashcards_for_export_json(self):
        """Test formatting flashcards for JSON export."""
        result = self.formatter.format_flashcards_for_export(
            self.sample_flashcards, "json"
        )
        
        import json
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]['question'] == "What is mitochondria?"
        assert data[0]['answer'] == "The powerhouse of the cell."
        assert data[0]['question_type'] == "factual"
        assert data[0]['difficulty'] == "medium"
        assert data[0]['tags'] == ["biology"]
    
    def test_determine_difficulty(self):
        """Test difficulty determination."""
        # Test easy difficulty
        question = "What is a basic concept?"
        answer = "A simple fundamental idea."
        difficulty = self.formatter._determine_difficulty(question, answer)
        assert difficulty == "easy"
        
        # Test hard difficulty
        question = "What is an advanced complex concept?"
        answer = "A detailed specialized topic."
        difficulty = self.formatter._determine_difficulty(question, answer)
        assert difficulty == "hard"
        
        # Test medium difficulty (default)
        question = "What is a standard concept?"
        answer = "A common topic."
        difficulty = self.formatter._determine_difficulty(question, answer)
        assert difficulty == "medium"
    
    def test_validate_flashcard_valid(self):
        """Test validation of valid flashcard."""
        flashcard = Flashcard(
            question="What is mitochondria?",
            answer="The powerhouse of the cell.",
            question_type=QuestionType.FACTUAL
        )
        
        is_valid, issues = self.formatter.validate_flashcard(flashcard)
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_flashcard_invalid(self):
        """Test validation of invalid flashcard."""
        # Too short question
        flashcard = Flashcard(
            question="Hi",
            answer="Valid answer.",
            question_type=QuestionType.FACTUAL
        )
        
        is_valid, issues = self.formatter.validate_flashcard(flashcard)
        assert is_valid is False
        assert "Question too short" in issues
        
        # Too short answer
        flashcard = Flashcard(
            question="Valid question?",
            answer="No",
            question_type=QuestionType.FACTUAL
        )
        
        is_valid, issues = self.formatter.validate_flashcard(flashcard)
        assert is_valid is False
        assert "Answer too short" in issues
        
        # Empty question
        flashcard = Flashcard(
            question="",
            answer="Valid answer.",
            question_type=QuestionType.FACTUAL
        )
        
        is_valid, issues = self.formatter.validate_flashcard(flashcard)
        assert is_valid is False
        assert "Empty question" in issues
        
        # Question and answer identical
        flashcard = Flashcard(
            question="Same text",
            answer="Same text",
            question_type=QuestionType.FACTUAL
        )
        
        is_valid, issues = self.formatter.validate_flashcard(flashcard)
        assert is_valid is False
        assert "Question and answer are identical" in issues
    
    def test_filter_flashcards_by_quality(self):
        """Test filtering flashcards by quality."""
        # Create mix of valid and invalid flashcards
        flashcards = [
            Flashcard(
                question="What is mitochondria?",
                answer="The powerhouse of the cell.",
                question_type=QuestionType.FACTUAL
            ),
            Flashcard(
                question="Hi",
                answer="No",
                question_type=QuestionType.FACTUAL
            )
        ]
        
        quality_flashcards = self.formatter.filter_flashcards_by_quality(
            flashcards, min_quality_score=0.7
        )
        
        assert len(quality_flashcards) == 1
        assert quality_flashcards[0].question == "What is mitochondria?"
    
    def test_calculate_quality_score(self):
        """Test quality score calculation."""
        # Good flashcard
        good_flashcard = Flashcard(
            question="What is mitochondria?",
            answer="The powerhouse of the cell that generates ATP.",
            question_type=QuestionType.CONCEPTUAL
        )
        
        score = self.formatter._calculate_quality_score(good_flashcard)
        assert score >= 0.7
        
        # Poor flashcard
        poor_flashcard = Flashcard(
            question="Hi",
            answer="No",
            question_type=QuestionType.FACTUAL
        )
        
        score = self.formatter._calculate_quality_score(poor_flashcard)
        assert score < 0.7


class TestFlashcardExporter:
    """Test cases for FlashcardExporter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.exporter = FlashcardExporter()
        
        # Sample flashcards for testing
        self.sample_flashcards = [
            Flashcard(
                question="What is mitochondria?",
                answer="The powerhouse of the cell.",
                question_type=QuestionType.FACTUAL,
                tags=["biology"]
            ),
            Flashcard(
                question="How does DNA work?",
                answer="It contains genetic information.",
                question_type=QuestionType.CONCEPTUAL,
                tags=["biology", "genetics"]
            )
        ]
    
    def test_init(self):
        """Test FlashcardExporter initialization."""
        exporter = FlashcardExporter()
        assert hasattr(exporter, 'formatter')
        assert isinstance(exporter.formatter, FlashcardFormatter)
    
    def test_export_to_csv(self):
        """Test CSV export."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            csv_path = tmp_file.name
        
        try:
            success = self.exporter.export_to_csv(
                self.sample_flashcards, csv_path, include_metadata=True
            )
            
            assert success is True
            
            # Check file content
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "Question,Answer,Type,Difficulty,Tags" in content
            assert "What is mitochondria?" in content
            assert "The powerhouse of the cell." in content
            assert "factual" in content
            assert "biology" in content
            
        finally:
            os.unlink(csv_path)
    
    def test_export_to_csv_no_metadata(self):
        """Test CSV export without metadata."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            csv_path = tmp_file.name
        
        try:
            success = self.exporter.export_to_csv(
                self.sample_flashcards, csv_path, include_metadata=False
            )
            
            assert success is True
            
            # Check file content
            with open(csv_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "Question,Answer" in content
            assert "Type,Difficulty,Tags" not in content
            
        finally:
            os.unlink(csv_path)
    
    def test_export_to_anki(self):
        """Test Anki export."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            anki_path = tmp_file.name
        
        try:
            success = self.exporter.export_to_anki(self.sample_flashcards, anki_path)
            
            assert success is True
            
            # Check file content
            with open(anki_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.strip().split('\n')
            assert len(lines) == 2
            assert "What is mitochondria?\tThe powerhouse of the cell." in lines
            assert "How does DNA work?\tIt contains genetic information." in lines
            
        finally:
            os.unlink(anki_path)
    
    def test_export_to_json(self):
        """Test JSON export."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            json_path = tmp_file.name
        
        try:
            success = self.exporter.export_to_json(
                self.sample_flashcards, json_path, include_metadata=True
            )
            
            assert success is True
            
            # Check file content
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert len(data) == 2
            assert data[0]['question'] == "What is mitochondria?"
            assert data[0]['answer'] == "The powerhouse of the cell."
            assert data[0]['question_type'] == "factual"
            assert data[0]['difficulty'] == "medium"
            assert data[0]['tags'] == ["biology"]
            
        finally:
            os.unlink(json_path)
    
    def test_export_to_txt(self):
        """Test text export."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            txt_path = tmp_file.name
        
        try:
            success = self.exporter.export_to_txt(
                self.sample_flashcards, txt_path, format_type="simple"
            )
            
            assert success is True
            
            # Check file content
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "Q: What is mitochondria?" in content
            assert "A: The powerhouse of the cell." in content
            
        finally:
            os.unlink(txt_path)
    
    def test_export_to_txt_numbered(self):
        """Test numbered text export."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            txt_path = tmp_file.name
        
        try:
            success = self.exporter.export_to_txt(
                self.sample_flashcards, txt_path, format_type="numbered"
            )
            
            assert success is True
            
            # Check file content
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "FlashGenie Flashcards" in content
            assert "1. Q: What is mitochondria?" in content
            assert "2. Q: How does DNA work?" in content
            
        finally:
            os.unlink(txt_path)
    
    def test_export_to_html(self):
        """Test HTML export."""
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            html_path = tmp_file.name
        
        try:
            success = self.exporter.export_to_html(
                self.sample_flashcards, html_path, title="Test Flashcards"
            )
            
            assert success is True
            
            # Check file content
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "<!DOCTYPE html>" in content
            assert "<title>Test Flashcards</title>" in content
            assert "What is mitochondria?" in content
            assert "The powerhouse of the cell." in content
            
        finally:
            os.unlink(html_path)
    
    def test_export_multiple_formats(self):
        """Test exporting to multiple formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = self.exporter.export_multiple_formats(
                self.sample_flashcards,
                temp_dir,
                "test_flashcards",
                formats=['csv', 'anki', 'json']
            )
            
            assert len(results) == 3
            assert all(isinstance(success, bool) for success in results.values())
            
            # Check that files were created
            csv_path = os.path.join(temp_dir, "test_flashcards.csv")
            anki_path = os.path.join(temp_dir, "test_flashcards.anki")
            json_path = os.path.join(temp_dir, "test_flashcards.json")
            
            assert os.path.exists(csv_path) or not results['csv']
            assert os.path.exists(anki_path) or not results['anki']
            assert os.path.exists(json_path) or not results['json']
    
    def test_create_export_summary(self):
        """Test creating export summary."""
        export_results = {'csv': True, 'anki': True, 'json': False}
        summary = self.exporter.create_export_summary(
            self.sample_flashcards, export_results
        )
        
        assert "Export Summary" in summary
        assert "Total flashcards: 2" in summary
        assert "CSV: ✓ Success" in summary
        assert "ANKI: ✓ Success" in summary
        assert "JSON: ✗ Failed" in summary
        assert "Question Type Distribution:" in summary
        assert "Difficulty Distribution:" in summary
