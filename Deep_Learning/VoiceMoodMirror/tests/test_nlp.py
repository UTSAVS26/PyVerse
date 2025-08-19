"""
Tests for NLP Modules
"""

import pytest
from unittest.mock import patch, MagicMock
from flashgenie.nlp.keyword_extractor import KeywordExtractor
from flashgenie.nlp.question_generator import QuestionGenerator
from flashgenie.nlp.answer_selector import AnswerSelector


class TestKeywordExtractor:
    """Test cases for KeywordExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = KeywordExtractor(use_spacy=False, use_keybert=False)
        
        # Sample text for testing
        self.sample_text = """
        The mitochondria is the powerhouse of the cell. 
        It generates ATP through cellular respiration.
        DNA is composed of nucleotides containing bases, phosphate, and a sugar.
        Photosynthesis is the process by which plants convert sunlight into energy.
        """
    
    def test_init(self):
        """Test KeywordExtractor initialization."""
        extractor = KeywordExtractor(use_spacy=True, use_keybert=True)
        assert extractor.use_spacy is True
        assert extractor.use_keybert is True
        
        extractor = KeywordExtractor(use_spacy=False, use_keybert=False)
        assert extractor.use_spacy is False
        assert extractor.use_keybert is False
    
    def test_extract_keywords_tfidf(self):
        """Test TF-IDF keyword extraction."""
        result = self.extractor.extract_keywords_tfidf(self.sample_text, top_k=5)
        
        assert len(result) <= 5
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], str) for item in result)
        assert all(isinstance(item[1], (int, float)) for item in result)
    
    def test_extract_keywords_tfidf_empty(self):
        """Test TF-IDF extraction with empty text."""
        result = self.extractor.extract_keywords_tfidf("", top_k=5)
        assert result == []
    
    def test_extract_keywords_rake(self):
        """Test RAKE keyword extraction."""
        result = self.extractor.extract_keywords_rake(self.sample_text, top_k=5)
        
        assert len(result) <= 5
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], str) for item in result)
        assert all(isinstance(item[1], (int, float)) for item in result)
    
    def test_extract_keywords_rake_empty(self):
        """Test RAKE extraction with empty text."""
        result = self.extractor.extract_keywords_rake("", top_k=5)
        assert result == []
    
    @patch('flashgenie.nlp.keyword_extractor.spacy')
    def test_extract_keywords_spacy(self, mock_spacy):
        """Test spaCy keyword extraction."""
        # Mock spaCy
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        
        # Mock entities
        mock_ent = MagicMock()
        mock_ent.text = "Mitochondria"
        mock_ent.label_ = "ORG"
        mock_doc.ents = [mock_ent]
        
        # Mock noun chunks
        mock_chunk = MagicMock()
        mock_chunk.text = "cellular respiration"
        mock_doc.noun_chunks = [mock_chunk]
        
        # Mock tokens
        mock_token = MagicMock()
        mock_token.pos_ = "NOUN"
        mock_token.is_stop = False
        mock_token.text = "mitochondria"
        mock_doc.__iter__ = lambda self: iter([mock_token])
        
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp
        
        extractor = KeywordExtractor(use_spacy=True, use_keybert=False)
        result = extractor.extract_keywords_spacy(self.sample_text, top_k=5)
        
        assert len(result) <= 5
        assert all(isinstance(item, tuple) for item in result)
    
    def test_extract_keywords_spacy_disabled(self):
        """Test spaCy extraction when disabled."""
        result = self.extractor.extract_keywords_spacy(self.sample_text, top_k=5)
        assert result == []
    
    @patch('flashgenie.nlp.keyword_extractor.KeyBERT')
    def test_extract_keywords_keybert(self, mock_keybert):
        """Test KeyBERT keyword extraction."""
        # Mock KeyBERT
        mock_model = MagicMock()
        mock_model.extract_keywords.return_value = [("mitochondria", 0.8), ("DNA", 0.7)]
        mock_keybert.return_value = mock_model
        
        extractor = KeywordExtractor(use_spacy=False, use_keybert=True)
        result = extractor.extract_keywords_keybert(self.sample_text, top_k=5)
        
        assert len(result) == 2
        assert result[0][0] == "mitochondria"
        assert result[0][1] == 0.8
    
    def test_extract_keywords_keybert_disabled(self):
        """Test KeyBERT extraction when disabled."""
        result = self.extractor.extract_keywords_keybert(self.sample_text, top_k=5)
        assert result == []
    
    def test_extract_keywords_combined(self):
        """Test combined keyword extraction."""
        result = self.extractor.extract_keywords_combined(self.sample_text, top_k=10)
        
        assert len(result) <= 10
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
    
    def test_extract_key_concepts(self):
        """Test key concept extraction."""
        result = self.extractor.extract_key_concepts(self.sample_text, min_length=2)
        
        assert isinstance(result, list)
        assert all(isinstance(concept, str) for concept in result)
    
    def test_get_context_for_keyword(self):
        """Test getting context for a keyword."""
        result = self.extractor.get_context_for_keyword(self.sample_text, "mitochondria")
        
        assert isinstance(result, list)
        assert all(isinstance(sentence, str) for sentence in result)
        assert all("mitochondria" in sentence.lower() for sentence in result)


class TestQuestionGenerator:
    """Test cases for QuestionGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = QuestionGenerator(use_transformers=False)
        
        # Sample text and keywords for testing
        self.sample_text = """
        The mitochondria is the powerhouse of the cell. 
        It generates ATP through cellular respiration.
        DNA is composed of nucleotides containing bases, phosphate, and a sugar.
        """
        self.sample_keywords = ["mitochondria", "DNA", "ATP", "cellular respiration"]
    
    def test_init(self):
        """Test QuestionGenerator initialization."""
        generator = QuestionGenerator(use_transformers=True)
        assert generator.use_transformers is True
        
        generator = QuestionGenerator(use_transformers=False)
        assert generator.use_transformers is False
    
    def test_generate_questions_rule_based(self):
        """Test rule-based question generation."""
        result = self.generator.generate_questions_rule_based(
            self.sample_text, self.sample_keywords, num_questions=5
        )
        
        assert len(result) <= 5
        assert all(isinstance(question, str) for question in result)
        assert all(len(question) > 0 for question in result)
    
    def test_generate_questions_rule_based_empty(self):
        """Test rule-based generation with empty input."""
        result = self.generator.generate_questions_rule_based("", [], num_questions=5)
        assert result == []
    
    def test_generate_fill_blank_questions(self):
        """Test fill-in-the-blank question generation."""
        result = self.generator.generate_fill_blank_questions(
            self.sample_text, self.sample_keywords, num_questions=3
        )
        
        assert len(result) <= 3
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)
        assert all(isinstance(item[0], str) for item in result)
        assert all(isinstance(item[1], str) for item in result)
    
    def test_generate_questions_from_sentences(self):
        """Test question generation from sentences."""
        sentences = [
            "The mitochondria is the powerhouse of the cell.",
            "DNA contains genetic information."
        ]
        result = self.generator.generate_questions_from_sentences(sentences, num_questions=3)
        
        assert len(result) <= 3
        assert all(isinstance(question, str) for question in result)
    
    def test_extract_sentences_with_keywords(self):
        """Test extraction of sentences containing keywords."""
        result = self.generator._extract_sentences_with_keywords(
            self.sample_text, self.sample_keywords
        )
        
        assert isinstance(result, list)
        assert all(isinstance(sentence, str) for sentence in result)
        assert all(any(keyword.lower() in sentence.lower() 
                      for keyword in self.sample_keywords) for sentence in result)
    
    def test_extract_concepts_from_sentence(self):
        """Test extraction of concepts from a sentence."""
        sentence = "The Mitochondria is the Powerhouse of the Cell."
        result = self.generator._extract_concepts_from_sentence(sentence)
        
        assert isinstance(result, list)
        assert all(isinstance(concept, str) for concept in result)
    
    def test_is_valid_question(self):
        """Test question validation."""
        valid_questions = [
            "What is mitochondria?",
            "How does DNA work?",
            "Why is ATP important?"
        ]
        
        invalid_questions = [
            "",
            "Short",
            "This is not a question"
        ]
        
        for question in valid_questions:
            assert self.generator._is_valid_question(question)
        
        for question in invalid_questions:
            assert not self.generator._is_valid_question(question)
    
    def test_generate_questions_combined(self):
        """Test combined question generation."""
        result = self.generator.generate_questions_combined(
            self.sample_text, self.sample_keywords, num_questions=10
        )
        
        assert len(result) <= 10
        assert all(isinstance(question, str) for question in result)
    
    @patch('flashgenie.nlp.question_generator.pipeline')
    def test_generate_questions_transformers(self, mock_pipeline):
        """Test transformer-based question generation."""
        # Mock transformers pipeline
        mock_pipeline.return_value = [
            {'generated_text': 'What is mitochondria?'},
            {'generated_text': 'How does DNA work?'}
        ]
        
        generator = QuestionGenerator(use_transformers=True)
        result = generator.generate_questions_transformers(self.sample_text, num_questions=2)
        
        assert len(result) <= 2
        assert all(isinstance(question, str) for question in result)


class TestAnswerSelector:
    """Test cases for AnswerSelector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = AnswerSelector(use_spacy=False)
        
        # Sample text and questions for testing
        self.sample_text = """
        The mitochondria is the powerhouse of the cell. 
        It generates ATP through cellular respiration.
        DNA is composed of nucleotides containing bases, phosphate, and a sugar.
        Photosynthesis is the process by which plants convert sunlight into energy.
        """
        self.sample_questions = [
            "What is mitochondria?",
            "How does DNA work?",
            "What is photosynthesis?"
        ]
    
    def test_init(self):
        """Test AnswerSelector initialization."""
        selector = AnswerSelector(use_spacy=True)
        assert selector.use_spacy is True
        
        selector = AnswerSelector(use_spacy=False)
        assert selector.use_spacy is False
    
    def test_find_answer_for_question(self):
        """Test finding answer for a single question."""
        question = "What is mitochondria?"
        result = self.selector.find_answer_for_question(question, self.sample_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "mitochondria" in result.lower()
    
    def test_find_answer_for_question_empty(self):
        """Test finding answer with empty input."""
        result = self.selector.find_answer_for_question("", self.sample_text)
        assert result == ""
        
        result = self.selector.find_answer_for_question("What is X?", "")
        assert result == ""
    
    def test_find_answers_for_questions(self):
        """Test finding answers for multiple questions."""
        result = self.selector.find_answers_for_questions(
            self.sample_questions, self.sample_text
        )
        
        assert len(result) == len(self.sample_questions)
        assert all(isinstance(answer, str) for answer in result)
        assert all(len(answer) > 0 for answer in result)
    
    def test_extract_definitions(self):
        """Test definition extraction."""
        keywords = ["mitochondria", "DNA", "photosynthesis"]
        result = self.selector.extract_definitions(self.sample_text, keywords)
        
        assert isinstance(result, dict)
        assert all(isinstance(definition, str) for definition in result.values())
    
    def test_extract_key_terms_from_question(self):
        """Test extraction of key terms from questions."""
        question = "What is the function of Mitochondria in cells?"
        result = self.selector._extract_key_terms_from_question(question)
        
        assert isinstance(result, list)
        assert "Mitochondria" in result
    
    def test_find_relevant_sentences(self):
        """Test finding relevant sentences."""
        key_terms = ["mitochondria", "DNA"]
        result = self.selector._find_relevant_sentences(self.sample_text, key_terms)
        
        assert isinstance(result, list)
        assert all(isinstance(sentence, str) for sentence in result)
        assert all(any(term.lower() in sentence.lower() for term in key_terms) 
                  for sentence in result)
    
    def test_select_best_answer(self):
        """Test selecting the best answer from multiple sentences."""
        sentences = [
            "The mitochondria is the powerhouse of the cell.",
            "It generates ATP through cellular respiration.",
            "This is a random sentence about something else."
        ]
        question = "What is mitochondria?"
        result = self.selector._select_best_answer(question, sentences, max_length=100)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "mitochondria" in result.lower()
    
    def test_calculate_relevance_score(self):
        """Test relevance score calculation."""
        question = "What is mitochondria?"
        sentence = "The mitochondria is the powerhouse of the cell."
        result = self.selector._calculate_relevance_score(question, sentence)
        
        assert isinstance(result, float)
        assert result >= 0.0
    
    def test_contains_definition_pattern(self):
        """Test definition pattern detection."""
        definition_sentence = "Mitochondria is the powerhouse of the cell."
        non_definition_sentence = "The cell contains many organelles."
        
        assert self.selector._contains_definition_pattern(definition_sentence)
        assert not self.selector._contains_definition_pattern(non_definition_sentence)
    
    def test_extract_definition_for_keyword(self):
        """Test definition extraction for a specific keyword."""
        result = self.selector._extract_definition_for_keyword(
            self.sample_text, "mitochondria"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "mitochondria" in result.lower()
    
    def test_generate_summary_answer(self):
        """Test summary answer generation."""
        question = "What is mitochondria?"
        relevant_text = "The mitochondria is the powerhouse of the cell."
        result = self.selector.generate_summary_answer(question, relevant_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result) <= 150  # max_length default
