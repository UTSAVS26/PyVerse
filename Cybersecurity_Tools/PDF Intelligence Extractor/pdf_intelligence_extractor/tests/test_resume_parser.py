"""
Test cases for ResumeParser.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from pdf_intelligence_extractor.extractors.resume_parser import ResumeParser


class TestResumeParser:
    """Test cases for ResumeParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ResumeParser()
        self.sample_resume_text = """
        John Doe
        john.doe@email.com
        (555) 123-4567
        
        EDUCATION
        Bachelor of Science in Computer Science
        XYZ University, 2021
        
        EXPERIENCE
        Data Scientist
        TechCorp Inc.
        Jan 2022 - Present
        
        SKILLS
        Python, Machine Learning, Data Analysis, SQL, AWS
        """
    
    def test_get_document_type(self):
        """Test document type identification."""
        assert self.parser.get_document_type() == "resume"
    
    def test_extract_name(self):
        """Test name extraction from resume text."""
        name = self.parser.extract_name(self.sample_resume_text)
        assert name == "John Doe"
    
    def test_extract_name_no_name(self):
        """Test name extraction when no name is found."""
        text_without_name = "This is a resume without a clear name at the top."
        name = self.parser.extract_name(text_without_name)
        assert name == ""
    
    def test_find_email_addresses(self):
        """Test email address extraction."""
        emails = self.parser.find_email_addresses(self.sample_resume_text)
        assert "john.doe@email.com" in emails
    
    def test_find_phone_numbers(self):
        """Test phone number extraction."""
        phones = self.parser.find_phone_numbers(self.sample_resume_text)
        assert "(555) 123-4567" in phones
    
    def test_extract_skills(self):
        """Test skills extraction."""
        skills = self.parser.extract_skills(self.sample_resume_text)
        expected_skills = ["Python", "Machine Learning", "Data Analysis", "SQL", "AWS"]
        for skill in expected_skills:
            assert skill in skills
    
    def test_extract_education(self):
        """Test education extraction."""
        education = self.parser.extract_education(self.sample_resume_text)
        assert len(education) > 0
        assert "Bachelor of Science in Computer Science" in [edu.get('degree', '') for edu in education]
        assert "XYZ University" in [edu.get('institute', '') for edu in education]
    
    def test_extract_experience(self):
        """Test experience extraction."""
        experience = self.parser.extract_experience(self.sample_resume_text)
        assert len(experience) > 0
        assert "Data Scientist" in [exp.get('role', '') for exp in experience]
        assert "TechCorp Inc." in [exp.get('company', '') for exp in experience]
    
    @patch('pdf_intelligence_extractor.extractors.base_parser.PDF_LIBS_AVAILABLE', False)
    def test_parse_resume(self):
        """Test complete resume parsing."""
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.sample_resume_text.encode('utf-8'))
            tmp_file_path = tmp_file.name
        
        try:
            result = self.parser.parse(tmp_file_path)
            
            # Verify result structure
            assert result["document_type"] == "resume"
            assert result["name"] == "John Doe"
            assert result["email"] == "john.doe@email.com"
            assert result["phone"] == "(555) 123-4567"
            assert len(result["skills"]) > 0
            assert len(result["education"]) > 0
            assert len(result["experience"]) > 0
            assert "metadata" in result
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_extract_skills_from_keywords(self):
        """Test skills extraction from predefined keywords."""
        text = "I have experience with Python, Java, and AWS."
        skills = self.parser.extract_skills(text)
        assert "Python" in skills
        assert "Java" in skills
        assert "AWS" in skills
    
    def test_extract_skills_from_section(self):
        """Test skills extraction from skills section."""
        text = """
        SKILLS
        Programming: Python, Java, JavaScript
        Databases: SQL, MongoDB
        Cloud: AWS, Docker
        """
        skills = self.parser.extract_skills(text)
        assert "Python" in skills
        assert "Java" in skills
        assert "JavaScript" in skills
        assert "SQL" in skills
        assert "MongoDB" in skills
        assert "AWS" in skills
        assert "Docker" in skills
    
    def test_extract_education_multiple_entries(self):
        """Test education extraction with multiple entries."""
        text = """
        EDUCATION
        Master of Science in Data Science
        ABC University, 2023
        
        Bachelor of Technology in Computer Science
        XYZ Institute, 2021
        """
        education = self.parser.extract_education(text)
        assert len(education) >= 1
        degrees = [edu.get('degree', '') for edu in education]
        assert any('Master' in degree for degree in degrees)
        assert any('Bachelor' in degree for degree in degrees)
    
    def test_extract_experience_multiple_entries(self):
        """Test experience extraction with multiple entries."""
        text = """
        EXPERIENCE
        Senior Data Scientist
        BigTech Corp
        Jan 2023 - Present
        
        Junior Developer
        Startup Inc
        Jun 2021 - Dec 2022
        """
        experience = self.parser.extract_experience(text)
        assert len(experience) >= 1
        roles = [exp.get('role', '') for exp in experience]
        companies = [exp.get('company', '') for exp in experience]
        assert any('Senior Data Scientist' in role for role in roles)
        assert any('Junior Developer' in role for role in roles)
        assert any('BigTech Corp' in company for company in companies)
        assert any('Startup Inc' in company for company in companies)
    
    def test_parse_empty_text(self):
        """Test parsing with empty text."""
        with patch.object(self.parser, 'extract_text_from_pdf', return_value=""):
            with patch.object(self.parser, 'extract_metadata_from_pdf', return_value={}):
                result = self.parser.parse("dummy_path")
                assert result["document_type"] == "resume"
                assert result["name"] == ""
                assert result["email"] == ""
                assert result["phone"] == ""
                assert result["skills"] == []
                assert result["education"] == []
                assert result["experience"] == []
    
    def test_parse_with_special_characters(self):
        """Test parsing with special characters in text."""
        text_with_special = """
        José García
        josé.garcía@email.com
        +1 (555) 123-4567
        
        EDUCATION
        Bachelor's Degree in Computer Science
        Universität München, 2021
        """
        
        name = self.parser.extract_name(text_with_special)
        emails = self.parser.find_email_addresses(text_with_special)
        phones = self.parser.find_phone_numbers(text_with_special)
        
        assert name == "José García"
        assert "josé.garcía@email.com" in emails
        assert "+1 (555) 123-4567" in phones 