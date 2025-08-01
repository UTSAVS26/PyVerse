"""
Resume parser for PDF Intelligence Extractor.
"""

import re
from typing import Dict, List, Any
from .base_parser import BaseParser


class ResumeParser(BaseParser):
    """
    Parser for extracting structured data from resume PDFs.
    """
    
    def __init__(self):
        super().__init__()
        self.skills_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'sql', 'mongodb', 'postgresql', 'mysql', 'aws', 'docker', 'kubernetes',
            'machine learning', 'ai', 'nlp', 'data analysis', 'pandas', 'numpy',
            'scikit-learn', 'tensorflow', 'pytorch', 'git', 'agile', 'scrum'
        ]
    
    def get_document_type(self) -> str:
        return "resume"
    
    def extract_name(self, text: str) -> str:
        """Extract candidate name from resume text."""
        # Look for patterns like "Name: John Doe" or prominent name at top
        name_patterns = [
            r'^([A-Z][a-z]+ [A-Z][a-z]+)',
            r'Name[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+)\s*\n'
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                return matches[0].strip()
        
        return ""
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text."""
        found_skills = []
        text_lower = text.lower()
        
        for skill in self.skills_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        # Also look for skills section
        skills_section = re.search(r'skills?[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
        if skills_section:
            skills_text = skills_section.group(1)
            # Extract individual skills
            individual_skills = re.findall(r'\b[A-Za-z\s\+#\.]+\b', skills_text)
            found_skills.extend([skill.strip() for skill in individual_skills if len(skill.strip()) > 2])
        
        return list(set(found_skills))
    
    def extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information from resume text."""
        education = []
        
        # Look for education section
        education_section = re.search(r'education[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
        if education_section:
            edu_text = education_section.group(1)
            
            # Extract degree and institute patterns
            degree_pattern = r'([A-Z][A-Za-z\s]+(?:Bachelor|Master|PhD|B\.Tech|M\.Tech|MBA|BSc|MSc))'
            institute_pattern = r'([A-Z][A-Za-z\s]+(?:University|College|Institute|School))'
            year_pattern = r'(20\d{2}|19\d{2})'
            
            degrees = re.findall(degree_pattern, edu_text)
            institutes = re.findall(institute_pattern, edu_text)
            years = re.findall(year_pattern, edu_text)
            
            for i in range(max(len(degrees), len(institutes))):
                edu_item = {}
                if i < len(degrees):
                    edu_item['degree'] = degrees[i].strip()
                if i < len(institutes):
                    edu_item['institute'] = institutes[i].strip()
                if i < len(years):
                    edu_item['year'] = years[i]
                
                if edu_item:
                    education.append(edu_item)
        
        return education
    
    def extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience from resume text."""
        experience = []
        
        # Look for experience section
        experience_section = re.search(r'experience[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)', text, re.IGNORECASE | re.DOTALL)
        if experience_section:
            exp_text = experience_section.group(1)
            
            # Extract company and role patterns
            company_pattern = r'([A-Z][A-Za-z\s&]+(?:Corp|Inc|Ltd|LLC|Company|Technologies|Solutions))'
            role_pattern = r'([A-Z][A-Za-z\s]+(?:Engineer|Developer|Manager|Analyst|Scientist|Consultant))'
            duration_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s*[-–]\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}))'
            
            companies = re.findall(company_pattern, exp_text)
            roles = re.findall(role_pattern, exp_text)
            durations = re.findall(duration_pattern, exp_text)
            
            for i in range(max(len(companies), len(roles))):
                exp_item = {}
                if i < len(companies):
                    exp_item['company'] = companies[i].strip()
                if i < len(roles):
                    exp_item['role'] = roles[i].strip()
                if i < len(durations):
                    exp_item['duration'] = durations[i].strip()
                
                if exp_item:
                    experience.append(exp_item)
        
        return experience
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse resume PDF and extract structured data.
        
        Args:
            pdf_path (str): Path to the resume PDF file
            
        Returns:
            Dict[str, Any]: Extracted resume data
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata_from_pdf(pdf_path)
        
        # Extract structured data
        name = self.extract_name(text)
        email = self.find_email_addresses(text)
        phone = self.find_phone_numbers(text)
        skills = self.extract_skills(text)
        education = self.extract_education(text)
        experience = self.extract_experience(text)
        
        return {
            "document_type": self.get_document_type(),
            "name": name,
            "email": email[0] if email else "",
            "phone": phone[0] if phone else "",
            "skills": skills,
            "education": education,
            "experience": experience,
            "metadata": metadata
        } 