#!/usr/bin/env python3
"""
Demo script for PDF Intelligence Extractor.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_intelligence_extractor.extractors import ResumeParser, InvoiceParser, ResearchParser
from pdf_intelligence_extractor.utils import AIHelpers, FileUtils


def create_sample_pdf_content(content, filename):
    """Create a temporary file with sample content for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return f.name


def demo_resume_parser():
    """Demo the resume parser."""
    print("📄 Demo: Resume Parser")
    print("-" * 30)
    
    sample_resume = """
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
    
    # Create temporary file
    temp_file = create_sample_pdf_content(sample_resume, "resume.txt")
    
    try:
        parser = ResumeParser()
        
        # Mock the PDF extraction methods
        with open(temp_file, 'r') as f:
            text = f.read()
        
        # Extract data
        name = parser.extract_name(text)
        email = parser.find_email_addresses(text)
        phone = parser.find_phone_numbers(text)
        skills = parser.extract_skills(text)
        education = parser.extract_education(text)
        experience = parser.extract_experience(text)
        
        print(f"Name: {name}")
        print(f"Email: {email[0] if email else 'Not found'}")
        print(f"Phone: {phone[0] if phone else 'Not found'}")
        print(f"Skills: {', '.join(skills)}")
        print(f"Education: {len(education)} entries found")
        print(f"Experience: {len(experience)} entries found")
        
        # Create result
        result = {
            "document_type": "resume",
            "name": name,
            "email": email[0] if email else "",
            "phone": phone[0] if phone else "",
            "skills": skills,
            "education": education,
            "experience": experience
        }
        
        # Save to JSON
        FileUtils.save_json(result, "demo_resume_output.json")
        print("✅ Results saved to demo_resume_output.json")
        
    finally:
        os.unlink(temp_file)


def demo_invoice_parser():
    """Demo the invoice parser."""
    print("\n💰 Demo: Invoice Parser")
    print("-" * 30)
    
    sample_invoice = """
    INVOICE #INV-2024-001
    
    From: ABC Company Inc.
    To: XYZ Corporation
    
    Invoice Date: 2024-01-15
    Due Date: 2024-02-15
    
    Item Description          Quantity    Price    Amount
    Web Development          10 hours    $100.00  $1,000.00
    Database Design          5 hours     $150.00  $750.00
    
    Subtotal: $1,750.00
    Tax: $175.00
    Total: $1,925.00
    """
    
    # Create temporary file
    temp_file = create_sample_pdf_content(sample_invoice, "invoice.txt")
    
    try:
        parser = InvoiceParser()
        
        # Mock the PDF extraction methods
        with open(temp_file, 'r') as f:
            text = f.read()
        
        # Extract data
        invoice_number = parser.extract_invoice_number(text)
        dates = parser.extract_dates(text)
        amounts = parser.extract_amounts(text)
        parties = parser.extract_parties(text)
        line_items = parser.extract_line_items(text)
        
        print(f"Invoice Number: {invoice_number}")
        print(f"Dates: {dates}")
        print(f"Amounts: {amounts}")
        print(f"Parties: {parties}")
        print(f"Line Items: {len(line_items)} items found")
        
        # Create result
        result = {
            "document_type": "invoice",
            "invoice_number": invoice_number,
            "dates": dates,
            "amounts": amounts,
            "parties": parties,
            "line_items": line_items
        }
        
        # Save to JSON
        FileUtils.save_json(result, "demo_invoice_output.json")
        print("✅ Results saved to demo_invoice_output.json")
        
    finally:
        os.unlink(temp_file)


def demo_research_parser():
    """Demo the research paper parser."""
    print("\n🧪 Demo: Research Paper Parser")
    print("-" * 30)
    
    sample_research = """
    Machine Learning Applications in Natural Language Processing
    
    Authors: John Smith, Jane Doe
    Affiliations: Stanford University, MIT
    
    Abstract
    This paper presents novel approaches to natural language processing using machine learning techniques.
    We demonstrate significant improvements in text classification and sentiment analysis tasks.
    
    Keywords: machine learning, natural language processing, text classification, sentiment analysis
    
    Introduction
    Natural language processing has seen remarkable progress in recent years...
    
    References
    1. Smith, J. et al. (2023). "Advances in NLP." Journal of AI Research.
    2. Doe, J. et al. (2023). "Deep Learning for Text." Conference on AI.
    
    DOI: 10.1234/example.doi
    """
    
    # Create temporary file
    temp_file = create_sample_pdf_content(sample_research, "research.txt")
    
    try:
        parser = ResearchParser()
        
        # Mock the PDF extraction methods
        with open(temp_file, 'r') as f:
            text = f.read()
        
        # Extract data
        title = parser.extract_title(text)
        authors = parser.extract_authors(text)
        affiliations = parser.extract_affiliations(text)
        abstract = parser.extract_abstract(text)
        keywords = parser.extract_keywords(text)
        references = parser.extract_references(text)
        doi = parser.extract_doi(text)
        publication_info = parser.extract_publication_info(text)
        
        print(f"Title: {title}")
        print(f"Authors: {', '.join(authors)}")
        print(f"Affiliations: {', '.join(affiliations)}")
        print(f"Abstract: {abstract[:100]}...")
        print(f"Keywords: {', '.join(keywords)}")
        print(f"References: {len(references)} references found")
        print(f"DOI: {doi}")
        print(f"Publication Info: {publication_info}")
        
        # Create result
        result = {
            "document_type": "research_paper",
            "title": title,
            "authors": authors,
            "affiliations": affiliations,
            "abstract": abstract,
            "keywords": keywords,
            "references": references,
            "doi": doi,
            "publication_info": publication_info
        }
        
        # Save to JSON
        FileUtils.save_json(result, "demo_research_output.json")
        print("✅ Results saved to demo_research_output.json")
        
    finally:
        os.unlink(temp_file)


def demo_ai_helpers():
    """Demo the AI helpers."""
    print("\n🤖 Demo: AI Helpers")
    print("-" * 30)
    
    # Test document classification
    resume_text = "This is my resume with education and experience sections."
    invoice_text = "Invoice for services rendered with total amount due."
    research_text = "Abstract: This research paper presents novel findings with references."
    
    print(f"Resume text classified as: {AIHelpers.classify_document_type(resume_text)}")
    print(f"Invoice text classified as: {AIHelpers.classify_document_type(invoice_text)}")
    print(f"Research text classified as: {AIHelpers.classify_document_type(research_text)}")
    
    # Test entity extraction
    entities = AIHelpers.extract_entities(resume_text + " Contact: john@email.com, Phone: (555) 123-4567")
    print(f"Extracted entities: {entities}")


def main():
    """Main demo function."""
    print("🚀 PDF Intelligence Extractor - Demo")
    print("=" * 50)
    
    try:
        # Run demos
        demo_resume_parser()
        demo_invoice_parser()
        demo_research_parser()
        demo_ai_helpers()
        
        print("\n🎉 Demo completed successfully!")
        print("📁 Check the generated JSON files for detailed results.")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 