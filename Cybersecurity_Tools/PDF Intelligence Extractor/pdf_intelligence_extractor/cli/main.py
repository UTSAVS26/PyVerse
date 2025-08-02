"""
Command Line Interface for PDF Intelligence Extractor.
"""

import click
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pdf_intelligence_extractor.extractors import ResumeParser, InvoiceParser, ResearchParser
from pdf_intelligence_extractor.utils import FileUtils, AIHelpers


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """PDF Intelligence Extractor - Extract structured data from PDFs."""
    pass


@cli.command()
@click.option('--file', '-f', required=True, help='Path to PDF file')
@click.option('--type', '-t', type=click.Choice(['resume', 'invoice', 'research']), 
              help='Document type (auto-detect if not specified)')
@click.option('--out', '-o', help='Output file path (JSON format)')
@click.option('--detect-type', is_flag=True, help='Auto-detect document type')
def parse(file, type, out, detect_type):
    """Parse a single PDF file and extract structured data."""
    
    if not FileUtils.validate_pdf_file(file):
        click.echo(f"Error: Invalid PDF file: {file}")
        return
    
    try:
        # Auto-detect document type if requested or not specified
        if detect_type or not type:
            with open(file, 'r', encoding='utf-8') as f:
                # For demo purposes, we'll use a simple text extraction
                # In practice, you'd extract text from PDF here
                text = f.read() if f.readable() else ""
            
            detected_type = AIHelpers.classify_document_type(text)
            if not type:
                type = detected_type
            click.echo(f"Detected document type: {detected_type}")
        
        # Initialize appropriate parser
        if type == 'resume':
            parser = ResumeParser()
        elif type == 'invoice':
            parser = InvoiceParser()
        elif type == 'research':
            parser = ResearchParser()
        else:
            click.echo(f"Error: Unsupported document type: {type}")
            return
        
        # Parse the PDF
        result = parser.parse(file)
        
        # Output results
        if out:
            FileUtils.save_json(result, out)
            click.echo(f"Results saved to: {out}")
        else:
            import json
            click.echo(json.dumps(result, indent=2))
            
    except Exception as e:
        click.echo(f"Error parsing PDF: {str(e)}")


@cli.command()
@click.option('--dir', '-d', required=True, help='Directory containing PDF files')
@click.option('--type', '-t', type=click.Choice(['resume', 'invoice', 'research']), 
              required=True, help='Document type')
@click.option('--out', '-o', required=True, help='Output directory for results')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), 
              default='json', help='Output format')
def batch(dir, type, out, format):
    """Parse multiple PDF files in a directory."""
    
    if not os.path.exists(dir):
        click.echo(f"Error: Directory does not exist: {dir}")
        return
    
    # Get all PDF files
    pdf_files = FileUtils.get_pdf_files(dir)
    if not pdf_files:
        click.echo(f"No PDF files found in directory: {dir}")
        return
    
    click.echo(f"Found {len(pdf_files)} PDF files")
    
    # Initialize parser
    if type == 'resume':
        parser = ResumeParser()
    elif type == 'invoice':
        parser = InvoiceParser()
    elif type == 'research':
        parser = ResearchParser()
    else:
        click.echo(f"Error: Unsupported document type: {type}")
        return
    
    results = []
    
    # Process each file
    for pdf_file in pdf_files:
        try:
            click.echo(f"Processing: {os.path.basename(pdf_file)}")
            result = parser.parse(pdf_file)
            result['source_file'] = os.path.basename(pdf_file)
            results.append(result)
        except Exception as e:
            click.echo(f"Error processing {pdf_file}: {str(e)}")
    
    # Save results
    if format == 'json':
        output_file = os.path.join(out, f"{type}_results.json")
        FileUtils.save_json(results, output_file)
    else:  # csv
        output_file = os.path.join(out, f"{type}_results.csv")
        FileUtils.save_csv(results, output_file)
    
    click.echo(f"Results saved to: {output_file}")
    click.echo(f"Successfully processed {len(results)} files")


@cli.command()
@click.option('--file', '-f', required=True, help='Path to PDF file')
def info(file):
    """Get information about a PDF file."""
    
    if not FileUtils.validate_pdf_file(file):
        click.echo(f"Error: Invalid PDF file: {file}")
        return
    
    file_info = FileUtils.get_file_info(file)
    
    click.echo("File Information:")
    for key, value in file_info.items():
        click.echo(f"  {key}: {value}")


@cli.command()
@click.option('--file', '-f', required=True, help='Path to PDF file')
def detect(file):
    """Detect document type of a PDF file."""
    
    if not FileUtils.validate_pdf_file(file):
        click.echo(f"Error: Invalid PDF file: {file}")
        return
    
    try:
        # Extract text from PDF (simplified for demo)
        # In practice, you'd use pdfplumber or PyMuPDF
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read() if f.readable() else ""
        
        doc_type = AIHelpers.classify_document_type(text)
        click.echo(f"Detected document type: {doc_type}")
        
    except Exception as e:
        click.echo(f"Error detecting document type: {str(e)}")


if __name__ == '__main__':
    cli() 