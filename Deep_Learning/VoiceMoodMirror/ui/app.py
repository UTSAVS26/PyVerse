"""
FlashGenie Streamlit Web Interface

A beautiful and modern web interface for the FlashGenie flashcard generator.
"""

import streamlit as st
import os
import tempfile
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Import FlashGenie components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flashgenie.main import FlashGenie
from flashgenie.flashcard.flashcard_formatter import QuestionType


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="FlashGenie - AI Flashcard Generator",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #ecf0f1;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    .success-box {
        background-color: #d5f4e6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #27ae60;
    }
    .info-box {
        background-color: #d6eaf8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧠 FlashGenie</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Flashcard Generator from PDFs and Text</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Input type selection
        input_type = st.radio(
            "Input Type",
            ["PDF Upload", "Text Input"],
            help="Choose whether to upload a PDF or input text directly"
        )
        
        # Number of questions
        num_questions = st.slider(
            "Number of Questions",
            min_value=5,
            max_value=50,
            value=15,
            help="Number of flashcards to generate"
        )
        
        # Export formats
        st.subheader("Export Formats")
        export_formats = st.multiselect(
            "Select export formats",
            ["CSV", "Anki", "JSON", "TXT", "HTML", "PDF"],
            default=["CSV", "Anki", "JSON", "HTML"],
            help="Choose which formats to export"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            use_spacy = st.checkbox("Use spaCy NLP", value=True, help="Enable advanced NLP processing")
            use_transformers = st.checkbox("Use Transformers", value=True, help="Enable transformer-based question generation")
            remove_references = st.checkbox("Remove References", value=True, help="Remove reference sections from text")
            remove_footnotes = st.checkbox("Remove Footnotes", value=True, help="Remove footnotes from text")
            min_quality_score = st.slider(
                "Minimum Quality Score",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Minimum quality score for flashcards (0-1)"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Input")
        
        if input_type == "PDF Upload":
            uploaded_file = st.file_uploader(
                "Upload a PDF file",
                type=['pdf'],
                help="Upload a PDF file to generate flashcards from"
            )
            
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    pdf_path = tmp_file.name
                
                st.success(f"✅ Uploaded: {uploaded_file.name}")
                
                # Show file info
                file_size = len(uploaded_file.getvalue()) / 1024  # KB
                st.info(f"📊 File size: {file_size:.1f} KB")
        
        else:  # Text Input
            text_input = st.text_area(
                "Enter your text",
                height=300,
                placeholder="Paste your text here to generate flashcards...",
                help="Enter or paste text content to generate flashcards from"
            )
            
            if text_input:
                st.success(f"✅ Text loaded ({len(text_input)} characters)")
    
    with col2:
        st.header("📊 Statistics")
        
        if input_type == "PDF Upload" and 'uploaded_file' in locals() and uploaded_file is not None:
            st.metric("File Size", f"{file_size:.1f} KB")
        elif input_type == "Text Input" and 'text_input' in locals() and text_input:
            st.metric("Characters", len(text_input))
            st.metric("Words", len(text_input.split()))
            st.metric("Sentences", len([s for s in text_input.split('.') if s.strip()]))
    
    # Process button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button(
            "🚀 Generate Flashcards",
            type="primary",
            use_container_width=True,
            help="Click to start generating flashcards"
        )
    
    # Processing and results
    if process_button:
        # Check if we have input
        if input_type == "PDF Upload":
            if 'uploaded_file' not in locals() or uploaded_file is None:
                st.error("❌ Please upload a PDF file first!")
                return
        else:  # Text Input
            if 'text_input' not in locals() or not text_input.strip():
                st.error("❌ Please enter some text first!")
                return
        
        # Show processing status
        with st.spinner("🧠 Processing your content..."):
            # Initialize FlashGenie with settings
            config = {
                'use_spacy': use_spacy,
                'use_transformers': use_transformers,
                'remove_references': remove_references,
                'remove_footnotes': remove_footnotes,
                'min_quality_score': min_quality_score,
                'num_questions': num_questions
            }
            
            flashgenie = FlashGenie(config)
            
            # Process content
            if input_type == "PDF Upload":
                result = flashgenie.process_pdf(
                    pdf_path,
                    "output",
                    num_questions,
                    [fmt.lower() for fmt in export_formats]
                )
                
                # Clean up temporary file
                os.unlink(pdf_path)
            else:
                result = flashgenie.process_text(
                    text_input,
                    "output",
                    num_questions,
                    [fmt.lower() for fmt in export_formats]
                )
        
        # Display results
        if result.get('success', False):
            st.success("🎉 Flashcards generated successfully!")
            
            # Show statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Flashcards", result['num_flashcards'])
            with col2:
                st.metric("Questions Generated", result['num_questions_generated'])
            with col3:
                st.metric("Keywords Extracted", result['num_keywords_extracted'])
            with col4:
                st.metric("Export Formats", len([k for k, v in result['export_results'].items() if v]))
            
            # Show export results
            st.subheader("📤 Export Results")
            export_results = result.get('export_results', {})
            
            for format_type, success in export_results.items():
                if success:
                    st.success(f"✅ {format_type.upper()}: Success")
                else:
                    st.error(f"❌ {format_type.upper()}: Failed")
            
            # Show sample flashcards
            st.subheader("📝 Sample Flashcards")
            
            # Load generated flashcards
            output_dir = result['output_dir']
            base_filename = result['base_filename']
            
            # Try to load CSV for display
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Show first few flashcards
                st.dataframe(
                    df[['Question', 'Answer', 'Type', 'Difficulty']].head(10),
                    use_container_width=True
                )
                
                # Download buttons
                st.subheader("💾 Download Files")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if os.path.exists(csv_path):
                        with open(csv_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                label="📊 Download CSV",
                                data=f.read(),
                                file_name=f"{base_filename}.csv",
                                mime="text/csv"
                            )
                
                with col2:
                    anki_path = os.path.join(output_dir, f"{base_filename}.anki")
                    if os.path.exists(anki_path):
                        with open(anki_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                label="📚 Download Anki",
                                data=f.read(),
                                file_name=f"{base_filename}.txt",
                                mime="text/plain"
                            )
                
                with col3:
                    json_path = os.path.join(output_dir, f"{base_filename}.json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            st.download_button(
                                label="📄 Download JSON",
                                data=f.read(),
                                file_name=f"{base_filename}.json",
                                mime="application/json"
                            )
            
            # Show summary
            with st.expander("📋 Processing Summary"):
                st.text(result.get('summary', 'No summary available'))
        
        else:
            st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    # Features section
    st.markdown("---")
    st.header("✨ Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h4>📚 PDF Processing</h4>
            <p>Extract text from any PDF document with advanced parsing capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h4>🧠 AI-Powered</h4>
            <p>Uses advanced NLP and transformer models to generate intelligent questions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h4>📤 Multiple Formats</h4>
            <p>Export to CSV, Anki, JSON, TXT, HTML, and PDF formats.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem;">
        <p>Made with ❤️ by FlashGenie Team</p>
        <p>Transform your learning materials into powerful flashcards!</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
