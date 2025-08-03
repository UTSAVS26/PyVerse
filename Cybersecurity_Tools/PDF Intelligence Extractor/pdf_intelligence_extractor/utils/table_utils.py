"""
Table extraction utilities for PDF Intelligence Extractor.
"""

# Try to import pandas, but don't fail if it's not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Table functionality may be limited.")

# Try to import pdfplumber
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not available. PDF table extraction may be limited.")

from typing import List, Dict, Any, Optional


class TableExtractor:
    """
    Enhanced table extraction utilities.
    """
    
    @staticmethod
    def extract_tables_from_pdf(pdf_path: str) -> List:
        """
        Extract tables from PDF and convert to pandas DataFrames.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List: List of extracted tables (DataFrames if pandas available, else raw data)
        """
        if not PDFPLUMBER_AVAILABLE:
            return []
        
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:  # At least header + one row
                            if PANDAS_AVAILABLE:
                                # Convert to DataFrame
                                df = pd.DataFrame(table[1:], columns=table[0])
                                tables.append(df)
                            else:
                                # Return raw table data
                                tables.append(table)
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
        
        return tables
    
    @staticmethod
    def clean_table_data(df):
        """
        Clean and normalize table data.
        
        Args:
            df: Input DataFrame or table data
            
        Returns:
            Cleaned data
        """
        if not PANDAS_AVAILABLE:
            # Simple cleaning for raw table data
            cleaned_table = []
            for row in df:
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                cleaned_table.append(cleaned_row)
            return cleaned_table
        
        # Remove empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Clean cell values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    @staticmethod
    def detect_table_type(df) -> str:
        """
        Detect the type of table based on its content.
        
        Args:
            df: Input DataFrame or table data
            
        Returns:
            str: Table type (invoice, resume, research, etc.)
        """
        # Convert to list of column names for analysis
        if PANDAS_AVAILABLE:
            columns = [col.lower() for col in df.columns]
        else:
            # For raw table data, use first row as headers
            columns = [str(cell).lower() for cell in df[0]] if df else []
        
        # Check for invoice-like patterns
        if any(col in ['item', 'description', 'quantity', 'price', 'amount'] 
               for col in columns):
            return 'invoice'
        
        # Check for resume-like patterns
        if any(col in ['education', 'experience', 'skills', 'company'] 
               for col in columns):
            return 'resume'
        
        # Check for research-like patterns
        if any(col in ['reference', 'citation', 'author', 'title'] 
               for col in columns):
            return 'research'
        
        return 'unknown'
    
    @staticmethod
    def extract_key_value_pairs(df) -> Dict[str, Any]:
        """
        Extract key-value pairs from a table.
        
        Args:
            df: Input DataFrame or table data
            
        Returns:
            Dict[str, Any]: Key-value pairs
        """
        pairs = {}
        
        if PANDAS_AVAILABLE:
            # If table has exactly 2 columns, treat as key-value
            if len(df.columns) == 2:
                for _, row in df.iterrows():
                    key = str(row.iloc[0]).strip()
                    value = str(row.iloc[1]).strip()
                    if key and value and key != 'nan' and value != 'nan':
                        pairs[key] = value
        else:
            # For raw table data
            if len(df) > 1 and len(df[0]) == 2:
                for row in df[1:]:  # Skip header
                    if len(row) >= 2:
                        key = str(row[0]).strip()
                        value = str(row[1]).strip()
                        if key and value and key != 'nan' and value != 'nan':
                            pairs[key] = value
        
        return pairs 