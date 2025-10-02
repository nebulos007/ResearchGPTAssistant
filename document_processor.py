"""
Document Processing Module for ResearchGPT Assistant

TODO: Implement the following functionality:
1. PDF text extraction and cleaning
2. Text preprocessing and chunking
3. Basic similarity search using TF-IDF
4. Document metadata extraction
"""

import PyPDF2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

class DocumentProcessor:
    def __init__(self, config):
        """
        Initialize Document Processor
        
        TODO: 
        1. Store configuration
        2. Initialize TF-IDF vectorizer
        3. Create empty document storage
        """
        self.config = config
        # TODO: Initialize TfidfVectorizer with appropriate parameters
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )  # Initialize TfidfVectorizer here
        
        # TODO: Create document storage structure
        self.documents = {}  # Store as: {doc_id: {'title': '', 'chunks': [], 'metadata': {}}}
        self.document_vectors = None  # Store TF-IDF vectors
        self.chunk_to_doc_mapping = []  # Map chunk index to document ID
        self.all_chunks = []  # Store all text chunks for vectorization
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from PDF file
        
        TODO: Implement PDF text extraction using PyPDF2
        1. Open PDF file
        2. Extract text from all pages
        3. Clean extracted text (remove extra whitespace, special characters)
        4. Return cleaned text
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Extracted and cleaned text
        """
        
        from PyPDF2 import PdfReader

        # TODO: Implement PDF text extraction
        extracted_text = ""
        # Your implementation here
        try:
            # Open PDF file in binary mode
            with open('data/sample_papers', 'rb') as file:
                # Create PDF reader object
                reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                    
                # Clean the extracted text
                # Remove multiple newlines
                text = re.sub(r'\n\s*\n', '\n\n', text)
                # Replace multiple spaces with single space
                text = re.sub(r'\s+', ' ', text)
                # Normalize paragraph breaks
                text = re.sub(r'\n\n+', '\n\n', text)
                
                return text.strip()
                
        except Exception as e:
            print(f"Error extracting text from {'data/sample_papers'}: {str(e)}")
            return ""
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text
        
        TODO: Implement text preprocessing:
        1. Remove extra whitespace and newlines
        2. Fix common PDF extraction issues
        3. Remove special characters if needed
        4. Ensure text is properly formatted
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Preprocessed text
        """
        # TODO: Implement text preprocessing
        cleaned_text = text.lower()
        # Your implementation here
        # Remove special characters but keep punctuation that's meaningful
        # Keep periods, commas, question marks, exclamation marks
        cleaned_text = re.sub(r'[^\w\s\.\,\?\!\-\:\;\(\)]', ' ', cleaned_text)
        
        # Normalize whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Remove extra periods (common in PDFs)
        cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)
        
        # Clean up sentence spacing
        cleaned_text = re.sub(r'\s*\.\s*', '. ', cleaned_text)
        cleaned_text = re.sub(r'\s*\,\s*', ', ', cleaned_text)
        
        # Remove leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        return cleaned_text
    
    def chunk_text(self, text, chunk_size=None, overlap=None):
        """
        Split text into manageable chunks
        
        TODO: Implement text chunking:
        1. Use config chunk_size and overlap if not provided
        2. Split text into overlapping chunks
        3. Ensure chunks don't break in middle of sentences
        4. Return list of text chunks
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Size of each chunk
            overlap (int): Overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.config.CHUNK_SIZE
        if overlap is None:
            overlap = self.config.OVERLAP
            
        # TODO: Implement chunking logic
        chunks = []
        # Your implementation here
        if not text or len(text) < chunk_size:
            return [text] if text else []

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            # Calculate end position
            end = start + chunk_size
            # If this is not the last chunk, try to end at a sentence boundary
            if end < len(text):
                #Look for sentence ending in the last 200 characters of the chunk
                sentence_end_pattern = r'[.!?]\s+'
                search_start = max(end - 200, start)
                #Find sentence boundaries in the search range
                matches = list(re.finditer(sentence_end_pattern, text[search_start:end]))
                if matches:
                    #Use the last match to determine the end of the chunk
                    last_match = matches[-1]
                    end = search_start + last_match.end()
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # Move to the next chunk
            start = end - overlap
            # Ensure we don't get stuck in a loop
            if start <= 0 or start >= len(text):
                break
            
        self.logger.info(f"Successfully chunked text into {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def process_document(self, pdf_path):
        """
        Process a single PDF document
        
        TODO: Implement complete document processing:
        1. Extract text from PDF
        2. Preprocess the text
        3. Create chunks
        4. Extract basic metadata (title, length, etc.)
        5. Store in document storage
        6. Return document ID
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            str: Document ID
        """
        # TODO: Implement complete document processing pipeline
     
        # Your implementation here
        # Generate document ID from filename
        doc_id = os.path.basename(pdf_path).replace('.pdf', '')
        
        self.logger.info(f"Processing document: {pdf_path}")
        
        try:
            # Step 1: Extract text from PDF
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            if not raw_text:
                self.logger.warning(f"No text extracted from {pdf_path}")
                return doc_id
            
            # Step 2: Preprocess the text
            preprocessed_text = self.preprocess_text(raw_text)
            
            # Step 3: Create chunks
            chunks = self.chunk_text(preprocessed_text)
            
            # Step 4: Extract basic metadata
            metadata = self._extract_metadata(raw_text, pdf_path)
            
            # Step 5: Store in document storage
            self.documents[doc_id] = {
                'title': metadata.get('title', doc_id),
                'chunks': chunks,
                'metadata': metadata,
                'file_path': pdf_path,
                'raw_text_length': len(raw_text),
                'processed_text_length': len(preprocessed_text),
                'num_chunks': len(chunks)
            }
            
            self.logger.info(f"Successfully processed document {doc_id}")
            self.logger.info(f"  - Raw text length: {len(raw_text)}")
            self.logger.info(f"  - Processed text length: {len(preprocessed_text)}")
            self.logger.info(f"  - Number of chunks: {len(chunks)}")
        except Exception as e:
            self.logger.error(f"Error processing document {pdf_path}: {e}")
        return doc_id
    
    def build_search_index(self):
        """
        Build TF-IDF search index for all documents
        
        TODO: Implement search index creation:
        1. Collect all text chunks from all documents
        2. Fit TF-IDF vectorizer on all chunks
        3. Transform chunks to vectors
        4. Store vectors for similarity search
        """
        # TODO: Build TF-IDF index
        all_chunks = []
        # Your implementation here
        
    def find_similar_chunks(self, query, top_k=5):
        """
        Find most similar document chunks to query
        
        TODO: Implement similarity search:
        1. Transform query using fitted TF-IDF vectorizer
        2. Calculate cosine similarity with all chunks
        3. Return top_k most similar chunks with scores
        
        Args:
            query (str): Search query
            top_k (int): Number of similar chunks to return
            
        Returns:
            list: List of (chunk_text, similarity_score, doc_id) tuples
        """
        # TODO: Implement similarity search
        similar_chunks = []
        # Your implementation here
        return similar_chunks
    
    def get_document_stats(self):
        """
        Get statistics about processed documents
        
        TODO: Return dictionary with:
        1. Number of documents processed
        2. Total chunks created
        3. Average document length
        4. List of document titles
        """
        # TODO: Calculate and return document statistics
        stats = {}
        # Your implementation here
        return stats
