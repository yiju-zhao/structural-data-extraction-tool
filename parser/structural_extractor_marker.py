#!/usr/bin/env python3
"""
Generalized Structural Data Extraction Tool

This module provides a unified interface for extracting structured data from various file formats
including PDF, images, PPT, PPTX, DOC, DOCX, XLS, XLSX, HTML, and EPUB files.

Uses marker's ExtractionConverter with configurable LLM services for structured data extraction.

Supported LLM Services:
=====================

1. OpenAI (default)
   - Service: marker.services.openai.OpenAIService
   - Configuration: --openai-api-key, --openai-model, --openai-base-url
   - Environment variables: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL

2. Gemini (Google AI)
   - Service: marker.services.gemini.GoogleGeminiService (default Gemini service)
   - Configuration: --gemini-api-key
   - Environment variables: GOOGLE_API_KEY, GEMINI_API_KEY

3. Google Vertex AI
   - Service: marker.services.vertex.GoogleVertexService
   - Configuration: --vertex-project-id
   - Environment variables: GOOGLE_CLOUD_PROJECT, VERTEX_PROJECT_ID

4. Claude (Anthropic)
   - Service: marker.services.claude.ClaudeService
   - Configuration: --claude-api-key, --claude-model-name
   - Environment variables: ANTHROPIC_API_KEY, CLAUDE_API_KEY, CLAUDE_MODEL_NAME

5. Ollama (Local models)
   - Service: marker.services.ollama.OllamaService
   - Configuration: --ollama-base-url, --ollama-model
   - Environment variables: OLLAMA_BASE_URL, OLLAMA_MODEL

Usage Examples:
==============

# Using OpenAI (environment variable)
export OPENAI_API_KEY="your-key"
python structural_extractor.py -i /path/to/docs -o output.csv

# Using Gemini with command line
python structural_extractor.py -i /path/to/docs -o output.csv \\
  --llm-service marker.services.gemini.GoogleGeminiService \\
  --gemini-api-key "your-key"

# Using Vertex AI
python structural_extractor.py -i /path/to/docs -o output.csv \\
  --llm-service marker.services.vertex.GoogleVertexService \\
  --vertex-project-id "your-project-id"

# Using Claude
python structural_extractor.py -i /path/to/docs -o output.csv \\
  --llm-service marker.services.claude.ClaudeService \\
  --claude-api-key "your-key"

# Using Ollama (local)
python structural_extractor.py -i /path/to/docs -o output.csv \\
  --llm-service marker.services.ollama.OllamaService \\
  --ollama-base-url "http://localhost:11434" \\
  --ollama-model "llama2"
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv

from marker.converters.extraction import ExtractionConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Base schema for document metadata extraction"""
    title: Optional[str] = Field(description="Document title")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    orgnization: List[str] = Field(default_factory=list, description="List of orgnizations")



class ContentExtractor(BaseModel):
    """Schema for general content extraction"""
    content_type: str = Field(description="Type of content (text, table, image, etc.)")
    content: str = Field(description="Extracted content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class StructuralDataExtractor:
    """
    Generalized structural data extraction tool supporting multiple file formats
    
    Uses marker's ExtractionConverter with LLM services for structured data extraction.
    Supports PDF (primary), images, PPTX, DOCX, XLSX, HTML, and EPUB files.
    
    Note: Non-PDF formats require 'marker-pdf[full]' installation.
    """
    
    # Supported file extensions (based on marker documentation)
    # Marker supports: PDF, image, PPTX, DOCX, XLSX, HTML, EPUB files
    SUPPORTED_FORMATS = {
        '.pdf',           # PDF files - primary support
        '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif',  # Image files
        '.pptx',          # PowerPoint (requires marker-pdf[full])
        '.docx',          # Word documents (requires marker-pdf[full])
        '.xlsx',          # Excel files (requires marker-pdf[full])
        '.html', '.htm',  # HTML files (requires marker-pdf[full])
        '.epub'           # EPUB files (requires marker-pdf[full])
    }
    
    def __init__(
        self,
        llm_service: str = "marker.services.openai.OpenAIService",
        llm_config: Optional[Dict[str, Any]] = None,
        extraction_schema: Optional[Type[BaseModel]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        use_gpu: bool = False,
        max_pages: int = 0,
        cache_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the structural data extractor
        
        Args:
            llm_service: LLM service to use for extraction
            llm_config: Configuration for the LLM service
            extraction_schema: Pydantic schema for extraction
            output_dir: Directory to save extracted data
            use_gpu: Whether to use GPU for processing
            max_pages: Maximum pages to process (0 for all)
            cache_dir: Directory for caching results
        """
        self.llm_service = llm_service
        self.llm_config = llm_config or {}
        self.extraction_schema = extraction_schema or DocumentMetadata
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.use_gpu = use_gpu
        self.max_pages = max_pages
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./.cache")
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize converters
        self._init_converters()
        
    def _init_converters(self):
        """Initialize marker converters"""
        try:
            # Get schema for extraction
            schema = self.extraction_schema.model_json_schema()
            
            # Base configuration for ExtractionConverter
            base_config = {
                "page_schema": json.dumps(schema),  # Convert schema to JSON string
                "use_llm": True,  # Enable LLM for extraction
                **self.llm_config
            }
            
            # Add GPU and page range settings if specified
            if self.use_gpu:
                base_config["use_gpu"] = self.use_gpu
            if self.max_pages > 0:
                base_config["page_range"] = f"0-{self.max_pages-1}"
            
            # Set LLM service
            if self.llm_service:
                base_config["llm_service"] = self.llm_service
            
            # Create config parser
            self.config_parser = ConfigParser(base_config)
            
            # Create artifact dictionary with models
            artifact_dict = create_model_dict()
            
            # Initialize extraction converter with config containing LLM settings
            self.extraction_converter = ExtractionConverter(
                artifact_dict=artifact_dict,
                config=self.config_parser.generate_config_dict()
            )
            
            logger.info(f"Initialized ExtractionConverter with LLM service: {self.llm_service}")
            
        except Exception as e:
            logger.error(f"Error initializing converters: {str(e)}")
            raise
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file format is supported"""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def extract_from_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Extract structured data from a single file
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Extracted structured data or None if extraction failed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if not self._is_supported_file(file_path):
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return None
        
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Use extraction converter for structured extraction
            result = self.extraction_converter(str(file_path))
            
            if result:
                logger.info(f"Successfully extracted data from {file_path}")
                return result
            else:
                logger.warning(f"No data extracted from {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting from {file_path}: {str(e)}")
            return None
    
    def extract_from_directory(
        self, 
        input_dir: Union[str, Path],
        recursive: bool = True,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Extract structured data from all supported files in a directory
        
        Args:
            input_dir: Directory containing files to process
            recursive: Whether to search subdirectories
            batch_size: Number of files to process in each batch
            
        Returns:
            List of extracted data dictionaries
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.error(f"Directory not found: {input_path}")
            return []
        
        # Find all supported files
        supported_files = []
        
        if recursive:
            for ext in self.SUPPORTED_FORMATS:
                supported_files.extend(input_path.rglob(f"*{ext}"))
        else:
            for ext in self.SUPPORTED_FORMATS:
                supported_files.extend(input_path.glob(f"*{ext}"))
        
        if not supported_files:
            logger.warning(f"No supported files found in {input_path}")
            return []
        
        logger.info(f"Found {len(supported_files)} supported files to process")
        
        # Process files in batches
        results = []
        for i in range(0, len(supported_files), batch_size):
            batch = supported_files[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(supported_files) + batch_size - 1) // batch_size}")
            
            for file_path in batch:
                result = self.extract_from_file(file_path)
                if result:
                    # Add file metadata
                    result['source_file'] = str(file_path)
                    result['file_size'] = file_path.stat().st_size
                    result['file_type'] = file_path.suffix.lower()
                    result['extraction_timestamp'] = datetime.now().isoformat()
                    results.append(result)
        
        logger.info(f"Successfully processed {len(results)} files")
        return results
    
    def save_to_csv(
        self,
        extracted_data: List[Dict[str, Any]],
        output_file: Optional[Union[str, Path]] = None,
        flatten_lists: bool = True
    ) -> Optional[Path]:
        """
        Save extracted data to CSV format
        
        Args:
            extracted_data: List of extracted data dictionaries
            output_file: Output CSV file path
            flatten_lists: Whether to flatten list fields to strings
            
        Returns:
            Path to the saved CSV file
        """
        if not extracted_data:
            logger.warning("No data to save")
            return None
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"extracted_data_{timestamp}.csv"
        else:
            output_file = Path(output_file)
        
        # Ensure output directory exists
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Flatten data for CSV format
        flattened_data = []
        for item in extracted_data:
            flattened_item = {}
            for key, value in item.items():
                if isinstance(value, list) and flatten_lists:
                    # Convert lists to semicolon-separated strings
                    flattened_item[key] = "; ".join(str(v) for v in value) if value else ""
                elif isinstance(value, dict):
                    # Convert dicts to JSON strings
                    flattened_item[key] = json.dumps(value)
                else:
                    flattened_item[key] = str(value) if value is not None else ""
            flattened_data.append(flattened_item)
        
        # Get all unique field names
        all_fields = set()
        for item in flattened_data:
            all_fields.update(item.keys())
        
        # Sort fields for consistent output
        fieldnames = sorted(all_fields)
        
        # Write to CSV
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
            
            logger.info(f"Saved {len(flattened_data)} records to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            raise
    
    def save_to_json(
        self,
        extracted_data: List[Dict[str, Any]],
        output_file: Optional[Union[str, Path]] = None
    ) -> Optional[Path]:
        """
        Save extracted data to JSON format
        
        Args:
            extracted_data: List of extracted data dictionaries
            output_file: Output JSON file path
            
        Returns:
            Path to the saved JSON file
        """
        if not extracted_data:
            logger.warning("No data to save")
            return None
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"extracted_data_{timestamp}.json"
        else:
            output_file = Path(output_file)
        
        # Ensure output directory exists
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as jsonfile:
                json.dump(extracted_data, jsonfile, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(extracted_data)} records to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error saving to JSON: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return sorted(list(self.SUPPORTED_FORMATS))
    
    def get_file_stats(self, input_dir: Union[str, Path]) -> Dict[str, int]:
        """
        Get statistics about supported files in a directory
        
        Args:
            input_dir: Directory to analyze
            
        Returns:
            Dictionary with file format counts
        """
        input_path = Path(input_dir)
        stats = {}
        
        for ext in self.SUPPORTED_FORMATS:
            count = len(list(input_path.rglob(f"*{ext}")))
            if count > 0:
                stats[ext] = count
        
        return stats


# Convenience functions for common use cases

def extract_academic_papers(
    input_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    llm_service: str = "marker.services.openai.OpenAIService"
) -> Optional[Path]:
    """
    Extract metadata from academic papers (PDFs)
    
    Args:
        input_dir: Directory containing PDF files
        output_csv: Output CSV file path
        llm_service: LLM service to use
        
    Returns:
        Path to the output CSV file
    """
    # Read API keys and configuration from environment variables
    llm_config = {}
    
    if "openai" in llm_service.lower():
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm_config["openai_api_key"] = api_key
        # Optional OpenAI configurations
        model = os.getenv("OPENAI_MODEL")
        if model:
            llm_config["openai_model"] = model
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            llm_config["openai_base_url"] = base_url
            
    elif "claude" in llm_service.lower() or "anthropic" in llm_service.lower():
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if api_key:
            llm_config["claude_api_key"] = api_key
        # Optional Claude configurations
        model = os.getenv("CLAUDE_MODEL_NAME")
        if model:
            llm_config["claude_model_name"] = model
            
    elif "gemini" in llm_service.lower():
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            llm_config["gemini_api_key"] = api_key
            
    elif "vertex" in llm_service.lower():
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT_ID")
        if project_id:
            llm_config["vertex_project_id"] = project_id
            
    elif "ollama" in llm_service.lower():
        # Optional Ollama configurations
        base_url = os.getenv("OLLAMA_BASE_URL")
        if base_url:
            llm_config["ollama_base_url"] = base_url
        model = os.getenv("OLLAMA_MODEL")
        if model:
            llm_config["ollama_model"] = model
    
    extractor = StructuralDataExtractor(
        llm_service=llm_service,
        llm_config=llm_config,
        extraction_schema=DocumentMetadata
    )
    
    # Extract data
    results = extractor.extract_from_directory(input_dir)
    
    # Save to CSV
    return extractor.save_to_csv(results, output_csv)


def extract_general_content(
    input_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    llm_service: str = "marker.services.openai.OpenAIService"
) -> Optional[Path]:
    """
    Extract general content from various file formats
    
    Args:
        input_dir: Directory containing files
        output_csv: Output CSV file path
        llm_service: LLM service to use
        
    Returns:
        Path to the output CSV file
    """
    # Read API keys and configuration from environment variables
    llm_config = {}
    
    if "openai" in llm_service.lower():
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            llm_config["openai_api_key"] = api_key
        # Optional OpenAI configurations
        model = os.getenv("OPENAI_MODEL")
        if model:
            llm_config["openai_model"] = model
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            llm_config["openai_base_url"] = base_url
            
    elif "claude" in llm_service.lower() or "anthropic" in llm_service.lower():
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if api_key:
            llm_config["claude_api_key"] = api_key
        # Optional Claude configurations
        model = os.getenv("CLAUDE_MODEL_NAME")
        if model:
            llm_config["claude_model_name"] = model
            
    elif "gemini" in llm_service.lower():
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if api_key:
            llm_config["gemini_api_key"] = api_key
            
    elif "vertex" in llm_service.lower():
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEX_PROJECT_ID")
        if project_id:
            llm_config["vertex_project_id"] = project_id
            
    elif "ollama" in llm_service.lower():
        # Optional Ollama configurations
        base_url = os.getenv("OLLAMA_BASE_URL")
        if base_url:
            llm_config["ollama_base_url"] = base_url
        model = os.getenv("OLLAMA_MODEL")
        if model:
            llm_config["ollama_model"] = model
    
    extractor = StructuralDataExtractor(
        llm_service=llm_service,
        llm_config=llm_config,
        extraction_schema=ContentExtractor
    )
    
    # Extract data
    results = extractor.extract_from_directory(input_dir)
    
    # Save to CSV
    return extractor.save_to_csv(results, output_csv)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Structural Data Extraction Tool")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--output", "-o", help="Output CSV file")
    parser.add_argument("--llm-service", default="marker.services.openai.OpenAIService", help="LLM service")
    parser.add_argument("--schema", choices=["academic", "general"], default="academic", help="Extraction schema")
    parser.add_argument("--stats", action="store_true", help="Show file statistics")
    
    # LLM Service Configuration Arguments
    # OpenAI/OpenAI-compatible services
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    parser.add_argument("--openai-model", help="OpenAI model name")
    parser.add_argument("--openai-base-url", help="OpenAI base URL")
    
    # Claude/Anthropic
    parser.add_argument("--claude-api-key", help="Claude API key")
    parser.add_argument("--claude-model-name", help="Claude model name")
    
    # Gemini
    parser.add_argument("--gemini-api-key", help="Gemini API key")
    
    # Google Vertex
    parser.add_argument("--vertex-project-id", help="Google Cloud project ID for Vertex")
    
    # Ollama
    parser.add_argument("--ollama-base-url", help="Ollama base URL")
    parser.add_argument("--ollama-model", help="Ollama model name")
    
    args = parser.parse_args()
    
    # Build LLM configuration from command line arguments
    llm_config = {}
    
    # OpenAI configuration
    if args.openai_api_key:
        llm_config["openai_api_key"] = args.openai_api_key
    if args.openai_model:
        llm_config["openai_model"] = args.openai_model
    if args.openai_base_url:
        llm_config["openai_base_url"] = args.openai_base_url
    
    # Claude configuration
    if args.claude_api_key:
        llm_config["claude_api_key"] = args.claude_api_key
    if args.claude_model_name:
        llm_config["claude_model_name"] = args.claude_model_name
    
    # Gemini configuration
    if args.gemini_api_key:
        llm_config["gemini_api_key"] = args.gemini_api_key
    
    # Vertex configuration
    if args.vertex_project_id:
        llm_config["vertex_project_id"] = args.vertex_project_id
    
    # Ollama configuration
    if args.ollama_base_url:
        llm_config["ollama_base_url"] = args.ollama_base_url
    if args.ollama_model:
        llm_config["ollama_model"] = args.ollama_model
    
    if args.stats:
        extractor = StructuralDataExtractor()
        stats = extractor.get_file_stats(args.input)
        print("Supported file statistics:")
        for ext, count in stats.items():
            print(f"  {ext}: {count} files")
    else:
        # Use command line config if provided, otherwise fall back to convenience functions
        if llm_config:
            # Use manual configuration
            if args.schema == "academic":
                extraction_schema = DocumentMetadata
            else:
                extraction_schema = ContentExtractor
            
            extractor = StructuralDataExtractor(
                llm_service=args.llm_service,
                llm_config=llm_config,
                extraction_schema=extraction_schema
            )
            
            results = extractor.extract_from_directory(args.input)
            output_file = extractor.save_to_csv(results, args.output)
        else:
            # Use convenience functions with environment variables
            if args.schema == "academic":
                output_file = extract_academic_papers(args.input, args.output, args.llm_service)
            else:
                output_file = extract_general_content(args.input, args.output, args.llm_service)
        
        print(f"Extraction complete. Results saved to: {output_file}") 