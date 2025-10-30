#!/usr/bin/env python3
"""
Advanced Web Content Extraction Tool

This tool uses a two-pass extraction strategy to accurately extract structured data from web pages.
Prevents information mixing between entities and maintains data integrity through intelligent chunking.

Author: Assistant
License: MIT
"""

import asyncio
import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import nltk
import openai
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
)
from crawl4ai.chunking_strategy import ChunkingStrategy
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from dotenv import load_dotenv
from nltk.tokenize import TextTilingTokenizer
from pydantic import BaseModel, Field, create_model

# Import PDF converter utility
try:
    from utility.pdf_to_html_converter import (
        PDFToHTMLConverter,
        is_pdf_file,
        convert_pdf_for_crawl4ai,
    )

    PDF_SUPPORT_AVAILABLE = True
except ImportError:
    PDF_SUPPORT_AVAILABLE = False

    # Define dummy functions to prevent import errors
    def is_pdf_file(file_path: str) -> bool:
        return False

    def convert_pdf_for_crawl4ai(pdf_path: str, output_dir=None, config=None) -> str:
        raise ImportError("PDF support not available")


# Initialize environment and required packages
load_dotenv()
try:
    nltk.download("stopwords", quiet=True)
except Exception:
    pass  # Ignore download errors


# ================================
# CONSTANTS
# ================================
class Constants:
    """Application constants"""

    # Default values
    DEFAULT_CHUNK_SIZE = 2500
    DEFAULT_OVERLAP_SIZE = 300
    DEFAULT_SIMILARITY_THRESHOLD = 0.8
    MIN_CHUNK_SIZE = 500
    MIN_QUALITY_FIELD_RATIO = 1 / 3

    # Text patterns
    EMPTY_VALUES = {"N/A", "NULL", "NONE", ""}

    # File extensions
    CSV_EXTENSION = ".csv"

    # Model configurations
    AVAILABLE_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4.1",
        "gpt-3.5-turbo",
        "o1-mini",
        "o1-preview",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
    ]


# ================================
# EXCEPTIONS
# ================================
class ExtractionError(Exception):
    """Base exception for extraction-related errors"""

    pass


class ModelConfigError(ExtractionError):
    """Exception for model configuration errors"""

    pass


class StructureAnalysisError(ExtractionError):
    """Exception for structure analysis errors"""

    pass


class DataProcessingError(ExtractionError):
    """Exception for data processing errors"""

    pass


# ================================
# CONFIGURATION CLASSES
# ================================
class ModelConfig:
    """Configuration for different LLM models used in various tasks"""

    # Default models for different tasks
    structure_analysis_model: str = "o4-mini"
    instruction_generation_model: str = "gpt-5"
    data_extraction_model: str = "gpt-4.1-mini"

    # Temperature settings for different tasks
    structure_analysis_temperature: float = 1.0
    instruction_generation_temperature: float = 1.0
    data_extraction_temperature: float = 0

    @classmethod
    def set_models(
        cls,
        structure_model: Optional[str] = None,
        instruction_model: Optional[str] = None,
        extraction_model: Optional[str] = None,
    ) -> None:
        """Set models for different tasks"""
        if structure_model:
            cls._validate_model(structure_model)
            cls.structure_analysis_model = structure_model
        if instruction_model:
            cls._validate_model(instruction_model)
            cls.instruction_generation_model = instruction_model
        if extraction_model:
            cls._validate_model(extraction_model)
            cls.data_extraction_model = extraction_model

    @classmethod
    def _validate_model(cls, model_name: str) -> None:
        """Validate if model is available"""
        if model_name not in Constants.AVAILABLE_MODELS:
            raise ModelConfigError(
                f"Model '{model_name}' not available. "
                f"Available models: {Constants.AVAILABLE_MODELS}"
            )

    @classmethod
    def get_model_info(cls) -> Dict[str, str]:
        """Get current model configuration"""
        return {
            "structure_analysis": cls.structure_analysis_model,
            "instruction_generation": cls.instruction_generation_model,
            "data_extraction": cls.data_extraction_model,
        }


class ExtractionConfig:
    """Configuration for extraction parameters"""

    def __init__(
        self,
        url: str,
        schema_fields: List[str],
        output_file: str,
        strategy: str = "two-pass",
        chunk_size: int = Constants.DEFAULT_CHUNK_SIZE,
        overlap_size: int = Constants.DEFAULT_OVERLAP_SIZE,
        similarity_threshold: float = Constants.DEFAULT_SIMILARITY_THRESHOLD,
        primary_keys: Optional[List[str]] = None,
        no_dedup: bool = False,
        verbose: bool = False,
        pdf_config: Optional[Dict] = None,
    ):
        self.original_input = url  # Store original input (could be URL or file path)
        self.url = url  # Will be updated if PDF conversion is needed
        self.schema_fields = schema_fields
        self.output_file = output_file
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold
        self.primary_keys = primary_keys or self._determine_primary_keys()
        self.no_dedup = no_dedup
        self.verbose = verbose
        self.pdf_config = pdf_config or {}
        self.is_pdf_input = False
        self.converted_html_path = None

        self._validate_and_process_input()

    def _validate_and_process_input(self) -> None:
        """Validate configuration parameters and process PDF input if needed"""
        # Check if input is a PDF file
        if is_pdf_file(self.original_input):
            if not PDF_SUPPORT_AVAILABLE:
                raise ExtractionError(
                    f"PDF file detected: {self.original_input}\n"
                    "PDF support is not available. Please install marker-pdf:\n"
                    "pip install marker-pdf"
                )
            self._handle_pdf_input()
        elif not self.url.startswith(("http://", "https://", "file://")):
            # Check if it looks like a file path but isn't a PDF
            if os.path.exists(self.original_input):
                raise ValueError(f"File detected but not a PDF: {self.original_input}")
            else:
                raise ValueError(
                    f"Invalid input: {self.original_input}\n"
                    "Input must be:\n"
                    "  - A URL (http:// or https://)\n"
                    "  - A PDF file path (.pdf extension)\n"
                    "  - A local HTML file (file:// URL)"
                )

        if self.chunk_size < Constants.MIN_CHUNK_SIZE:
            raise ValueError(f"Chunk size must be at least {Constants.MIN_CHUNK_SIZE}")

        if self.overlap_size >= self.chunk_size:
            raise ValueError("Overlap size must be smaller than chunk size")

        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")

        if not self.schema_fields:
            raise ValueError("At least one schema field must be specified")

    def _handle_pdf_input(self) -> None:
        """Handle PDF input by converting to HTML"""
        if not PDF_SUPPORT_AVAILABLE:
            raise ExtractionError(
                "PDF support is not available. Please install marker-pdf: pip install marker-pdf"
            )

        self.is_pdf_input = True

        try:
            # Create output directory for converted HTML
            output_dir = Path(self.output_file).parent / "converted_pdfs"
            output_dir.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                print(f"📄 PDF detected: {self.original_input}")
                print(f"🔄 Converting PDF to HTML using marker library...")

            # Convert PDF to HTML
            file_url = convert_pdf_for_crawl4ai(
                self.original_input, str(output_dir), self.pdf_config
            )

            self.url = file_url
            self.converted_html_path = file_url.replace("file://", "")

            if self.verbose:
                print(f"✅ PDF converted successfully!")
                print(f"🌐 Processing HTML at: {file_url}")

        except Exception as e:
            raise ExtractionError(f"Failed to convert PDF: {e}")

    def _determine_primary_keys(self) -> Optional[List[str]]:
        """Determine primary keys for deduplication based on schema fields"""
        # Return None if no primary keys should be auto-determined
        # This makes deduplication optional unless explicitly specified
        return None

    def cleanup(self) -> None:
        """Clean up temporary files if needed"""
        if self.converted_html_path and Path(self.converted_html_path).exists():
            try:
                Path(self.converted_html_path).unlink()
                if self.verbose:
                    print(
                        f"🧹 Cleaned up temporary HTML file: {self.converted_html_path}"
                    )
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Could not clean up temporary file: {e}")


# ================================
# DATA MODELS
# ================================
class StructureInfo(BaseModel):
    """Model for structure analysis results"""

    separator_pattern: str = Field(description="Pattern that separates entities")
    entity_indicators: List[str] = Field(
        description="Indicators that mark the start of new entities"
    )
    content_structure: str = Field(
        description="Description of how content is organized"
    )
    field_patterns: Dict[str, str] = Field(
        description="Patterns for identifying specific fields"
    )


def create_dynamic_model(
    schema_fields: List[str], model_name: str = "DynamicModel"
) -> BaseModel:
    """Create a dynamic Pydantic model based on user-provided schema fields"""
    field_definitions = {}
    for field in schema_fields:
        field_definitions[field] = (str, Field(description=f"The {field} field"))

    return create_model(model_name, **field_definitions)


# ================================
# CHUNKING STRATEGIES
# ================================
class BaseChunkingStrategy:
    """Base class for chunking strategies"""

    def __init__(self, max_chunk_size: int = Constants.DEFAULT_CHUNK_SIZE):
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> List[str]:
        """Abstract method to chunk text"""
        raise NotImplementedError


class StructureAwareChunking(ChunkingStrategy):
    """Chunking strategy that uses structure information from first pass"""

    def __init__(
        self,
        structure_info: StructureInfo,
        max_chunk_size: int = Constants.DEFAULT_CHUNK_SIZE,
        overlap_size: int = Constants.DEFAULT_OVERLAP_SIZE,
    ):
        super().__init__()
        self.structure_info = structure_info
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def chunk(self, text: str) -> List[str]:
        """Chunk text based on identified structure patterns"""
        if not text or not text.strip():
            return []

        try:
            boundaries = self._find_entity_boundaries(text)

            if not boundaries:
                return self._fallback_chunking(text)

            chunks = self._create_chunks_from_boundaries(text, boundaries)
            return self._add_overlap(chunks)

        except Exception as e:
            logging.warning(f"Error in structure-aware chunking: {e}")
            return self._fallback_chunking(text)

    def _find_entity_boundaries(self, text: str) -> List[int]:
        """Find entity boundaries using structure patterns"""
        boundaries = [0]  # Always start at beginning

        # Use separator pattern from structure analysis
        if self.structure_info.separator_pattern:
            pattern = self.structure_info.separator_pattern
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            boundaries.extend([match.start() for match in matches])

        # Use entity indicators
        for indicator in self.structure_info.entity_indicators:
            matches = re.finditer(indicator, text, re.MULTILINE | re.IGNORECASE)
            boundaries.extend([match.start() for match in matches])

        return sorted(set(boundaries))

    def _create_chunks_from_boundaries(
        self, text: str, boundaries: List[int]
    ) -> List[str]:
        """Create chunks from identified boundaries"""
        chunks = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            entity_content = text[start:end].strip()

            if len(entity_content) > self.max_chunk_size:
                sub_chunks = self._split_large_entity(entity_content)
                chunks.extend(sub_chunks)
            else:
                chunks.append(entity_content)

        return chunks

    def _split_large_entity(self, entity_content: str) -> List[str]:
        """Split large entity while preserving logical boundaries"""
        paragraphs = entity_content.split("\n\n")
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if (
                len(current_chunk) + len(paragraph) > self.max_chunk_size
                and current_chunk
            ):
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks to preserve context"""
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - add preview from next chunk
                if i + 1 < len(chunks):
                    next_preview = chunks[i + 1][: self.overlap_size]
                    overlapped_chunks.append(
                        f"{chunk}\n\n--- PREVIEW ---\n{next_preview}"
                    )
                else:
                    overlapped_chunks.append(chunk)
            elif i == len(chunks) - 1:
                # Last chunk - add context from previous chunk
                prev_context = chunks[i - 1][-self.overlap_size :]
                overlapped_chunks.append(f"--- CONTEXT ---\n{prev_context}\n\n{chunk}")
            else:
                # Middle chunks - add context from both sides
                prev_context = chunks[i - 1][-self.overlap_size :]
                next_preview = chunks[i + 1][: self.overlap_size]
                overlapped_chunks.append(
                    f"--- CONTEXT ---\n{prev_context}\n\n{chunk}\n\n--- PREVIEW ---\n{next_preview}"
                )

        return overlapped_chunks

    def _fallback_chunking(self, text: str) -> List[str]:
        """Fallback to simple paragraph-based chunking"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if (
                len(current_chunk) + len(paragraph) > self.max_chunk_size
                and current_chunk
            ):
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


class TopicSegmentationChunking(ChunkingStrategy):
    """Text tiling based topic segmentation chunking strategy"""

    def __init__(self, w: int = 20, k: int = 10, similarity_method: str = "cosine"):
        super().__init__()
        self.w = w
        self.k = k
        self.similarity_method = similarity_method
        self._initialize_tokenizer()

    def _initialize_tokenizer(self) -> None:
        """Initialize the text tiling tokenizer"""
        try:
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            self.tokenizer = TextTilingTokenizer(
                w=self.w, k=self.k, similarity_method=self.similarity_method
            )
        except ImportError as e:
            raise ExtractionError(
                "NLTK is required for TopicSegmentationChunking. "
                "Install it with: pip install nltk"
            ) from e

    def chunk(self, text: str) -> List[str]:
        """Chunk text using TextTiling algorithm"""
        if not text or not text.strip():
            return []

        try:
            segments = self.tokenizer.tokenize(text)
            chunks = [segment.strip() for segment in segments if segment.strip()]

            if not chunks:
                chunks = [text.strip()]

            return chunks

        except Exception as e:
            logging.warning(f"Error in topic segmentation chunking: {e}")
            # Fallback to simple paragraph splitting
            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            return chunks if chunks else [text.strip()]


# ================================
# EXTRACTION SERVICE
# ================================
class ExtractionService:
    """Service class for handling web content extraction"""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        if self.config.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def extract(self) -> int:
        """Main extraction method"""
        try:
            self._validate_environment()

            # Show input type info
            if self.config.is_pdf_input:
                print(f"📄 Processing PDF: {self.config.original_input}")
            else:
                print(f"🌐 Processing URL: {self.config.url}")

            if self.config.strategy == "two-pass":
                result = await self._two_pass_extraction()
            else:
                result = await self._single_pass_extraction()

            return result

        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            return 1
        finally:
            # Cleanup temporary files if needed
            if hasattr(self.config, "cleanup"):
                self.config.cleanup()

    def _validate_environment(self) -> None:
        """Validate environment setup"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ExtractionError("OPENAI_API_KEY environment variable is not set")

    async def _two_pass_extraction(self) -> int:
        """Execute two-pass extraction strategy"""
        print("🚀 Starting two-pass extraction strategy...")

        # Pass 1: Structure Analysis
        structure_info = await self._analyze_content_structure()

        # Pass 2: Structure-aware extraction
        print("📊 Starting structure-aware data extraction...")

        chunker = self._create_chunker(structure_info)
        return await self._extract_with_chunker(chunker, structure_info)

    async def _single_pass_extraction(self) -> int:
        """Execute single-pass extraction strategy"""
        print("⚡ Starting single-pass extraction strategy...")

        chunker = TopicSegmentationChunking(w=30, k=15, similarity_method="cosine")
        print("✅ Using topic segmentation chunking")

        return await self._extract_with_chunker(chunker, None)

    async def _analyze_content_structure(self) -> Optional[StructureInfo]:
        """Analyze content structure to understand organization patterns"""
        structure_instruction = await self._get_structure_analysis_instruction()

        try:
            browser_cfg = BrowserConfig(headless=True)

            strategy = LLMExtractionStrategy(
                llm_config=LLMConfig(
                    provider=f"openai/{ModelConfig.structure_analysis_model}",
                    api_token=os.getenv("OPENAI_API_KEY"),
                ),
                schema=StructureInfo.model_json_schema(),
                extraction_type="schema",
                instruction=structure_instruction,
                apply_chunking=False,
                input_format="markdown",
                extra_args={"temperature": ModelConfig.structure_analysis_temperature},
            )

            crawl_config = CrawlerRunConfig(
                extraction_strategy=strategy, cache_mode=CacheMode.BYPASS
            )

            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                print("🔍 Analyzing content structure...")
                result = await crawler.arun(url=self.config.url, config=crawl_config)

                if result.success and result.extracted_content:
                    return self._parse_structure_result(result.extracted_content)
                else:
                    self.logger.warning(
                        f"Structure analysis failed: {result.error_message if result else 'Unknown error'}"
                    )
                    return None

        except Exception as e:
            self.logger.warning(f"Error in structure analysis: {e}")
            return None

    async def _get_structure_analysis_instruction(self) -> str:
        """Generate contextual instruction for structure analysis using LLM"""

        # Determine content type context
        content_context = ""
        if self.config.is_pdf_input:
            content_context = f"PDF document: {Path(self.config.original_input).name}"
        else:
            content_context = f"Web page: {self.config.url}"

        # Create a direct, clear instruction for structure analysis
        instruction = f"""
        TASK: Analyze the content structure to understand how information is organized for extracting these fields: {", ".join(self.config.schema_fields)}

        You must return a JSON object with exactly these fields:
        - separator_pattern: A regex pattern that separates individual records/entities
        - entity_indicators: A list of text patterns that indicate the start of new entities
        - content_structure: A description of how the content is organized
        - field_patterns: A dictionary mapping field names to patterns for finding them

        DO NOT extract actual data. Instead, analyze HOW the data is structured.

        For the content from {content_context}, identify:

        1. **separator_pattern**: What pattern separates different records? (e.g., "\\n\\n", "<div", "---", etc.)
        2. **entity_indicators**: What text patterns signal a new entity? (e.g., ["Paper:", "Title:", "Session", etc.])
        3. **content_structure**: How is the content organized? (e.g., "Listed items with titles followed by details")
        4. **field_patterns**: For each field {", ".join(self.config.schema_fields)}, what pattern helps identify it?

        Example output format:
        {{
            "separator_pattern": "\\n\\n|<div",
            "entity_indicators": ["Paper:", "Title:", "Session:"],
            "content_structure": "Content organized as sequential items with titles and metadata",
            "field_patterns": {{
                "title": "Title:|Paper title:",
                "author": "Author:|By:",
                "time": "Time:|When:"
            }}
        }}

        Focus on STRUCTURE ANALYSIS, not data extraction.
        """

        if self.config.verbose:
            print(f"📝 Using direct structure analysis instruction")

        return instruction

    def _get_fallback_instruction(self) -> str:
        """Fallback instruction for structure analysis"""
        return f"""
        TASK: Analyze the content structure to understand how information is organized for extracting these fields: {", ".join(self.config.schema_fields)}

        You must return a JSON object with exactly these fields:
        - separator_pattern: A regex pattern that separates individual records/entities
        - entity_indicators: A list of text patterns that indicate the start of new entities
        - content_structure: A description of how the content is organized
        - field_patterns: A dictionary mapping field names to patterns for finding them

        DO NOT extract actual data. Instead, analyze HOW the data is structured.

        Focus on identifying:
        1. **separator_pattern**: What pattern separates different records? (e.g., "\\n\\n", "<div", "---")
        2. **entity_indicators**: What text patterns signal a new entity? (e.g., ["Paper:", "Title:", "Session"])
        3. **content_structure**: How is the content organized? 
        4. **field_patterns**: For each field {", ".join(self.config.schema_fields)}, what pattern helps identify it?

        Return valid JSON with these exact field names. Focus on STRUCTURE ANALYSIS, not data extraction.
        """

    def _parse_structure_result(
        self, extracted_content: str
    ) -> Optional[StructureInfo]:
        """Parse structure analysis result"""
        try:
            structure_data = json.loads(extracted_content)
            if isinstance(structure_data, list) and len(structure_data) > 0:
                structure_data = structure_data[0]

            # Check if we got the expected structure analysis fields
            required_fields = [
                "separator_pattern",
                "entity_indicators",
                "content_structure",
                "field_patterns",
            ]
            if not all(field in structure_data for field in required_fields):
                if self.config.verbose:
                    print(f"⚠️  Structure analysis returned unexpected format:")
                    print(f"   Expected fields: {required_fields}")
                    print(f"   Received fields: {list(structure_data.keys())}")
                    print(f"   Raw response: {extracted_content[:300]}...")
                self.logger.warning(
                    "Structure analysis returned unexpected format - falling back to topic segmentation"
                )
                return None

            structure_info = StructureInfo(**structure_data)
            print(f"✅ Structure analysis complete:")
            if self.config.verbose:
                print(f"   - Separator pattern: {structure_info.separator_pattern}")
                print(f"   - Entity indicators: {structure_info.entity_indicators}")
                print(f"   - Content structure: {structure_info.content_structure}")
                print(f"   - Field patterns: {structure_info.field_patterns}")

            return structure_info

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in structure analysis: {e}")
            if self.config.verbose:
                print(
                    f"   Raw response (first 300 chars): {extracted_content[:300]}..."
                )
            return None
        except Exception as e:
            self.logger.error(f"Error parsing structure analysis: {e}")
            if self.config.verbose:
                print(f"   Raw response: {extracted_content[:500]}...")
            return None

    def _create_chunker(
        self, structure_info: Optional[StructureInfo]
    ) -> ChunkingStrategy:
        """Create appropriate chunking strategy"""
        if structure_info:
            chunker = StructureAwareChunking(
                structure_info,
                max_chunk_size=self.config.chunk_size,
                overlap_size=self.config.overlap_size,
            )
            print("✅ Using structure-aware chunking")
        else:
            chunker = TopicSegmentationChunking(w=30, k=15, similarity_method="cosine")
            print("⚠️  Falling back to topic segmentation chunking")

        return chunker

    async def _extract_with_chunker(
        self, chunker: ChunkingStrategy, structure_info: Optional[StructureInfo]
    ) -> int:
        """Extract data using specified chunker"""
        dynamic_model = create_dynamic_model(self.config.schema_fields, "ExtractedData")
        instruction = await self._generate_extraction_instruction(structure_info)

        strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider=f"openai/{ModelConfig.data_extraction_model}",
                api_token=os.getenv("OPENAI_API_KEY"),
            ),
            schema=dynamic_model.model_json_schema(),
            extraction_type="schema",
            instruction=instruction,
            chunking_strategy=chunker,
            apply_chunking=True,
            input_format="markdown",
            extra_args={"temperature": ModelConfig.data_extraction_temperature},
        )

        crawl_config = CrawlerRunConfig(
            extraction_strategy=strategy, cache_mode=CacheMode.BYPASS
        )

        browser_cfg = BrowserConfig(headless=True)

        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            print(f"🔄 Extracting data from {self.config.url}...")
            result = await crawler.arun(url=self.config.url, config=crawl_config)

            if result.success:
                data = json.loads(result.extracted_content)
                if self.config.verbose:
                    print(
                        f"📋 Raw extraction results: {len(data) if isinstance(data, list) else 'Not a list'}"
                    )

                processor = DataProcessor(self.config, structure_info)
                success = await processor.save_results(data, dynamic_model)

                if success:
                    if self.config.verbose:
                        strategy.show_usage()
                    return 0
                else:
                    return 1
            else:
                print(f"❌ Extraction failed: {result.error_message}")
                return 1

    async def _generate_extraction_instruction(
        self, structure_info: Optional[StructureInfo]
    ) -> str:
        """Generate enhanced extraction instruction"""
        structure_context = ""
        if structure_info:
            structure_context = f"""
            
            IMPORTANT STRUCTURAL INFORMATION:
            - Content structure: {structure_info.content_structure}
            - Entities are separated by: {structure_info.separator_pattern}
            - New entities are indicated by: {", ".join(structure_info.entity_indicators)}
            - Field patterns: {structure_info.field_patterns}
            
            Use this structural information to ensure you don't mix data between different entities.
            """

        prompt = f"""
        Create a comprehensive extraction instruction for schema fields: {", ".join(self.config.schema_fields)}
        
        {structure_context}
        
        The instruction must emphasize:

        1. **ENTITY INTEGRITY**: Each record must contain information that logically belongs together. 
           - Before extracting any field, identify which entity/item it belongs to
           - Never mix information from different entities into the same record

        2. **BOUNDARY RECOGNITION**: 
           - Pay careful attention to separators and boundaries between entities
           - Use the structural patterns identified to maintain proper boundaries

        3. **CONTEXTUAL VALIDATION**:
           - For each extracted record, verify that all fields make logical sense together

        4. **COMPLETENESS WITH ACCURACY**:
           - Extract ALL entities from the content
           - For missing fields, use 'N/A' rather than guessing

        5. **FIELD-SPECIFIC GUIDELINES**:
           {chr(10).join([f"   - {field}: Extract exactly as it appears, maintaining formatting and completeness" for field in self.config.schema_fields])}

        Generate a detailed instruction that will prevent information mixing and ensure accurate entity-based extraction.
        """

        try:
            client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model=ModelConfig.instruction_generation_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating detailed extraction instructions for web content parsing with focus on maintaining entity integrity.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=ModelConfig.instruction_generation_temperature,
            )

            instruction = response.choices[0].message.content.strip()
            print(
                f"📝 Enhanced extraction instruction generated ({len(instruction)} chars)"
            )
            return instruction

        except Exception as e:
            self.logger.warning(f"Error generating enhanced instruction: {e}")
            return f"""
            Extract all records with the following fields: {", ".join(self.config.schema_fields)}.
            
            CRITICAL: Maintain entity integrity - never mix information from different entities.
            Each record must contain fields that belong to the same entity/item.
            
            {structure_context}
            
            For missing fields, use 'N/A'. Extract ALL entities, not just the first few.
            """


# ================================
# DATA PROCESSING
# ================================
class DataProcessor:
    """Handle post-processing and saving of extracted data"""

    def __init__(
        self, config: ExtractionConfig, structure_info: Optional[StructureInfo]
    ):
        self.config = config
        self.structure_info = structure_info

    async def save_results(self, data: Any, model_class: BaseModel) -> bool:
        """Save extraction results with enhanced post-processing"""
        try:
            if not self._validate_data(data):
                return self._create_empty_csv(model_class)

            processed_data = self._process_data(data, model_class)
            self._write_csv(processed_data, model_class)
            self._print_statistics(len(data), len(processed_data))

            return True

        except Exception as e:
            logging.error(f"Error saving results: {e}")
            return False

    def _validate_data(self, data: Any) -> bool:
        """Validate extracted data"""
        if not data or not isinstance(data, list) or len(data) == 0:
            print("⚠️  No data to save or data format is incorrect")
            return False
        return True

    def _create_empty_csv(self, model_class: BaseModel) -> bool:
        """Create empty CSV file"""
        try:
            fieldnames = list(model_class.model_fields.keys())
            with open(
                self.config.output_file, "w", newline="", encoding="utf-8"
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            print(f"📄 Empty CSV file created: {self.config.output_file}")
            return False
        except Exception as e:
            logging.error(f"Error creating empty CSV: {e}")
            return False

    def _process_data(self, data: List[Dict], model_class: BaseModel) -> List[Dict]:
        """Process extracted data through enhancement pipeline"""
        fieldnames = list(model_class.model_fields.keys())

        if self.config.verbose:
            print("🔧 Post-processing extracted data...")

        # 1. Filter and normalize fields
        filtered_data = self._filter_fields(data, fieldnames)

        # 2. Remove empty rows
        non_empty_data = self._remove_empty_rows(filtered_data)

        # 3. Quality validation (removed automatic duplicate removal)
        valid_data = self._validate_quality(non_empty_data, fieldnames)

        return valid_data

    def _filter_fields(self, data: List[Dict], fieldnames: List[str]) -> List[Dict]:
        """Filter and normalize fields"""
        filtered_data = []
        for item in data:
            if isinstance(item, dict):
                filtered_item = {
                    key: value for key, value in item.items() if key in fieldnames
                }
                # Ensure all required fields are present
                for field in fieldnames:
                    if field not in filtered_item:
                        filtered_item[field] = "N/A"
                filtered_data.append(filtered_item)

        if self.config.verbose:
            print(f"   ✅ Field filtering: {len(filtered_data)} records")

        return filtered_data

    def _remove_empty_rows(self, data: List[Dict]) -> List[Dict]:
        """Remove entirely empty rows"""

        def is_empty_row(row: Dict) -> bool:
            for value in row.values():
                if value and str(value).strip() and str(value).strip().upper() != "N/A":
                    return False
            return True

        non_empty_data = [row for row in data if not is_empty_row(row)]

        if self.config.verbose:
            print(f"   ✅ Empty row removal: {len(non_empty_data)} records")

        return non_empty_data

    def _validate_quality(self, data: List[Dict], fieldnames: List[str]) -> List[Dict]:
        """Validate record quality"""
        valid_data = []
        min_fields = max(1, int(len(fieldnames) * Constants.MIN_QUALITY_FIELD_RATIO))

        for record in data:
            non_empty_fields = sum(
                1
                for value in record.values()
                if value and str(value).strip() and str(value).strip().upper() != "N/A"
            )

            if non_empty_fields >= min_fields:
                valid_data.append(record)
            elif self.config.verbose:
                print(f"   ⚠️  Low quality record filtered: {record}")

        if self.config.verbose:
            print(f"   ✅ Quality validation: {len(valid_data)} records")

        return valid_data

    def _write_csv(self, data: List[Dict], model_class: BaseModel) -> None:
        """Write data to CSV file and create deduplicated version"""
        fieldnames = list(model_class.model_fields.keys())

        # Write original CSV file
        with open(
            self.config.output_file, "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        # Create deduplicated version (unless disabled)
        if not self.config.no_dedup and self.config.primary_keys:
            deduplicated_data = self._simple_deduplication(data, fieldnames)
            if len(deduplicated_data) < len(data):
                # Create deduplicated filename
                base_name = os.path.splitext(self.config.output_file)[0]
                dedup_file = f"{base_name}_deduplicated.csv"

                with open(dedup_file, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(deduplicated_data)

                if self.config.verbose:
                    print(f"📄 Deduplicated file created: {dedup_file}")
                    print(
                        f"   Primary keys used: {', '.join(self.config.primary_keys)}"
                    )
                    print(
                        f"   Removed {len(data) - len(deduplicated_data)} duplicate records"
                    )
            elif self.config.verbose:
                print(
                    f"✅ No duplicates found based on primary keys: {', '.join(self.config.primary_keys)}"
                )
        elif not self.config.primary_keys and self.config.verbose:
            print("ℹ️  Deduplication skipped (no primary keys specified)")
        elif self.config.verbose:
            print("ℹ️  Deduplication skipped (--no-dedup flag used)")

    def _simple_deduplication(
        self, data: List[Dict], fieldnames: List[str]
    ) -> List[Dict]:
        """Simple deduplication using pandas drop_duplicates() based on primary keys"""
        import pandas as pd

        # Convert data to DataFrame
        df = pd.DataFrame(data)

        # Get valid primary keys that exist in the data
        valid_primary_keys = [
            key for key in self.config.primary_keys if key in fieldnames
        ]

        if not valid_primary_keys:
            if self.config.verbose:
                print(
                    f"   ⚠️  No valid primary keys found in data. Available fields: {fieldnames}"
                )
            return data

        # Remove duplicates based on primary keys, keeping the last occurrence
        deduplicated_df = df.drop_duplicates(subset=valid_primary_keys, keep="last")

        if self.config.verbose:
            print(f"   🔧 Deduplication using keys: {valid_primary_keys}")
            print(
                f"   📊 Before: {len(df)} records, After: {len(deduplicated_df)} records"
            )

        # Convert back to list of dictionaries
        return deduplicated_df.to_dict("records")

    def _print_statistics(self, original_count: int, final_count: int) -> None:
        """Print extraction statistics"""
        print(f"💾 Results saved to {self.config.output_file}")

        # Check if deduplicated file exists
        base_name = os.path.splitext(self.config.output_file)[0]
        dedup_file = f"{base_name}_deduplicated.csv"
        dedup_exists = os.path.exists(dedup_file) and not self.config.no_dedup

        print(f"📊 Final statistics:")
        print(f"   - Total records extracted: {original_count}")
        print(f"   - After processing: {final_count}")

        if dedup_exists:
            try:
                with open(dedup_file, "r", encoding="utf-8") as f:
                    dedup_count = sum(1 for line in f) - 1  # Subtract header
                keys_info = (
                    f"using keys: {', '.join(self.config.primary_keys)}"
                    if self.config.primary_keys
                    else "no keys specified"
                )
                print(f"   - After deduplication: {dedup_count} ({keys_info})")
                print(f"   💾 Deduplicated version: {dedup_file}")
            except Exception:
                pass
        elif not self.config.no_dedup and self.config.primary_keys:
            print(
                f"   ✅ No duplicates found using primary keys: {', '.join(self.config.primary_keys)}"
            )
        elif not self.config.no_dedup and not self.config.primary_keys:
            print(f"   ℹ️  Deduplication skipped: no primary keys specified")


# ================================
# COMMAND LINE INTERFACE
# ================================
class CLIHandler:
    """Handle command line interface"""

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Advanced web content extraction with two-pass strategy",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Web URL extraction
  python crawl.py "https://example.com/papers" --schema title author doi affiliation
  python crawl.py "https://example.com" --schema title description --output results.csv --strategy two-pass
  
  # PDF file extraction
  python crawl.py "document.pdf" --schema title author abstract --strategy two-pass
  python crawl.py "research_paper.pdf" --schema title author doi --pdf-use-llm --pdf-force-ocr
  
  # Advanced options
  python crawl.py "https://example.com" --schema name email --strategy single-pass --chunk-size 2000
  python crawl.py "https://example.com" --schema title author --primary-keys title --verbose
  python crawl.py "document.pdf" --schema id name email --primary-keys id email --no-dedup --verbose
            """,
        )

        # Required arguments
        parser.add_argument(
            "url", nargs="?", help="URL to crawl or PDF file path to extract data from"
        )
        parser.add_argument(
            "--schema",
            nargs="+",
            required=False,
            help="Schema fields to extract (e.g., --schema title doi author affiliation)",
        )

        # Output options
        parser.add_argument(
            "--output",
            default=None,
            help="Output CSV filename (default: auto-generated based on timestamp)",
        )

        # Strategy options
        parser.add_argument(
            "--strategy",
            choices=["two-pass", "single-pass"],
            default="two-pass",
            help="Extraction strategy: two-pass (default) for better accuracy, single-pass for speed",
        )

        # Processing options
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=Constants.DEFAULT_CHUNK_SIZE,
            help=f"Maximum chunk size for processing (default: {Constants.DEFAULT_CHUNK_SIZE})",
        )
        parser.add_argument(
            "--overlap-size",
            type=int,
            default=Constants.DEFAULT_OVERLAP_SIZE,
            help=f"Overlap size between chunks (default: {Constants.DEFAULT_OVERLAP_SIZE})",
        )
        parser.add_argument(
            "--similarity-threshold",
            type=float,
            default=Constants.DEFAULT_SIMILARITY_THRESHOLD,
            help=f"Similarity threshold for duplicate detection (default: {Constants.DEFAULT_SIMILARITY_THRESHOLD})",
        )
        parser.add_argument(
            "--primary-keys",
            nargs="+",
            default=None,
            help="Primary key fields for deduplication (default: auto-detect from first field + common identifiers like title, doi)",
        )
        parser.add_argument(
            "--no-dedup",
            action="store_true",
            help="Skip automatic deduplication and only keep original file",
        )

        # Model configuration
        parser.add_argument(
            "--structure-model",
            default=None,
            choices=Constants.AVAILABLE_MODELS,
            help=f"Model for structure analysis (default: {ModelConfig.structure_analysis_model})",
        )
        parser.add_argument(
            "--instruction-model",
            default=None,
            choices=Constants.AVAILABLE_MODELS,
            help=f"Model for instruction generation (default: {ModelConfig.instruction_generation_model})",
        )
        parser.add_argument(
            "--extraction-model",
            default=None,
            choices=Constants.AVAILABLE_MODELS,
            help=f"Model for data extraction (default: {ModelConfig.data_extraction_model})",
        )

        # PDF processing options
        parser.add_argument(
            "--pdf-use-llm",
            action="store_true",
            help="Use LLM for improved PDF conversion quality (requires API key)",
        )
        parser.add_argument(
            "--pdf-force-ocr",
            action="store_true",
            help="Force OCR on PDF documents for better text extraction",
        )
        parser.add_argument(
            "--pdf-extract-images",
            action="store_true",
            default=False,
            help="Extract images from PDF files (default: False - text only)",
        )

        # Information and debugging
        parser.add_argument(
            "--show-models",
            action="store_true",
            help="Show available models and current configuration",
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output"
        )

        return parser

    @staticmethod
    def show_models() -> None:
        """Show available models and current configuration"""
        print("🤖 **Available Models:**")
        for i, model in enumerate(Constants.AVAILABLE_MODELS, 1):
            print(f"   {i}. {model}")
        print(f"\n📊 **Current Model Configuration:**")
        model_info = ModelConfig.get_model_info()
        print(f"   🔍 Structure Analysis: {model_info['structure_analysis']}")
        print(f"   📝 Instruction Generation: {model_info['instruction_generation']}")
        print(f"   📊 Data Extraction: {model_info['data_extraction']}")

        print(f"\n📄 **PDF Support:**")
        if PDF_SUPPORT_AVAILABLE:
            print(f"   ✅ PDF conversion available (marker library)")
        else:
            print(
                f"   ❌ PDF conversion not available (install: pip install marker-pdf)"
            )

    @staticmethod
    def generate_output_filename(url: str) -> str:
        """Generate output filename based on URL and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            domain = url.split("/")[2].replace(".", "_")
        except IndexError:
            domain = "unknown_domain"
        return f"output/{domain}_extraction_{timestamp}.csv"

    @staticmethod
    def print_configuration(config: ExtractionConfig) -> None:
        """Print extraction configuration"""
        print("🚀 Starting extraction with configuration:")

        # Show input source
        if config.is_pdf_input:
            print(f"   📄 PDF Input: {config.original_input}")
            print(f"   🌐 Converted to: {config.url}")
        else:
            print(f"   📍 URL: {config.url}")

        print(f"   📋 Schema fields: {', '.join(config.schema_fields)}")
        print(f"   💾 Output file: {config.output_file}")
        print(f"   🔄 Strategy: {config.strategy}")
        print(f"   📊 Chunk size: {config.chunk_size}")
        print(f"   🔗 Overlap size: {config.overlap_size}")
        print(f"   🎯 Similarity threshold: {config.similarity_threshold}")
        if config.primary_keys:
            print(
                f"   🗝️  Primary keys for deduplication: {', '.join(config.primary_keys)}"
            )
        else:
            print(f"   🗝️  Primary keys for deduplication: None (no deduplication)")
        if config.no_dedup:
            print(f"   🚫 Deduplication: Disabled")

        # Show PDF configuration if applicable
        if config.is_pdf_input and config.pdf_config:
            print(f"   📄 PDF Configuration:")
            if config.pdf_config.get("use_llm"):
                print(f"      🧠 LLM Enhancement: Enabled")
            if config.pdf_config.get("force_ocr"):
                print(f"      👁️  Force OCR: Enabled")
            if config.pdf_config.get("extract_images"):
                print(f"      🖼️  Extract Images: Enabled")

        # Show model configuration
        model_info = ModelConfig.get_model_info()
        print(f"   🤖 Model Configuration:")
        print(f"      🔍 Structure Analysis: {model_info['structure_analysis']}")
        print(
            f"      📝 Instruction Generation: {model_info['instruction_generation']}"
        )
        print(f"      📊 Data Extraction: {model_info['data_extraction']}")
        print()


# ================================
# MAIN APPLICATION
# ================================
async def main() -> int:
    """Main application entry point"""
    try:
        parser = CLIHandler.create_parser()
        args = parser.parse_args()

        # Handle special cases
        if args.show_models:
            CLIHandler.show_models()
            return 0

        # Debug PDF detection if verbose
        if args.verbose and args.url:
            print(f"🔍 Input analysis:")
            print(f"   Input: {args.url}")
            print(f"   PDF support available: {PDF_SUPPORT_AVAILABLE}")
            if PDF_SUPPORT_AVAILABLE:
                print(f"   Is PDF file: {is_pdf_file(args.url)}")
            print(f"   File exists: {os.path.exists(args.url)}")
            print(f"   Is absolute path: {os.path.isabs(args.url)}")
            print()

        # Validate required arguments
        if not args.url or not args.schema:
            parser.print_help()
            return 1

        # Set custom models if provided
        ModelConfig.set_models(
            structure_model=args.structure_model,
            instruction_model=args.instruction_model,
            extraction_model=args.extraction_model,
        )

        # Generate output filename if not provided
        output_file = args.output or CLIHandler.generate_output_filename(args.url)

        # Ensure output directory exists
        output_dir = (
            os.path.dirname(output_file) if os.path.dirname(output_file) else "output"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Create PDF configuration if needed
        pdf_config = {}
        if hasattr(args, "pdf_use_llm") and args.pdf_use_llm:
            pdf_config["use_llm"] = True
        if hasattr(args, "pdf_force_ocr") and args.pdf_force_ocr:
            pdf_config["force_ocr"] = True
        if hasattr(args, "pdf_extract_images"):
            pdf_config["extract_images"] = args.pdf_extract_images

        # Create extraction configuration
        config = ExtractionConfig(
            url=args.url,
            schema_fields=args.schema,
            output_file=output_file,
            strategy=args.strategy,
            chunk_size=args.chunk_size,
            overlap_size=args.overlap_size,
            similarity_threshold=args.similarity_threshold,
            primary_keys=args.primary_keys,
            no_dedup=args.no_dedup,
            verbose=args.verbose,
            pdf_config=pdf_config if pdf_config else None,
        )

        # Print configuration
        CLIHandler.print_configuration(config)

        # Execute extraction
        service = ExtractionService(config)
        return await service.extract()

    except (ValueError, ModelConfigError, ExtractionError) as e:
        print(f"❌ Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  Extraction interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
