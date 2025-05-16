#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Union, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import marker modules
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import save_output

# Import our metadata extractor
from metadata_extraction import MetadataExtractor


class PDFProcessor:
    """Class for processing PDFs and extracting metadata"""

    def __init__(self, use_gpu: bool = False, max_pages: int = 0):
        """
        Initialize the PDF processor

        Args:
            use_gpu: Whether to use GPU for processing
            max_pages: Maximum number of pages to process
        """
        self.use_gpu = use_gpu
        self.max_pages = max_pages

        # Initialize the converter once and reuse it
        self._init_converter()

    def _init_converter(self):
        """Initialize the PDF converter once for reuse"""
        # Configure marker with JSON output format
        config = {
            "output_format": "json",
            "use_gpu": self.use_gpu,
            "page_range": f"0,{self.max_pages}" if self.max_pages > 0 else None,
        }

        # Filter out None values
        config = {k: v for k, v in config.items() if v is not None}

        self.config_parser = ConfigParser(config)

        # Use marker Python API with configuration
        try:
            self.converter = PdfConverter(
                config=self.config_parser.generate_config_dict(),
                processor_list=self.config_parser.get_processors(),
                renderer=self.config_parser.get_renderer(),
                artifact_dict=create_model_dict(),
            )
            logger.info(f"Initialized PDF converter with GPU={self.use_gpu}")
        except Exception as e:
            logger.warning(
                f"Error initializing converter with GPU={self.use_gpu}, trying with CPU: {str(e)}"
            )
            # Try with CPU if GPU fails
            config["use_gpu"] = False
            self.use_gpu = False
            self.config_parser = ConfigParser(config)
            self.converter = PdfConverter(
                config=self.config_parser.generate_config_dict(),
                processor_list=self.config_parser.get_processors(),
                renderer=self.config_parser.get_renderer(),
                artifact_dict=create_model_dict(),
            )
            logger.info("Initialized PDF converter with CPU")

    def convert_pdf_to_json(
        self, pdf_file: Union[str, Path], output_dir: Union[str, Path]
    ) -> Optional[Path]:
        """
        Convert PDF to JSON format using marker Python API.

        Args:
            pdf_file: Path to PDF file
            output_dir: Directory to save resulting JSON

        Returns:
            Path to the generated JSON file, or None if conversion failed
        """
        try:
            # Create output filename based on the PDF name
            pdf_path = Path(pdf_file)
            pdf_name = pdf_path.stem
            output_dir_path = Path(output_dir)
            json_output = output_dir_path / f"{pdf_name}.json"

            # Skip if JSON already exists
            if json_output.exists():
                logger.info(f"Skipping {pdf_file}: JSON already exists")
                return json_output

            # Convert the PDF to JSON using the pre-initialized converter
            rendered = self.converter(str(pdf_file))

            # Save the rendered output to JSON file
            save_output(
                rendered,
                str(output_dir_path),
                self.config_parser.get_base_filename(str(json_output)),
            )

            logger.info(f"Successfully converted {pdf_file} to JSON")
            return json_output
        except KeyboardInterrupt:
            logger.warning(f"Process interrupted by user while processing {pdf_file}")
            return None
        except Exception as e:
            logger.error(f"Error converting {pdf_file} to JSON: {str(e)}")
            return None

    def process_pdfs(
        self,
        input_folder: Union[str, Path],
        json_folder: Union[str, Path],
        batch_size: int = 10,
    ) -> List[Path]:
        """
        Process all PDF files in the input folder and convert to JSON using marker.

        Args:
            input_folder: Path to folder containing PDF files
            json_folder: Path to folder where JSON files will be saved
            batch_size: Number of PDFs to process in each batch

        Returns:
            List of paths to generated JSON files
        """
        # Create json folder if it doesn't exist
        input_path = Path(input_folder)
        json_path = Path(json_folder)
        json_path.mkdir(exist_ok=True, parents=True)

        # Get all PDF files in the input folder
        pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_folder}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Process each PDF file in batches
        json_files = []
        try:
            total_batches = (len(pdf_files) + batch_size - 1) // batch_size
            for i in range(0, len(pdf_files), batch_size):
                batch = pdf_files[i : i + batch_size]
                logger.info(
                    f"Processing batch {i // batch_size + 1}/{total_batches} ({len(batch)} files)"
                )

                for pdf_file in tqdm(batch, desc="Converting PDFs to JSON"):
                    json_file = self.convert_pdf_to_json(pdf_file, json_path)
                    if json_file:
                        json_files.append(json_file)
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")

        logger.info(
            f"Converted {len(json_files)} PDFs to JSON format. Results saved to {json_folder}"
        )
        return json_files


class MetadataProcessor:
    """Class for processing JSON files and extracting structured metadata"""

    def __init__(self):
        """Initialize the metadata processor"""
        self.extractor = MetadataExtractor()

    def process_json_files(
        self, json_files: List[Path], output_folder: Union[str, Path]
    ) -> List[Path]:
        """
        Process JSON files to extract metadata.

        Args:
            json_files: List of paths to JSON files
            output_folder: Path to folder where extracted metadata will be saved

        Returns:
            List of metadata file paths
        """
        # Create output folder if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)

        if not json_files:
            logger.warning("No JSON files to process")
            return []

        logger.info(f"Processing {len(json_files)} JSON files for metadata extraction")

        # Process each JSON file
        metadata_files = []
        try:
            for json_file in tqdm(json_files, desc="Extracting metadata"):
                try:
                    # Extract file name without extension for output file naming
                    file_name = Path(json_file).stem
                    output_file = output_path / f"{file_name}_metadata.json"

                    # Skip if metadata file already exists
                    if output_file.exists():
                        logger.info(f"Skipping {json_file}: metadata already exists")
                        metadata_files.append(output_file)
                        continue

                    # Extract metadata from JSON
                    extracted_data = self.extractor.extract_from_json(str(json_file))

                    # Get structured metadata using OpenAI
                    structured_metadata = self.extractor.get_structured_metadata(
                        extracted_data
                    )

                    # Write extracted metadata to output folder
                    with open(output_file, "w") as f:
                        json.dump(structured_metadata, f, indent=2)

                    metadata_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error processing {json_file}: {str(e)}")
        except KeyboardInterrupt:
            logger.warning("Process interrupted by user")

        logger.info(
            f"Processed {len(metadata_files)} files. Metadata saved to {output_folder}"
        )
        return metadata_files


# Legacy functions for backward compatibility
def convert_pdf_to_json(pdf_file, output_dir, max_pages, use_gpu):
    """Legacy function for backward compatibility"""
    processor = PDFProcessor(use_gpu=use_gpu, max_pages=max_pages)
    return processor.convert_pdf_to_json(pdf_file, output_dir)


def process_pdfs(input_folder, json_folder, max_pages, use_gpu, batch_size):
    """Legacy function for backward compatibility"""
    processor = PDFProcessor(use_gpu=use_gpu, max_pages=max_pages)
    return processor.process_pdfs(input_folder, json_folder, batch_size)


def process_json_files(json_files, output_folder):
    """Legacy function for backward compatibility"""
    processor = MetadataProcessor()
    return processor.process_json_files(json_files, output_folder)
