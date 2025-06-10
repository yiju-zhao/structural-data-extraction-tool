import os
import json
import logging
import multiprocessing
import argparse
import sys
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Union, Optional, Dict

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


def clean_title(title: str) -> str:
    """Clean the title by replacing non-alphanumeric characters with underscores."""
    # Replace all non-alphanumeric characters (except for underscores) with underscores
    cleaned = re.sub(r'[^\w\d]', '_', title)
    # Replace consecutive underscores with a single underscore
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned


def parse_page_ranges(page_filter_str):
    """
    Parse page filter string into marker-compatible format.
    
    Args:
        page_filter_str (str): Comma-separated page numbers and ranges like "1,5-10,20" (1-indexed)
    
    Returns:
        str: Marker-compatible page range string (0-indexed) like "0,4-9,19"
    """
    if not page_filter_str:
        return None
    
    # Parse individual pages and ranges
    marker_parts = []
    
    for part in page_filter_str.split(','):
        part = part.strip()
        if '-' in part:
            # Handle range like "5-10"
            start, end = part.split('-', 1)
            start_page = int(start.strip())
            end_page = int(end.strip())
            
            # Validate page numbers (must be >= 1)
            if start_page < 1 or end_page < 1:
                raise ValueError(f"Page numbers must be >= 1, got range {start_page}-{end_page}")
            if start_page > end_page:
                raise ValueError(f"Invalid range {start_page}-{end_page}: start page must be <= end page")
            
            # Convert to 0-indexed for marker
            marker_parts.append(f"{start_page - 1}-{end_page - 1}")
        else:
            # Handle single page like "1" or "20"
            page_num = int(part.strip())
            
            # Validate page number (must be >= 1)
            if page_num < 1:
                raise ValueError(f"Page numbers must be >= 1, got page {page_num}")
            
            # Convert to 0-indexed for marker
            marker_parts.append(str(page_num - 1))
    
    return ",".join(marker_parts)


class PDFProcessor:
    """Class for processing PDFs and extracting content"""

    def __init__(self, use_gpu: bool = False, max_pages: int = 0, gpu_id: int = None, output_format: str = "markdown", page_range: str = None):
        """
        Initialize the PDF processor

        Args:
            use_gpu: Whether to use GPU for processing
            max_pages: Maximum number of pages to process
            gpu_id: Specific GPU ID to use (for multi-GPU systems)
            output_format: Output format ("markdown" or "json")
            page_range: Page range string in marker format (0-indexed)
        """
        self.use_gpu = use_gpu
        self.max_pages = max_pages
        self.gpu_id = gpu_id
        self.output_format = output_format
        self.page_range = page_range
        
        # If using a specific GPU, set the environment variable
        if self.use_gpu and self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            logger.info(f"Using GPU ID: {self.gpu_id}")

        # Initialize the converter once and reuse it
        self._init_converter()

    def _init_converter(self):
        """Initialize the PDF converter once for reuse"""
        # Configure marker with specified output format
        config = {
            "output_format": self.output_format,
            "use_gpu": self.use_gpu,
        }
        
        # Add page range if specified
        if self.page_range:
            config["page_range"] = self.page_range
        elif self.max_pages > 0:
            config["page_range"] = f"0-{self.max_pages-1}"

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
            logger.info(f"Initialized PDF converter with GPU={self.use_gpu}, format={self.output_format}")
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
            logger.info(f"Initialized PDF converter with CPU, format={self.output_format}")

    def convert_pdf_to_markdown(
        self, pdf_file: Union[str, Path], output_dir: Union[str, Path]
    ) -> Optional[Path]:
        """
        Convert PDF to markdown format using marker Python API.

        Args:
            pdf_file: Path to PDF file
            output_dir: Directory to save resulting markdown

        Returns:
            Path to the generated markdown file, or None if conversion failed
        """
        try:
            # Create output filename based on the PDF name
            pdf_path = Path(pdf_file)
            pdf_name = pdf_path.stem
            output_dir_path = Path(output_dir)
            markdown_output = output_dir_path / f"{pdf_name}.md"

            # Skip if markdown already exists
            if markdown_output.exists():
                logger.info(f"Skipping {pdf_file}: markdown already exists")
                return markdown_output

            # Convert the PDF to markdown using the pre-initialized converter
            rendered = self.converter(str(pdf_file))

            # Save the rendered output to markdown file
            save_output(
                rendered,
                str(output_dir_path),
                self.config_parser.get_base_filename(str(markdown_output)),
            )

            logger.info(f"Successfully converted {pdf_file} to markdown")
            return markdown_output
        except KeyboardInterrupt:
            logger.warning(f"Process interrupted by user while processing {pdf_file}")
            return None
        except Exception as e:
            logger.error(f"Error converting {pdf_file} to markdown: {str(e)}")
            return None

    def convert_pdf_to_json(
        self, pdf_file: Union[str, Path], output_dir: Union[str, Path]
    ) -> Optional[Path]:
        """
        Convert PDF to JSON format using marker Python API.
        Legacy method for backward compatibility.

        Args:
            pdf_file: Path to PDF file
            output_dir: Directory to save resulting JSON

        Returns:
            Path to the generated JSON file, or None if conversion failed
        """
        # Temporarily change output format to JSON for this call
        original_format = self.output_format
        self.output_format = "json"
        self._init_converter()
        
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
        finally:
            # Restore original format
            self.output_format = original_format
            self._init_converter()

    def process_pdf_batch(self, batch: List[Path], json_path: Path) -> List[Path]:
        """Process a batch of PDFs and return the resulting JSON files"""
        json_files = []
        for pdf_file in batch:
            json_file = self.convert_pdf_to_json(pdf_file, json_path)
            if json_file:
                json_files.append(json_file)
        return json_files

    def process_pdfs(
        self,
        input_folder: Union[str, Path],
        json_folder: Union[str, Path],
        batch_size: int = 10,
        num_gpus: int = 0
    ) -> List[Path]:
        """
        Process all PDF files in the input folder and convert to JSON using marker.
        
        Args:
            input_folder: Path to folder containing PDF files
            json_folder: Path to folder where JSON files will be saved
            batch_size: Number of PDFs to process in each batch
            num_gpus: Number of GPUs to use in parallel (0 for CPU-only)
            
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
        
        # If using multiple GPUs, use parallel processing
        json_files = []
        if self.use_gpu and num_gpus > 1:
            logger.info(f"Using {num_gpus} GPUs in parallel")
            return self._process_pdfs_parallel(pdf_files, json_path, batch_size, num_gpus)
        else:
            # Process each PDF file in batches (single GPU or CPU)
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
        
    def _process_pdfs_parallel(
        self, pdf_files: List[Path], json_path: Path, batch_size: int, num_gpus: int
    ) -> List[Path]:
        """Process PDFs in parallel using multiple GPUs"""
        # Distribute files across GPUs
        gpu_batches: Dict[int, List[Path]] = {i: [] for i in range(num_gpus)}
        for i, pdf_file in enumerate(pdf_files):
            gpu_id = i % num_gpus
            gpu_batches[gpu_id].append(pdf_file)
            
        logger.info(f"Distributed {len(pdf_files)} files across {num_gpus} GPUs")
        
        # Process each GPU's files in a separate process
        pool = multiprocessing.Pool(processes=num_gpus)
        results = []
        
        for gpu_id, gpu_files in gpu_batches.items():
            # Only create a process if there are files to process for this GPU
            if not gpu_files:
                continue
                
            logger.info(f"GPU {gpu_id}: Processing {len(gpu_files)} files")
            
            # Create sub-batches for this GPU
            for i in range(0, len(gpu_files), batch_size):
                sub_batch = gpu_files[i:i+batch_size]
                # Start a worker process for this sub-batch on this GPU
                result = pool.apply_async(process_gpu_batch, 
                                         args=(sub_batch, json_path, self.max_pages, gpu_id))
                results.append(result)
        
        # Close the pool and wait for all processes to finish
        pool.close()
        
        # Track progress with tqdm
        with tqdm(total=len(results), desc="Processing GPU batches") as pbar:
            finished_results = []
            for result in results:
                # This will block until the result is ready
                batch_results = result.get()
                finished_results.extend(batch_results)
                pbar.update(1)
        
        pool.join()
        
        return finished_results


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


# Worker function for parallel processing (must be at module level for multiprocessing)
def process_gpu_batch(batch_files, json_path, max_pages, gpu_id):
    """Process a batch of PDFs on a specific GPU"""
    # Set environment variable for this process to use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Create a processor for this batch with the specific GPU
    processor = PDFProcessor(use_gpu=True, max_pages=max_pages, gpu_id=gpu_id)
    
    # Process the batch
    results = []
    for pdf_file in batch_files:
        json_file = processor.convert_pdf_to_json(pdf_file, json_path)
        if json_file:
            results.append(json_file)
    
    return results


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


def parse_pdf(pdf_path, output_dir=None, page_filter=None, use_gpu=False):
    """
    Parse a PDF file using marker and extract content to markdown.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_dir (str): Base directory to save parsed output files (if None, uses current directory)
        page_filter (str): Optional page filter string like "1,5-10,20" (1-indexed)
        use_gpu (bool): Whether to use GPU for processing
    
    Returns:
        dict: Dictionary containing paths to generated files and extracted content
    """
    
    # Validate input file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("Input file must be a PDF file")
    
    # Parse page ranges if provided
    page_range = parse_page_ranges(page_filter) if page_filter else None
    
    # Setup paths and directories with cleaned PDF name
    pdf_file_name = os.path.basename(pdf_path)
    name_without_suff = os.path.splitext(pdf_file_name)[0]
    cleaned_pdf_name = clean_title(name_without_suff)
    
    # Use cleaned PDF name for output directory
    base_dir = output_dir if output_dir else "."
    output_dir = os.path.join(base_dir, f"{cleaned_pdf_name}_output")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize PDF processor with marker
        processor = PDFProcessor(
            use_gpu=use_gpu,
            output_format="markdown",
            page_range=page_range
        )
        
        # Convert PDF to markdown
        markdown_file = processor.convert_pdf_to_markdown(pdf_path, output_dir)
        
        if not markdown_file:
            raise Exception("Failed to convert PDF to markdown")
        
        # Read the generated markdown content
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Add page filter info to results
        page_info = {}
        if page_filter:
            page_info = {
                'page_filter': page_filter,
                'page_range_marker_format': page_range
            }
        
        # Return results
        results = {
            'status': 'success',
            'input_file': pdf_path,
            'output_directory': output_dir,
            'page_info': page_info,
            'generated_files': {
                'markdown': str(markdown_file),
            },
            'extracted_content': {
                'markdown': markdown_content,
            }
        }
        
        return results
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        print(f"ERROR: {error_msg}")
        return {
            'status': 'error',
            'error': error_msg,
            'input_file': pdf_path
        }


def print_results_summary(results):
    """Print a summary of the parsing results."""
    
    if results['status'] == 'error':
        print(f"\nFailed to process: {results['input_file']}")
        print(f"Error: {results['error']}")
        return
    
    print(f"Output directory: {results['output_directory']}")
    
    # Print page filter info if applicable
    if results.get('page_info') and results['page_info'].get('page_filter'):
        print(f"Page filter applied: {results['page_info']['page_filter']}")
        print(f"Marker page range: {results['page_info']['page_range_marker_format']}")
    
    # Print generated files
    print(f"Generated markdown file: {results['generated_files']['markdown']}")


def main():
    """Main function to handle command line arguments and run PDF parsing."""
    
    parser = argparse.ArgumentParser(
        description="Parse PDF files using marker package to generate markdown"
    )
    
    parser.add_argument(
        'pdf_path',
        help='Path to the input PDF file'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Base output directory for parsed files (default: current directory with PDF name)'
    )
    
    parser.add_argument(
        '--filter-page',
        help='Specify which pages to process. Accepts comma-separated page numbers and ranges (1-indexed). Example: "1,5-10,20" will process pages 1, 5 through 10, and page 20.'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU for processing (default: CPU)'
    )

    args = parser.parse_args()    
    # Convert to absolute paths
    pdf_path = os.path.abspath(args.pdf_path)
    output_dir = os.path.abspath(args.output) if args.output else None
    
    try:
        # Parse the PDF
        results = parse_pdf(pdf_path, output_dir, args.filter_page, args.use_gpu)
        
        # Print results summary
        print_results_summary(results)
        
        # Exit with appropriate code
        sys.exit(0 if results['status'] == 'success' else 1)
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
