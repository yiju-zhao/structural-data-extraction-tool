import os
import argparse
import logging
from pathlib import Path
from marker_parser import PDFProcessor, MetadataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MetadataExtractionPipeline:
    """Main pipeline for PDF processing and metadata extraction"""

    def __init__(self, args):
        """
        Initialize the pipeline with command line arguments

        Args:
            args: Command line arguments
        """
        # Set paths
        self.input_dir = Path(args.input)
        self.json_dir = Path(args.json_dir) if args.json_dir else self.input_dir
        self.output_dir = Path(args.output) if args.output else self.input_dir

        # Set processing options
        self.skip_pdf_parsing = args.skip_pdf_parsing
        self.batch_size = args.batch_size
        self.max_pages = args.max_pages
        self.use_gpu = args.use_gpu
        self.num_gpus = args.num_gpus if args.use_gpu else 0

        # Create directories if they don't exist
        self.json_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create processors
        # Only initialize PDFProcessor if we're not skipping PDF parsing
        self.pdf_processor = None
        if not self.skip_pdf_parsing:
            self.pdf_processor = PDFProcessor(
                use_gpu=self.use_gpu, max_pages=self.max_pages
            )
        self.metadata_processor = MetadataProcessor()

    def display_configuration(self):
        """Display pipeline configuration"""
        logger.info("=== PDF Metadata Extraction Pipeline ===")
        logger.info(f"Input directory:  {self.input_dir}")
        logger.info(f"JSON directory:   {self.json_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Skip PDF parsing: {self.skip_pdf_parsing}")
        logger.info(f"Batch size:       {self.batch_size}")
        logger.info(f"Max pages:        {self.max_pages}")
        logger.info(f"Use GPU:          {self.use_gpu}")
        if self.use_gpu and self.num_gpus > 1:
            logger.info(f"Number of GPUs:   {self.num_gpus}")
        logger.info("=====================================")

    def run(self):
        """Run the complete pipeline"""
        # Display configuration info
        self.display_configuration()

        # Process the PDFs to JSON if not skipped
        json_files = []
        if not self.skip_pdf_parsing:
            json_files = self.pdf_processor.process_pdfs(
                self.input_dir, self.json_dir, batch_size=self.batch_size, num_gpus=self.num_gpus
            )
        else:
            # Find all PDF files in the input directory
            pdf_files = list(self.input_dir.glob("*.pdf"))
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.input_dir}")
            
            # Check for corresponding JSON files
            json_files = []
            for pdf_file in pdf_files:
                pdf_name = pdf_file.stem
                json_file = self.json_dir / f"{pdf_name}.json"
                
                if json_file.exists():
                    json_files.append(json_file)
                else:
                    logger.warning(f"JSON file not found for {pdf_file.name}: {json_file}")
            
            logger.info(
                f"Found {len(json_files)} existing JSON files out of {len(pdf_files)} PDFs in {self.json_dir}"
            )

        # Process the JSON files to extract metadata
        if json_files:
            metadata_files = self.metadata_processor.process_json_files(
                json_files, self.output_dir
            )
            logger.info(f"\n=== Processing Complete ===")
            logger.info(f"Processed {len(metadata_files)} metadata files")
            logger.info(f"Metadata files saved to: {self.output_dir}")
        else:
            logger.warning("\n=== No files to process ===")


def main():
    """
    Main entry point for the PDF processing and metadata extraction pipeline.
    Handles command line arguments and orchestrates the workflow.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(
        description="Process PDF files to extract metadata"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to folder containing PDF files",
    )
    parser.add_argument(
        "--json-dir",
        "-j",
        type=str,
        help="Path to folder where intermediate JSON files will be saved (defaults to input directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to folder where extracted metadata will be saved (defaults to input directory)",
    )
    parser.add_argument(
        "--skip-pdf-parsing",
        action="store_true",
        help="Skip PDF parsing and use existing JSON files in json-dir",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of PDFs to process in each batch (default: 10)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Maximum number of pages to process per PDF (default: 0 for all pages)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU for processing if available (default: False)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for parallel processing (default: 1)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create and run the pipeline
    pipeline = MetadataExtractionPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
