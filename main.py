#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from processor import process_pdfs, process_json_files

def main():
    """
    Main entry point for the PDF processing and metadata extraction pipeline.
    Handles command line arguments and orchestrates the workflow.
    """
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Process PDF files to extract metadata')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to folder containing PDF files')
    parser.add_argument('--json-dir', '-j', type=str,
                        help='Path to folder where intermediate JSON files will be saved (defaults to input directory)')
    parser.add_argument('--output', '-o', type=str,
                        help='Path to folder where extracted metadata will be saved (defaults to input directory)')
    parser.add_argument('--skip-pdf-parsing', action='store_true',
                        help='Skip PDF parsing and use existing JSON files in json-dir')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of PDFs to process in each batch (default: 10)')
    parser.add_argument('--max-pages', type=int, default=1,
                        help='Maximum number of pages to process per PDF (default: 1)')
    parser.add_argument('--use-gpu', action='store_true', default=False,
                        help='Use GPU for processing if available (default: False)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default directories if not provided
    input_dir = args.input
    json_dir = args.json_dir if args.json_dir else input_dir
    output_dir = args.output if args.output else input_dir
    
    # Display configuration
    print("=== PDF Metadata Extraction Pipeline ===")
    print(f"Input directory:  {input_dir}")
    print(f"JSON directory:   {json_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Skip PDF parsing: {args.skip_pdf_parsing}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Max pages:        {args.max_pages}")
    print(f"Use GPU:          {args.use_gpu}")
    print("=====================================")
    
    # Process the PDFs to JSON if not skipped
    json_files = []
    if not args.skip_pdf_parsing:
        json_files = process_pdfs(
            input_dir, 
            json_dir, 
            max_pages=args.max_pages,
            use_gpu=args.use_gpu,
            batch_size=args.batch_size
        )
    else:
        # Use existing JSON files
        json_files = list(Path(json_dir).glob('*.json'))
        print(f"Using {len(json_files)} existing JSON files from {json_dir}")
    
    # Process the JSON files to extract metadata
    process_json_files(json_files, output_dir)
    
    print("\n=== Processing Complete ===")
    print(f"Metadata files saved to: {output_dir}")

if __name__ == '__main__':
    main() 