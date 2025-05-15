#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from metadata_extraction import extract_metadata_from_json, get_structured_metadata

# Import marker modules
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

def convert_pdf_to_json(pdf_file, output_dir, max_pages=50, use_gpu=True):
    """
    Convert PDF to AdaParse JSON format using marker Python API with ConfigParser.
    
    Args:
        pdf_file: Path to PDF file
        output_dir: Directory to save resulting JSON
        max_pages: Maximum number of pages to process
        use_gpu: Whether to use GPU for processing
        
    Returns:
        Path to the generated JSON file, or None if conversion failed
    """
    try:
        # Create output filename based on the PDF name
        pdf_name = Path(pdf_file).stem
        json_output = Path(output_dir) / f"{pdf_name}.json"
        
        # Configure marker with JSON output format
        config = {
            "output_format": "json",
            "use_gpu": use_gpu,
            "max_pages": max_pages,
        }
        config_parser = ConfigParser(config)
        
        # Use marker Python API with configuration
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        
        # Convert the PDF to JSON
        rendered = converter(str(pdf_file))
        
        # Save the rendered output to JSON file
        with open(json_output, 'w') as f:
            json.dump(rendered, f, indent=2)
        
        # Return the path to the generated JSON file
        return json_output
    except Exception as e:
        print(f"Error converting {pdf_file} to JSON: {str(e)}")
        return None

def process_pdfs(input_folder, json_folder, max_pages=1, use_gpu=False, batch_size=2):
    """
    Process all PDF files in the input folder and convert to JSON using marker.
    
    Args:
        input_folder: Path to folder containing PDF files
        json_folder: Path to folder where JSON files will be saved
        max_pages: Maximum number of pages to process per PDF
        use_gpu: Whether to use GPU for processing
        batch_size: Number of PDFs to process in each batch
        
    Returns:
        List of paths to generated JSON files
    """
    # Create json folder if it doesn't exist
    os.makedirs(json_folder, exist_ok=True)
    
    # Get all PDF files in the input folder
    pdf_files = list(Path(input_folder).glob('*.pdf'))
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}")
        return []
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    json_files = []
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(pdf_files) + batch_size - 1)//batch_size} ({len(batch)} files)")
        
        for pdf_file in tqdm(batch, desc="Converting PDFs to JSON"):
            json_file = convert_pdf_to_json(
                pdf_file, 
                json_folder, 
                max_pages=max_pages, 
                use_gpu=use_gpu
            )
            if json_file:
                json_files.append(json_file)
    
    print(f"Converted {len(json_files)} PDFs to JSON format. Results saved to {json_folder}")
    return json_files

def process_json_files(json_files, output_folder):
    """
    Process AdaParse JSON files to extract metadata.
    
    Args:
        json_files: List of paths to AdaParse JSON files
        output_folder: Path to folder where extracted metadata will be saved
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    if not json_files:
        print("No JSON files to process")
        return
    
    print(f"Processing {len(json_files)} JSON files for metadata extraction")
    
    # Process each JSON file
    for json_file in tqdm(json_files, desc="Extracting metadata"):
        try:
            # Extract file name without extension for output file naming
            file_name = Path(json_file).stem
            
            # Extract metadata from JSON
            extracted_data = extract_metadata_from_json(str(json_file))
            
            # Get structured metadata using OpenAI
            structured_metadata = get_structured_metadata(extracted_data)
            
            # Write extracted metadata to output folder
            output_file = Path(output_folder) / f"{file_name}_metadata.json"
            with open(output_file, 'w') as f:
                json.dump(structured_metadata, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    print(f"Processed {len(json_files)} files. Metadata saved to {output_folder}")

# This standalone main function will only run if process.py is called directly
def main():
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
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default directories if not provided
    input_dir = args.input
    json_dir = args.json_dir if args.json_dir else input_dir
    output_dir = args.output if args.output else input_dir
    
    # Process the PDFs to JSON if not skipped
    json_files = []
    if not args.skip_pdf_parsing:
        json_files = process_pdfs(input_dir, json_dir)
    else:
        # Use existing JSON files
        json_files = list(Path(json_dir).glob('*.json'))
        print(f"Using {len(json_files)} existing JSON files from {json_dir}")
    
    # Process the JSON files to extract metadata
    process_json_files(json_files, output_dir)

# Only run the main function if this script is called directly
if __name__ == '__main__':
    main()
