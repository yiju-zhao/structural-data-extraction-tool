import os
import argparse
import sys
from pathlib import Path
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod


def parse_page_ranges(page_filter_str):
    """
    Parse page filter string into list of (start, end) tuples for continuous ranges.
    
    Args:
        page_filter_str (str): Comma-separated page numbers and ranges like "1,5-10,20" (1-indexed)
    
    Returns:
        list: List of (start_page, end_page) tuples representing continuous ranges (0-indexed for internal use)
    """
    if not page_filter_str:
        return None
    
    pages = set()
    
    # Split by comma and process each part
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
            
            # Convert to 0-indexed and add to set
            pages.update(range(start_page - 1, end_page))
        else:
            # Handle single page like "1" or "20"
            page_num = int(part.strip())
            
            # Validate page number (must be >= 1)
            if page_num < 1:
                raise ValueError(f"Page numbers must be >= 1, got page {page_num}")
            
            # Convert to 0-indexed
            pages.add(page_num - 1)
    
    # Convert to sorted list
    sorted_pages = sorted(pages)
    
    # Group consecutive pages into ranges
    ranges = []
    if sorted_pages:
        start = sorted_pages[0]
        end = sorted_pages[0]
        
        for i in range(1, len(sorted_pages)):
            if sorted_pages[i] == end + 1:
                # Consecutive page, extend current range
                end = sorted_pages[i]
            else:
                # Gap found, close current range and start new one
                ranges.append((start, end))
                start = sorted_pages[i]
                end = sorted_pages[i]
        
        # Add the last range
        ranges.append((start, end))
    
    return ranges


def parse_pdf(pdf_path, output_dir="output", page_filter=None):
    """
    Parse a PDF file using MinerU and extract content.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_dir (str): Directory to save parsed output files
        page_filter (str): Optional page filter string like "1,5-10,20" (1-indexed)
    
    Returns:
        dict: Dictionary containing paths to generated files and extracted content
    """
    
    # Validate input file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("Input file must be a PDF file")
    
    # Parse page ranges if provided
    page_ranges = parse_page_ranges(page_filter) if page_filter else None
    
    # Setup paths and directories
    pdf_file_name = os.path.basename(pdf_path)
    name_without_suff = os.path.splitext(pdf_file_name)[0]
    
    local_image_dir = os.path.join(output_dir, "images")
    local_md_dir = output_dir
    image_dir = os.path.basename(local_image_dir)
    
    # Create output directories
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    
    # Initialize writers and readers
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)
    reader = FileBasedDataReader("")
    
    try:
        # Read PDF bytes
        pdf_bytes = reader.read(pdf_path)
        
        # Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)
        
        # Classify and process based on PDF type
        if ds.classify() == SupportedPdfParseMethod.OCR:
            if page_ranges:
                # Process specific page ranges
                all_pipe_results = []
                all_infer_results = []
                for start_page, end_page in page_ranges:
                    print(f"Processing pages {start_page}-{end_page}...")
                    infer_result = ds.apply(doc_analyze, ocr=True, start_page_id=start_page, end_page_id=end_page)
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                    all_infer_results.append(infer_result)
                    all_pipe_results.append(pipe_result)
                # Use the first result for main processing (we might need to merge results later)
                infer_result = all_infer_results[0]
                pipe_result = all_pipe_results[0]
            else:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            if page_ranges:
                # Process specific page ranges
                all_pipe_results = []
                all_infer_results = []
                for start_page, end_page in page_ranges:
                    print(f"Processing pages {start_page}-{end_page}...")
                    infer_result = ds.apply(doc_analyze, ocr=False, start_page_id=start_page, end_page_id=end_page)
                    pipe_result = infer_result.pipe_txt_mode(image_writer)
                    all_infer_results.append(infer_result)
                    all_pipe_results.append(pipe_result)
                # Use the first result for main processing (we might need to merge results later)
                infer_result = all_infer_results[0]
                pipe_result = all_pipe_results[0]
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)
        
        # Draw model result on each page
        model_pdf_path = os.path.join(local_md_dir, f"{name_without_suff}_model.pdf")
        infer_result.draw_model(model_pdf_path)
        
        # Draw layout result on each page
        layout_pdf_path = os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf")
        pipe_result.draw_layout(layout_pdf_path)
        
        # Draw spans result on each page
        spans_pdf_path = os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf")
        pipe_result.draw_span(spans_pdf_path)
        
        # Get and dump markdown content
        md_content = pipe_result.get_markdown(image_dir)
        md_file_path = os.path.join(local_md_dir, f"{name_without_suff}.md")
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)
        
        # Get and dump content list
        content_list_content = pipe_result.get_content_list(image_dir)
        content_list_path = os.path.join(local_md_dir, f"{name_without_suff}_content_list.json")
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)
        
        # Get and dump middle json
        middle_json_content = pipe_result.get_middle_json()
        middle_json_path = os.path.join(local_md_dir, f"{name_without_suff}_middle.json")
        pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')
        
        # Get model inference result
        model_inference_result = infer_result.get_infer_res()
        
        # Add page filter info to results
        page_info = {}
        if page_filter:
            page_info = {
                'page_filter': page_filter,
                'processed_ranges': page_ranges
            }
        
        # Return results
        results = {
            'status': 'success',
            'input_file': pdf_path,
            'output_directory': output_dir,
            'page_info': page_info,
            'generated_files': {
                'markdown': md_file_path,
                'content_list_json': content_list_path,
                'middle_json': middle_json_path,
                'model_pdf': model_pdf_path,
                'layout_pdf': layout_pdf_path,
                'spans_pdf': spans_pdf_path,
                'images_directory': local_image_dir
            },
            'extracted_content': {
                'markdown': md_content,
                'content_list': content_list_content,
                'middle_json': middle_json_content,
                'model_inference': model_inference_result
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
        print(f"Processed ranges: {results['page_info']['processed_ranges']}")

def main():
    """Main function to handle command line arguments and run PDF parsing."""
    
    parser = argparse.ArgumentParser(
        description="Parse PDF files using MinerU (magic_pdf) package"
    )
    
    parser.add_argument(
        'pdf_path',
        help='Path to the input PDF file'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output',
        help='Output directory for parsed files (default: output)'
    )
    
    parser.add_argument(
        '--filter-page',
        help='Specify which pages to process. Accepts comma-separated page numbers and ranges (1-indexed). Example: "1,5-10,20" will process pages 1, 5 through 10, and page 20.'
    )

    args = parser.parse_args()    
    # Convert to absolute paths
    pdf_path = os.path.abspath(args.pdf_path)
    output_dir = os.path.abspath(args.output)
    
    try:
        # Parse the PDF
        results = parse_pdf(pdf_path, output_dir, args.filter_page)
        
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