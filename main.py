import argparse
import json
import os
import csv
import sys
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil

from data_extractor import (
    load_secrets, parse_schema_input, find_sections_with_fallback,
    Schema_Instruction, extract_with_openai, log_prompt_interaction, setup_logging,
    is_empty_row, clean_title
)
from openai import OpenAI


def parse_pdf_to_markdown_only(pdf_path: str, markdown_output_dir: str, page_filter: str = None, parser_type: str = "marker") -> str:
    """
    Parse a PDF file and save the markdown file while returning the content.
    
    Args:
        pdf_path (str): Path to the PDF file
        markdown_output_dir (str): Directory to save the markdown file
        page_filter (str): Optional page filter string like "1,5-10,20"
        parser_type (str): Parser to use - either "marker" or "mineru"
    
    Returns:
        str: Markdown content from the PDF
    """
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Choose parser based on parser_type
            if parser_type == "marker":
                from marker_parser import parse_pdf as marker_parse_pdf
                
                # Parse PDF using marker
                results = marker_parse_pdf(pdf_path, temp_dir, page_filter)
            else:
                from mineru_parser import parse_pdf as mineru_parse_pdf
                results = mineru_parse_pdf(pdf_path, temp_dir, page_filter)
            
            if results['status'] == 'error':
                print(f"Error parsing {pdf_path}: {results['error']}")
                return ""
            
            # Extract markdown content from results
            markdown_content = results['extracted_content']['markdown']
            
            # Save markdown file to output directory
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            cleaned_pdf_name = clean_title(pdf_name)
            md_filename = f"{cleaned_pdf_name}.md"
            md_output_path = os.path.join(markdown_output_dir, md_filename)
            
            # Ensure output directory exists
            os.makedirs(markdown_output_dir, exist_ok=True)
            
            # Write markdown content to file
            with open(md_output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"  Saved markdown: {md_output_path}")
            
            return markdown_content
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""


def process_markdown_with_schema(
    client: OpenAI,
    markdown_content: str,
    schema_fields: List[str],
    schema_definition: str,
    pdf_filename: str,
    log_filename: str
) -> List[Dict[str, Any]]:
    """
    Process markdown content and extract structured data according to schema.
    
    Args:
        client: OpenAI client
        markdown_content: The markdown content to process
        schema_fields: List of field names for the schema
        schema_definition: Detailed schema definition
        pdf_filename: Name of the source PDF file
        log_filename: Path to log file
    
    Returns:
        List of extracted records with source file information
    """
    if not markdown_content.strip():
        print(f"No markdown content found for {pdf_filename}")
        return []
    
    # Split markdown into sections with fallback (H1 -> H2 -> H3 -> etc.)
    sections, heading_level = find_sections_with_fallback(markdown_content)
    if not sections:
        print(f"No sections found in {pdf_filename} (tried H1-H6)")
        return []
    
    heading_name = f"H{heading_level}"
    print(f"Processing {len(sections)} {heading_name} sections from {pdf_filename}")
    
    extracted_list = []
    for idx, section in enumerate(sections, start=1):
        print(f"  Processing section {idx}/{len(sections)} from {pdf_filename}")
        
        try:
            parsed_list = extract_with_openai(
                client=client,
                schema_fields=schema_fields,
                section_text=section,
                schema_definition=schema_definition,
                log_filename=log_filename,
            )
            
            # Add source file information to each record
            for record in parsed_list:
                record['source_file'] = pdf_filename
                record['source_section'] = f"section_{idx}"
            
            extracted_list.extend(parsed_list)
            print(f"    Extracted {len(parsed_list)} records from section {idx}")
            
        except Exception as e:
            print(f"    Error extracting from section {idx} in {pdf_filename}: {str(e)}")
    
    return extracted_list


def process_pdf_folder(
    pdf_folder: str,
    schema_input: str,
    page_filter: str = None,
    output_csv: str = None,
    parser_type: str = "marker"
) -> None:
    """
    Process all PDF files in a folder and extract structured data to CSV.
    
    Args:
        pdf_folder: Path to folder containing PDF files
        schema_input: Schema definition as string
        page_filter: Optional page filter string
        output_csv: Output CSV file path
        parser_type: Parser to use - either "marker" or "mineru"
    """
    # Initialize logging
    setup_logging()
    
    # Load API key and create OpenAI client
    api_key = load_secrets("secrets.toml")
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()
    
    # Parse schema input
    try:
        schema_fields = parse_schema_input(schema_input, client)
    except Exception as e:
        print(f"Schema parsing error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Note: We keep the original user schema fields only
    # Source file information will be added to records but filtered out in final CSV
    
    # Find all PDF files in the folder
    pdf_folder_path = Path(pdf_folder)
    if not pdf_folder_path.exists():
        print(f"Error: PDF folder '{pdf_folder}' not found.", file=sys.stderr)
        sys.exit(1)
    
    pdf_files = list(pdf_folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{pdf_folder}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files to process using {parser_type} parser")
    
    # Determine output paths
    if output_csv:
        output_csv_path = output_csv
        output_dir = os.path.dirname(output_csv) or "."
        csv_filename = os.path.basename(output_csv)
        csv_name_without_ext = os.path.splitext(csv_filename)[0]
    else:
        folder_name = clean_title(pdf_folder_path.name)
        output_csv_path = f"{folder_name}_results.csv"
        output_dir = "."
        csv_name_without_ext = f"{folder_name}"
    
    # Create markdown output directory
    markdown_output_dir = os.path.join(output_dir, f"{csv_name_without_ext}_markdown")
    
    log_filename = os.path.join(output_dir, f"{csv_name_without_ext}_logging.json")
    
    print(f"Logging initialized. All prompt interactions will be saved to {log_filename}")
    
    # Process first PDF to generate schema definition
    first_pdf = pdf_files[0]
    print(f"Generating schema definition using {first_pdf.name}...")
    
    first_markdown = parse_pdf_to_markdown_only(str(first_pdf), markdown_output_dir, page_filter, parser_type)
    if not first_markdown:
        print(f"Failed to extract markdown from {first_pdf.name} for schema generation")
        sys.exit(1)
    
    # Generate schema definition
    schema_definition = Schema_Instruction(client, schema_fields, first_markdown, log_filename)
    
    # Process all PDF files
    all_extracted_data = []
    successful_files = 0
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        try:
            # Extract markdown from PDF
            markdown_content = parse_pdf_to_markdown_only(str(pdf_file), markdown_output_dir, page_filter, parser_type)
            
            if not markdown_content:
                print(f"Skipping {pdf_file.name} - no markdown content extracted")
                continue
            
            # Extract structured data from markdown
            extracted_records = process_markdown_with_schema(
                client=client,
                markdown_content=markdown_content,
                schema_fields=schema_fields,
                schema_definition=schema_definition,
                pdf_filename=pdf_file.name,
                log_filename=log_filename
            )
            
            all_extracted_data.extend(extracted_records)
            successful_files += 1
            print(f"Successfully processed {pdf_file.name}: {len(extracted_records)} records extracted")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
    
    print(f"\nProcessing complete:")
    print(f"- Successfully processed: {successful_files}/{len(pdf_files)} files")
    print(f"- Total records extracted: {len(all_extracted_data)}")
    
    if not all_extracted_data:
        print("No data extracted from any files; exiting.", file=sys.stderr)
        sys.exit(1)
    
    # Write results to CSV - strictly enforce user-specified schema only
    fieldnames = schema_fields
    
    # First, ensure ONLY user-specified schema fields are included
    schema_compliant_data = []
    unexpected_fields_found = set()
    
    for item in all_extracted_data:
        # Check for unexpected fields and warn
        item_fields = set(item.keys()) if isinstance(item, dict) else set()
        schema_fields_set = set(schema_fields)
        unexpected = item_fields - schema_fields_set
        if unexpected:
            unexpected_fields_found.update(unexpected)
        
        # Create row with ONLY the user-specified schema fields
        row = {k: item.get(k, "") for k in fieldnames}
        schema_compliant_data.append(row)
    
    # Warn about unexpected fields
    if unexpected_fields_found:
        print(f"Warning: Found unexpected fields not in user schema: {sorted(unexpected_fields_found)}")
        print(f"These fields will be excluded from the output CSV.")
    
    # Then, filter out empty rows after schema compliance check
    non_empty_data = []
    for row in schema_compliant_data:
        if not is_empty_row(row):
            non_empty_data.append(row)
    
    empty_rows_filtered = len(schema_compliant_data) - len(non_empty_data)
    print(f"Schema compliance: Using only user-specified fields: {fieldnames}")
    print(f"Filtered out {empty_rows_filtered} empty rows")
    
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in non_empty_data:
                writer.writerow(row)
    except Exception as e:
        print(f"Error writing CSV '{output_csv_path}': {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nResults written to: {output_csv_path}")
    print(f"Markdown files saved to: {markdown_output_dir}")
    print(f"All prompt interactions logged to: {log_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Process multiple PDF files in a folder and extract structured data to CSV."
    )
    
    parser.add_argument(
        "--pdf-folder", "-f", required=True,
        help="Path to folder containing PDF files to process."
    )
    
    parser.add_argument(
        "--schema", "-s", required=True,
        help="List of field names as a Python list string, e.g., \"['Title','Date','Location']\""
    )
    
    parser.add_argument(
        "--filter-page", 
        help="Specify which pages to process from each PDF. Accepts comma-separated page numbers and ranges (1-indexed). Example: \"1,5-10,20\""
    )
    
    parser.add_argument(
        "--output", "-o", 
        help="Path to output CSV file. If not provided, derives from folder name."
    )
    
    parser.add_argument(
        "--parser", 
        choices=["marker", "mineru"],
        default="marker",
        help="Parser to use for PDF processing. Choose 'marker' for marker processor or 'mineru' for MinerU processor (default: marker)"
    )
    
    args = parser.parse_args()
    
    try:
        process_pdf_folder(
            pdf_folder=args.pdf_folder,
            schema_input=args.schema,
            page_filter=args.filter_page,
            output_csv=args.output,
            parser_type=args.parser
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 