import argparse
import json
import os
import csv
import sys
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
import fitz  # PyMuPDF
from tqdm import tqdm

from data_extractor import (
    load_secrets, parse_schema_input, find_sections_with_fallback,
    Schema_Instruction, Extraction_Instruction, extract_with_openai, log_prompt_interaction, setup_logging,
    is_empty_row, clean_title, interactive_schema_definition
)
from openai import OpenAI


def extract_first_page_text_only(pdf_path: str, should_truncate: bool = True) -> str:
    """
    Extracts text from the first page of a PDF and returns it as string.
    Optionally truncates the text before the 'Abstract' or 'Introduction' section.
    
    Args:
        pdf_path (str): The path to the input PDF file.
        should_truncate (bool): If True, apply the truncation logic.
    
    Returns:
        str: The extracted text content
    """
    try:
        # Open PDF and extract text from the first page
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        text = page.get_text()
        doc.close()

        # Process text based on the truncate flag
        if should_truncate:
            text = truncate_before_section(text)
        
        return text
        
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""


def truncate_before_section(text: str) -> str:
    """
    Truncates the text to keep only content before a specified section.
    Searches first for "Abstract", then for "Introduction" (case-insensitively).
    """
    import re
    
    # Pattern to find 'Abstract' at the start of a line
    abstract_pattern = re.compile(r'^\s*Abstract\b', re.IGNORECASE | re.MULTILINE)
    match = abstract_pattern.search(text)

    # If 'Abstract' is not found, search for 'Introduction'
    if not match:
        introduction_pattern = re.compile(r'^\s*Introduction\b', re.IGNORECASE | re.MULTILINE)
        match = introduction_pattern.search(text)

    # If a match was found, truncate the text
    if match:
        return text[:match.start()].strip()
    
    # Otherwise, return the original text
    return text


def save_markdown_file(content: str, pdf_filename: str, markdown_output_dir: str) -> str:
    """
    Save markdown content to file and return the file path.
    
    Args:
        content: Markdown content to save
        pdf_filename: Original PDF filename
        markdown_output_dir: Directory to save markdown files
    
    Returns:
        str: Path to saved markdown file
    """
    # Create output directory if it doesn't exist
    os.makedirs(markdown_output_dir, exist_ok=True)
    
    # Generate markdown filename
    pdf_name = os.path.splitext(pdf_filename)[0]
    cleaned_pdf_name = clean_title(pdf_name)
    md_filename = f"{cleaned_pdf_name}.md"
    md_output_path = os.path.join(markdown_output_dir, md_filename)
    
    # Write markdown content to file
    with open(md_output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return md_output_path


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
            
            return markdown_content
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""


def process_markdown_with_schema(
    client: OpenAI,
    markdown_content: str,
    schema_fields: List[str],
    schema_definition: str,
    extraction_instruction: str,
    pdf_filename: str,
    log_filename: str,
    model: str = "gpt-4.1-nano"
) -> List[Dict[str, Any]]:
    """
    Process markdown content and extract structured data according to schema.
    
    Args:
        client: OpenAI client
        markdown_content: The markdown content to process
        schema_fields: List of field names for the schema
        schema_definition: Detailed schema definition
        extraction_instruction: Pre-generated extraction instruction
        pdf_filename: Name of the source PDF file
        log_filename: Path to log file
    
    Returns:
        List of extracted records with source file information
    """
    if not markdown_content.strip():
        print(f"No markdown content found for {pdf_filename}")
        return []
    
    extracted_list = []
    try:
        # Include the detailed schema definition
        schema_info = f"""## Schema Definition
{schema_definition}

## Important Notes
- Keep spelling, casing, and punctuation exactly as they appear in the text.  
- Extract **only** information explicitly present.  
- For multi-value fields, return an **array** (unless the schema says otherwise).  
- Do **not** invent or omit keys; follow the schema structure strictly.  
- If a required value is missing, output an empty string (`""`) or empty array (`[]`) as appropriate.

## Expected JSON Output Format
Return a JSON object with this structure:
{{"items": [{{record1}}, {{record2}}, ...]}}

Where each record contains the following fields: {', '.join(schema_fields)}
"""

        messages = [
            {"role": "system", "content": extraction_instruction + "\n\n" + schema_info},
            {"role": "user", "content": markdown_content},
        ]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content
        
        # Parse the JSON response
        parsed_json = json.loads(response_content)
        
        # Extract the items array, handle different possible response formats
        if isinstance(parsed_json, dict) and "items" in parsed_json:
            result = parsed_json["items"]
        elif isinstance(parsed_json, list):
            result = parsed_json
        elif isinstance(parsed_json, dict):
            # If it's a single record, wrap it in a list
            result = [parsed_json]
        else:
            result = []
        
        # Ensure result is a list
        if not isinstance(result, list):
            result = [result] if result else []
        
        # Add source file information to each record
        for record in result:
            record['source_file'] = pdf_filename
        
        extracted_list.extend(result)
        
        # Log the interaction
        log_inputs = {
            "extraction_instruction": extraction_instruction,
            "schema_definition": schema_definition,
            "markdown_content": markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
            "model": model
        }
        log_prompt_interaction("process_markdown_with_schema", log_inputs, json.dumps(result, indent=2), log_filename)
        
    except json.JSONDecodeError as e:
        print(f"    Failed to parse JSON response from {pdf_filename}: {str(e)}")
        log_inputs = {
            "extraction_instruction": extraction_instruction,
            "schema_definition": schema_definition,
            "markdown_content": markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content,
            "error": f"JSON parsing error: {str(e)}",
            "raw_response": response_content
        }
        log_prompt_interaction("process_markdown_with_schema_json_error", log_inputs, f"JSON Error: {str(e)}", log_filename)
    except Exception as e:
        print(f"    Error extracting from {pdf_filename}: {str(e)}")
    
    return extracted_list


def process_markdown_with_schema_advanced(
    client: OpenAI,
    markdown_content: str,
    schema_fields: List[str],
    schema_definition: str,
    pdf_filename: str,
    log_filename: str,
    model: str = "gpt-4.1-nano"
) -> List[Dict[str, Any]]:
    """
    Process markdown content and extract structured data according to schema using advanced mode.
    
    Args:
        client: OpenAI client
        markdown_content: The markdown content to process
        schema_fields: List of field names for the schema
        schema_definition: Detailed schema definition
        pdf_filename: Name of the source PDF file
        log_filename: Path to log file
        model: OpenAI model to use
    
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
                model=model
            )
            
            # Add source file information to each record
            for record in parsed_list:
                record['source_file'] = pdf_filename
                record['source_section'] = f"section_{idx}"
            
            extracted_list.extend(parsed_list)
            
        except Exception as e:
            print(f"    Error extracting from section {idx} in {pdf_filename}: {str(e)}")
    
    return extracted_list


def process_pdf_folder(
    pdf_folder: str,
    schema_input: str,
    page_filter: str = None,
    output_csv: str = None,
    parser_type: str = "marker",
    mode: str = "advanced",
    model: str = "gpt-4.1-nano",
    prompt_json: bool = False
) -> None:
    """
    Process all PDF files in a folder and extract structured data to CSV.
    
    Args:
        pdf_folder: Path to folder containing PDF files
        schema_input: Schema definition as string
        page_filter: Optional page filter string
        output_csv: Output CSV file path
        parser_type: Parser to use - either "marker" or "mineru"
        mode: Processing mode - either "paper" or "advanced"
        model: OpenAI model to use
        prompt_json: Whether to save prompt interactions to JSON log file
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
    
    print(f"Found {len(pdf_files)} PDF files to process using {parser_type} parser in {mode} mode")
    
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
    
    # Set up logging filename only if prompt_json is enabled
    if prompt_json:
        log_filename = os.path.join(output_dir, f"{csv_name_without_ext}_logging.json")
        print(f"Prompt logging enabled. All prompt interactions will be saved to {log_filename}")
    else:
        log_filename = None
        print("Prompt logging disabled. No logging file will be created.")
    
    # Process first PDF to generate schema definition and extraction instruction (only once for the entire folder)
    first_pdf = pdf_files[0]
    print(f"Generating schema definition and extraction instruction using {first_pdf.name}...")
    
    if mode == "paper":
        # Use paper parser for first page text extraction
        first_content = extract_first_page_text_only(str(first_pdf), should_truncate=True)
        if not first_content:
            print(f"Failed to extract text from {first_pdf.name} for schema generation")
            sys.exit(1)
        
        # Save the markdown file for paper mode
        save_markdown_file(first_content, first_pdf.name, markdown_output_dir)
        
        # Generate schema definition interactively
        schema_definition = interactive_schema_definition(
            client=client,
            schema_fields=schema_fields,
            sample_text=first_content,
            cleaned_name=csv_name_without_ext,
            output_dir=output_dir,
            log_filename=log_filename,
            model=model
        )
        
        # Generate extraction instruction once
        print("Generating extraction instruction...")
        extraction_instruction = Extraction_Instruction(client, schema_definition, first_content, log_filename, model)
        
    else:  # advanced mode
        first_markdown = parse_pdf_to_markdown_only(str(first_pdf), markdown_output_dir, page_filter, parser_type)
        if not first_markdown:
            print(f"Failed to extract markdown from {first_pdf.name} for schema generation")
            sys.exit(1)
        
        # Generate schema definition
        schema_definition = Schema_Instruction(client, schema_fields, first_markdown, log_filename, model)
    
    # Initialize CSV file with headers and get already processed files
    # Add source_file to track which PDF each record came from for fault tolerance
    fieldnames = schema_fields + ['source_file']
    initialize_csv_file(output_csv_path, fieldnames)
    processed_files = get_processed_files_from_csv(output_csv_path)
    
    print(f"Found {len(processed_files)} already processed files in existing CSV")
    
    # Filter out already processed PDFs
    pdf_files_to_process = []
    skipped_csv = 0
    skipped_markdown = 0
    
    for pdf_file in pdf_files:
        # Check if already processed (exists in CSV)
        if pdf_file.name in processed_files:
            skipped_csv += 1
            continue
            
        # Check if markdown file already exists
        if check_markdown_file_exists(pdf_file.name, markdown_output_dir):
            # If markdown exists but not in CSV, we can still process the extraction part
            # without regenerating markdown
            pass
        
        pdf_files_to_process.append(pdf_file)
    
    print(f"Skipping {skipped_csv} files already in CSV")
    print(f"Processing {len(pdf_files_to_process)} remaining files")
    
    if not pdf_files_to_process:
        print("All files have already been processed!")
        return
    
    # Process remaining PDF files
    successful_files = 0
    
    for pdf_file in tqdm(pdf_files_to_process, desc="Processing PDFs", unit="file"):
        try:
            # Check if markdown file already exists
            markdown_exists = check_markdown_file_exists(pdf_file.name, markdown_output_dir)
            
            if mode == "paper":
                if markdown_exists:
                    # Read existing markdown file
                    pdf_name = os.path.splitext(pdf_file.name)[0]
                    cleaned_pdf_name = clean_title(pdf_name)
                    md_filename = f"{cleaned_pdf_name}.md"
                    md_path = os.path.join(markdown_output_dir, md_filename)
                    
                    try:
                        with open(md_path, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()
                    except Exception as e:
                        print(f"Error reading existing markdown for {pdf_file.name}: {e}")
                        # Fallback to regenerating
                        markdown_content = extract_first_page_text_only(str(pdf_file), should_truncate=True)
                        save_markdown_file(markdown_content, pdf_file.name, markdown_output_dir)
                else:
                    # Extract first page text from PDF
                    markdown_content = extract_first_page_text_only(str(pdf_file), should_truncate=True)
                    
                    if not markdown_content:
                        print(f"Skipping {pdf_file.name} - no text content extracted")
                        continue
                    
                    # Save markdown file
                    save_markdown_file(markdown_content, pdf_file.name, markdown_output_dir)
                
                # Extract structured data from markdown using paper mode processing
                extracted_records = process_markdown_with_schema(
                    client=client,
                    markdown_content=markdown_content,
                    schema_fields=schema_fields,
                    schema_definition=schema_definition,
                    extraction_instruction=extraction_instruction,
                    pdf_filename=pdf_file.name,
                    log_filename=log_filename,
                    model=model
                )
            else:  # advanced mode
                if markdown_exists:
                    # Read existing markdown file
                    pdf_name = os.path.splitext(pdf_file.name)[0]
                    cleaned_pdf_name = clean_title(pdf_name)
                    md_filename = f"{cleaned_pdf_name}.md"
                    md_path = os.path.join(markdown_output_dir, md_filename)
                    
                    try:
                        with open(md_path, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()
                    except Exception as e:
                        print(f"Error reading existing markdown for {pdf_file.name}: {e}")
                        # Fallback to regenerating
                        markdown_content = parse_pdf_to_markdown_only(str(pdf_file), markdown_output_dir, page_filter, parser_type)
                else:
                    # Extract markdown from PDF
                    markdown_content = parse_pdf_to_markdown_only(str(pdf_file), markdown_output_dir, page_filter, parser_type)
                
                if not markdown_content:
                    print(f"Skipping {pdf_file.name} - no markdown content extracted")
                    continue
                
                # Extract structured data from markdown using advanced mode processing
                extracted_records = process_markdown_with_schema_advanced(
                    client=client,
                    markdown_content=markdown_content,
                    schema_fields=schema_fields,
                    schema_definition=schema_definition,
                    pdf_filename=pdf_file.name,
                    log_filename=log_filename,
                    model=model
                )
            
            # Immediately append records to CSV for fault tolerance
            append_records_to_csv(output_csv_path, extracted_records, fieldnames)
            successful_files += 1
            
            # Print progress
            records_count = len([r for r in extracted_records if not is_empty_row({k: r.get(k, "") for k in schema_fields})])
            print(f"Extracted {records_count} records from {pdf_file.name}")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {str(e)}")
    
    # Calculate final statistics
    total_processed = len(processed_files) + successful_files
    
    # If all processing completed successfully, clean up the CSV to enforce user schema only
    if successful_files > 0 or total_processed == len(pdf_files):
        print(f"\nCleaning up CSV to enforce user-specified schema only...")
        
        # Read all data from the working CSV
        all_data = []
        unexpected_fields_found = set()
        
        if os.path.exists(output_csv_path):
            try:
                with open(output_csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Check for unexpected fields and warn
                        item_fields = set(row.keys()) if isinstance(row, dict) else set()
                        schema_fields_set = set(schema_fields)
                        unexpected = item_fields - schema_fields_set
                        if unexpected:
                            unexpected_fields_found.update(unexpected)
                        
                        # Create row with ONLY the user-specified schema fields
                        clean_row = {k: row.get(k, "") for k in schema_fields}
                        if not is_empty_row(clean_row):
                            all_data.append(clean_row)
            except Exception as e:
                print(f"Warning: Could not read CSV for cleanup: {e}")
        
        # Warn about unexpected fields that will be removed
        if unexpected_fields_found:
            print(f"Removing unexpected fields not in user schema: {sorted(unexpected_fields_found)}")
        
        # Write clean CSV with only user-specified schema fields
        try:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=schema_fields)
                writer.writeheader()
                for row in all_data:
                    writer.writerow(row)
            print(f"CSV cleaned: removed {len(unexpected_fields_found)} unexpected field types")
        except Exception as e:
            print(f"Warning: Could not clean CSV: {e}")
    
    # Count total records in final CSV
    total_records = len(all_data) if 'all_data' in locals() else 0
    if total_records == 0 and os.path.exists(output_csv_path):
        try:
            with open(output_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                total_records = sum(1 for row in reader)
        except Exception:
            pass
    
    print(f"\nProcessing complete:")
    print(f"- Total files in folder: {len(pdf_files)}")
    print(f"- Previously processed: {len(processed_files)}")
    print(f"- Newly processed: {successful_files}")
    print(f"- Total processed: {total_processed}")
    print(f"- Total records in final CSV: {total_records}")
    print(f"- Skipped (already in CSV): {skipped_csv}")
    
    print(f"\nResults available in: {output_csv_path}")
    print(f"Markdown files saved to: {markdown_output_dir}")
    if prompt_json and log_filename:
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
    
    parser.add_argument(
        "--mode", 
        choices=["paper", "advanced"],
        default="advanced",
        help="Processing mode. 'paper' mode extracts first page text and uses interactive schema definition. 'advanced' mode uses full PDF parsing and section-based extraction (default: advanced)"
    )
    
    parser.add_argument(
        "--model", 
        default="gpt-4.1-nano",
        help="OpenAI model to use for data extraction (default: gpt-4.1-nano)"
    )
    
    parser.add_argument(
        "--prompt-json", 
        action="store_true",
        help="Save all prompt interactions to a logging JSON file. If not set, no logging file will be created."
    )
    
    args = parser.parse_args()
    
    try:
        process_pdf_folder(
            pdf_folder=args.pdf_folder,
            schema_input=args.schema,
            page_filter=args.filter_page,
            output_csv=args.output,
            parser_type=args.parser,
            mode=args.mode,
            model=args.model,
            prompt_json=args.prompt_json
        )
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)


def get_processed_files_from_csv(csv_path: str) -> set:
    """
    Read existing CSV file and return a set of already processed PDF filenames.
    Note: This function expects a 'source_file' column that tracks which PDF each row came from.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Set of processed PDF filenames
    """
    processed_files = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Check if this is an old CSV without source_file column
                fieldnames = reader.fieldnames or []
                if 'source_file' not in fieldnames:
                    print("Warning: Existing CSV doesn't have 'source_file' column. Cannot determine which files were already processed.")
                    return processed_files
                    
                for row in reader:
                    if 'source_file' in row and row['source_file']:
                        processed_files.add(row['source_file'])
        except Exception as e:
            print(f"Warning: Could not read existing CSV file {csv_path}: {e}")
    return processed_files


def initialize_csv_file(csv_path: str, fieldnames: List[str]) -> None:
    """
    Initialize CSV file with headers if it doesn't exist.
    
    Args:
        csv_path: Path to the CSV file
        fieldnames: List of field names for the CSV header
    """
    if not os.path.exists(csv_path):
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        except Exception as e:
            print(f"Error creating CSV file {csv_path}: {e}")
            raise


def append_records_to_csv(csv_path: str, records: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """
    Append records to existing CSV file.
    
    Args:
        csv_path: Path to the CSV file
        records: List of records to append
        fieldnames: List of field names for the CSV (includes source_file)
    """
    if not records:
        return
        
    # Filter records to only include specified fields and non-empty rows
    schema_compliant_data = []
    for item in records:
        # Create row with all specified fieldnames (including source_file)
        row = {k: item.get(k, "") for k in fieldnames}
        
        # Check if row is empty (excluding source_file from emptiness check)
        schema_fields_only = {k: v for k, v in row.items() if k != 'source_file'}
        if not is_empty_row(schema_fields_only):
            schema_compliant_data.append(row)
    
    if schema_compliant_data:
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                for row in schema_compliant_data:
                    writer.writerow(row)
        except Exception as e:
            print(f"Error appending to CSV file {csv_path}: {e}")
            raise


def check_markdown_file_exists(pdf_filename: str, markdown_output_dir: str) -> bool:
    """
    Check if markdown file already exists for the given PDF.
    
    Args:
        pdf_filename: Name of the PDF file
        markdown_output_dir: Directory where markdown files are saved
        
    Returns:
        True if markdown file exists, False otherwise
    """
    pdf_name = os.path.splitext(pdf_filename)[0]
    cleaned_pdf_name = clean_title(pdf_name)
    md_filename = f"{cleaned_pdf_name}.md"
    md_path = os.path.join(markdown_output_dir, md_filename)
    return os.path.exists(md_path)


if __name__ == "__main__":
    main() 