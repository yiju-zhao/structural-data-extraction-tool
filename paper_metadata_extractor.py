import argparse
import json
import os
import csv
import sys
from typing import Dict, Any, List

from openai import OpenAI

from data_extractor import (
    setup_logging,
    log_prompt_interaction,
    load_secrets,
    parse_schema_input,
    load_schema,
    Schema_Instruction,
    Extraction_Instruction,
    extract_with_openai,
    clean_title,
    is_empty_row
)


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Extract structured data from entire Markdown content and export to CSV.")
    parser.add_argument(
        "--markdown", "-m", required=True,
        help="Path to input Markdown file."
    )
    parser.add_argument(
        "--schema", "-s", required=True,
        help="List of field names as a Python list string, e.g., \"['Title','Date','Location']\""
    )
    parser.add_argument(
        "--output", "-o", required=False,
        help="Path to output CSV file. If not provided, derives from markdown filename."
    )
    parser.add_argument(
        "--model", required=False, default="gpt-4.1-nano-2025-04-14",
        help="OpenAI model to use for extraction"
    )
    parser.add_argument(
        "--prompt-json", action="store_true",
        help="Save prompt logging and schema JSON files (default: False)"
    )
    args = parser.parse_args()

    api_key = load_secrets("secrets.toml")
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    if args.output:
        output_csv_path = args.output
        output_dir = os.path.dirname(args.output) or "."
        # Extract base name for schema file
        csv_filename = os.path.basename(args.output)
        csv_name_without_ext = os.path.splitext(csv_filename)[0]
        # Remove _results suffix if present to get clean name
        if csv_name_without_ext.endswith("_results"):
            cleaned_name = csv_name_without_ext[:-8]  # Remove "_results"
        else:
            cleaned_name = csv_name_without_ext
    else:
        # Derive from markdown filename
        md_filename = os.path.basename(args.markdown)
        md_name_without_ext = os.path.splitext(md_filename)[0]
        cleaned_name = clean_title(md_name_without_ext)
        output_csv_path = f"{cleaned_name}_results.csv"
        output_dir = "."

    schema_output_path = os.path.join(output_dir, f"{cleaned_name}_schema.json") if args.prompt_json else None
    log_filename = os.path.join(output_dir, f"{cleaned_name}_logging.json") if args.prompt_json else None
    
    if args.prompt_json:
        print(f"Logging initialized. All prompt interactions will be saved to {log_filename}")

    try:
        schema_fields = parse_schema_input(args.schema, client, schema_output_path, log_filename)
    except Exception as e:
        print(f"Schema parsing error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.markdown, "r", encoding="utf-8") as f:
            md_text = f.read()
    except FileNotFoundError:
        print(f"Error: Markdown file '{args.markdown}' not found.", file=sys.stderr)
        sys.exit(1)

    if not md_text.strip():
        print("Markdown file is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing entire markdown content (Size: {len(md_text)} chars)...")

    print("Generating detailed schema definition...")
    sample_text = md_text[:1000] if len(md_text) > 1000 else md_text
    schema_definition = Schema_Instruction(client, schema_fields, sample_text, log_filename, args.model)
    
    extracted_list: List[Dict[str, Any]] = []
    
    try:
        print("Extracting data from entire markdown content...")
        parsed_list = extract_with_openai(
            client=client,
            schema_fields=schema_fields,
            section_text=md_text,
            schema_definition=schema_definition,
            log_filename=log_filename,
            model=args.model
        )
        
        num_records = len(parsed_list)
        print(f"Extracted {num_records} records from the markdown content")
        extracted_list.extend(parsed_list)
        
    except Exception as ex:
        print(f"Error extracting from markdown content: {ex}", file=sys.stderr)
        print(f"Content preview: {md_text[:200]}...", file=sys.stderr)

    print(f"\nTotal records extracted: {len(extracted_list)}")

    if not extracted_list:
        print("No data extracted; exiting.", file=sys.stderr)
        sys.exit(1)

    fieldnames = schema_fields
    
    schema_compliant_data = []
    unexpected_fields_found = set()
    
    for item in extracted_list:
        # Check for unexpected fields and warn
        item_fields = set(item.keys()) if isinstance(item, dict) else set()
        schema_fields_set = set(schema_fields)
        unexpected = item_fields - schema_fields_set
        if unexpected:
            unexpected_fields_found.update(unexpected)
        
        # Create row with ONLY the user-specified schema fields
        row = {k: item.get(k, "") for k in fieldnames}
        schema_compliant_data.append(row)
    
    if unexpected_fields_found:
        print(f"Warning: Found unexpected fields not in user schema: {sorted(unexpected_fields_found)}")
        print(f"These fields will be excluded from the output CSV.")
    
    # Then, filter out empty rows after schema compliance check
    non_empty_data = []
    for row in schema_compliant_data:
        if not is_empty_row(row):
            non_empty_data.append(row)
    
    empty_rows_filtered = len(schema_compliant_data) - len(non_empty_data)
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

    print(f"Extraction complete. CSV written to: {output_csv_path}")
    if args.prompt_json:
        print(f"All prompt interactions logged to: {log_filename}")


if __name__ == "__main__":
    main() 