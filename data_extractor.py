import argparse
import json
import os
import re
import csv
import sys
import ast
import logging
from datetime import datetime
from typing import Dict, Any, List

import toml
from openai import OpenAI


def setup_logging():
    """
    Setup logging configuration to store all prompt interactions in logging.json
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logging.json', mode='a'),
            logging.StreamHandler()
        ]
    )


def log_prompt_interaction(function_name: str, inputs: Dict[str, Any], output: str, log_filename: str = "logging.json"):
    """
    Log prompt inputs and outputs to specified logging file
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "function": function_name,
        "inputs": inputs,
        "output": output
    }
    
    # Write to logging file
    try:
        # Read existing logs
        if os.path.exists(log_filename):
            with open(log_filename, 'r', encoding='utf-8') as f:
                try:
                    existing_logs = json.load(f)
                    if not isinstance(existing_logs, list):
                        existing_logs = []
                except json.JSONDecodeError:
                    existing_logs = []
        else:
            existing_logs = []
        
        # Append new log entry
        existing_logs.append(log_entry)
        
        # Write back to file
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Warning: Could not write to {log_filename}: {e}", file=sys.stderr)


def load_secrets(toml_path: str = "secrets.toml") -> str:
    """
    Load OPENAI_API_KEY from a TOML file in the current folder.
    Expects:
        secrets.toml
        └─ OPENAI_API_KEY = "sk-..."
    """
    if not os.path.exists(toml_path):
        print(f"Error: '{toml_path}' not found. Please create it with your OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    try:
        data = toml.load(toml_path)
    except Exception as e:
        print(f"Error parsing '{toml_path}': {e}", file=sys.stderr)
        sys.exit(1)

    api_key = data.get("OPENAI_API_KEY")
    if not api_key:
        print(f"Error: 'OPENAI_API_KEY' not set in '{toml_path}'.", file=sys.stderr)
        sys.exit(1)

    return api_key


def parse_schema_input(schema_input: str, client: OpenAI, schema_output_path: str = None, log_filename: str = "logging.json") -> List[str]:
    """
    Parse schema input which should be a list of field names.
    
    Example input: "['Title','Organizers','Date','Time','Location']"
    """
    try:
        # Parse the string as a Python literal (list)
        keys = ast.literal_eval(schema_input)
        if not isinstance(keys, list):
            raise ValueError("Schema input must be a list of field names")
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("All field names must be strings")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid schema format. Expected a list like ['Title','Date','Location']. Error: {e}")
    
    print("Using provided field names for schema...")
    schema_fields = keys

    # Save the schema fields to a file if path is provided
    if schema_output_path:
        try:
            with open(schema_output_path, "w", encoding="utf-8") as f:
                json.dump(schema_fields, f, indent=2, ensure_ascii=False)
            print(f"Schema fields saved to: {schema_output_path}")
        except Exception as e:
            print(f"Warning: Could not save schema to '{schema_output_path}': {e}", file=sys.stderr)
    
    return schema_fields


def load_schema(schema_path: str, client: OpenAI = None, schema_output_path: str = None, log_filename: str = "logging.json") -> List[str]:
    """
    Load schema from a JSON file. The file should contain a list of field names.

    Example format:
    ["name", "date", "participants"]
    
    If a schema is generated from keys, it will be saved to schema_output_path.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Field names format - return as is
        return data
    elif isinstance(data, dict):
        # Old full schema format - extract keys
        print("Converting old schema format to field names list...")
        schema_fields = list(data.keys())
        
        # Save the converted schema to a file if path is provided
        if schema_output_path:
            try:
                with open(schema_output_path, "w", encoding="utf-8") as f:
                    json.dump(schema_fields, f, indent=2, ensure_ascii=False)
                print(f"Converted schema saved to: {schema_output_path}")
            except Exception as e:
                print(f"Warning: Could not save converted schema to '{schema_output_path}': {e}", file=sys.stderr)
        
        return schema_fields
    else:
        raise ValueError("Schema file must contain a JSON array of field names.")


def Schema_Instruction(
    client: OpenAI,
    schema_fields: List[str],
    sample_text: str,
    log_filename: str = "logging.json"
) -> str:
    """
    Generate a detailed schema definition by analyzing the schema fields and sample text.
    Returns a string with detailed field descriptions.
    """
    system_prompt = """Create a schema definition by analyzing the schema fields and sample text. 

## Schema Fields
{schema_fields}

## Analyze the type and definition of the schema from the sample text
```text
{sample_text}
```

1. For every **field** in the schema, provide a bullet like:  
   - **`fieldName`**: *description* — expected **type** & precise content to pull from the text.  
   - If the field is expected to contain multiple items, describe it as an array.

## Important Notes
- Keep spelling, casing, and punctuation exactly as they appear in the text.  
- Extract **only** information explicitly present.  
- For multi-value fields, return an **array** (unless the schema says otherwise).  
- Do **not** invent or omit keys; follow the schema structure strictly.  
- If a required value is missing, output an empty string (`""`) or empty array (`[]`) as appropriate.

# Output Format

Return the schema definition, where each field is described with its purpose and expected format. Below is a sample output:

1. `authors`: An array of author names as strings
   
2. `affiliations`: An array of institution/organization names as strings
  
3. `keywords`: An array of 5-8 important keywords or topics from the abstract as strings"""

    # Format the prompt with actual values
    formatted_prompt = system_prompt.format(
        schema_fields=json.dumps(schema_fields, indent=2),
        sample_text=sample_text
    )

    messages = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": "Generate the detailed schema definition based on the provided schema fields and sample text."},
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages,
    )

    result = completion.choices[0].message.content
    
    # Log the interaction
    log_inputs = {
        "schema_fields": schema_fields,
        "sample_text": sample_text[:500] + "..." if len(sample_text) > 500 else sample_text,  # Truncate long sample text for logging
        "formatted_prompt": formatted_prompt,
        "model": "gpt-4.1-2025-04-14"
    }
    log_prompt_interaction("Schema_Instruction", log_inputs, result, log_filename)

    return result


def Extraction_Instruction(
    client: OpenAI,
    schema_fields: List[str],
    sample_text: str,
    log_filename: str = "logging.json"
) -> str:
    """
    Generate a dynamic system prompt for data extraction based on the schema fields and sample text.
    Returns a system prompt that will be used for extraction.
    """
    system_prompt_template = """Analyze the provided schema fields and sample text to create an optimized extraction prompt.

## Schema Fields
{schema_fields}

## Sample Text
```text
{sample_text}
```

Based on the schema structure and sample text patterns, generate a comprehensive system prompt for data extraction that includes:

1. Clear instructions for extracting structured data according to the schema
2. Specific guidance based on the content patterns observed in the sample text
3. Instructions for handling multiple records if the text contains multiple items
4. Field-specific extraction guidelines based on the schema fields
5. Error handling instructions for missing or malformed data

CRITICAL REQUIREMENTS that must be included in the generated prompt:
- IMPORTANT: This section may contain MULTIPLE records. Please extract ALL records from the section, not just the first one. Return a JSON object with an "items" key containing an array of objects, where each object represents one complete record according to the schema.
- If a section contains multiple records, extract each one as a separate record in the array.
- CRITICAL: Do not truncate or skip any records. Process the ENTIRE section and extract ALL valid records.
- CRITICAL: Maintain the exact spelling and formatting of names.
- CRITICAL: Extract only information explicitly stated in the text.
- Return the response as a JSON object with this structure: {{"items": [{{record1}}, {{record2}}, ...]}}

The generated prompt should be optimized for the specific data structure and content type shown in the sample text and must emphasize complete extraction of all records.
Output only the system prompt text that will be used for extraction, without any additional formatting or explanations."""

    # Format the prompt with actual values
    formatted_prompt = system_prompt_template.format(
        schema_fields=json.dumps(schema_fields, indent=2),
        sample_text=sample_text[:2000] + "..." if len(sample_text) > 2000 else sample_text
    )

    messages = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": "Generate the optimized extraction system prompt based on the schema fields and sample text."},
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages,
    )

    result = completion.choices[0].message.content
    
    # Log the interaction
    log_inputs = {
        "schema_fields": schema_fields,
        "sample_text": sample_text[:500] + "..." if len(sample_text) > 500 else sample_text,  # Truncate long sample text for logging
        "formatted_prompt": formatted_prompt,
        "model": "gpt-4.1-2025-04-14"
    }
    log_prompt_interaction("Extraction_Instruction", log_inputs, result, log_filename)

    return result


def split_markdown_into_h1_sections(md_text: str) -> List[str]:
    """
    Split the markdown text into a list of H1-level sections.
    Each element starts with the "# " heading line and includes everything until the next "# ".
    """
    pattern = re.compile(r"(?m)(?=^# )")
    parts = pattern.split(md_text)
    if parts and not parts[0].lstrip().startswith("#"):
        parts = parts[1:]
    sections = [part.rstrip() for part in parts if part.strip()]
    return sections


def extract_with_openai(
    client: OpenAI,
    schema_fields: List[str],
    section_text: str,
    schema_definition: str,
    log_filename: str = "logging.json",
) -> List[Dict[str, Any]]:
    """
    Call OpenAI's chat completion endpoint to parse multiple records from a section.
    Returns a list of Python dicts.
    """
    # Use the Extraction_Instruction to generate a dynamic system prompt
    enhanced_system_prompt = Extraction_Instruction(client, schema_fields, section_text[:1000], log_filename)

    # Include the detailed schema definition
    schema_info = f"""## Schema Definition
{schema_definition}

## Expected JSON Output Format
Return a JSON object with this structure:
{{"items": [{{record1}}, {{record2}}, ...]}}

Where each record contains the following fields: {', '.join(schema_fields)}
"""

    messages = [
        {"role": "system", "content": enhanced_system_prompt + "\n\n" + schema_info},
        {"role": "user", "content": section_text},
    ]

    try:
        completion = client.chat.completions.create(
            model="o4-mini-2025-04-16",
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
        
        # Log the interaction
        log_inputs = {
            "enhanced_system_prompt": enhanced_system_prompt,
            "schema_definition": schema_definition,
            "section_text": section_text[:500] + "..." if len(section_text) > 500 else section_text,  # Truncate for logging
            "model": "o4-mini-2025-04-16"
        }
        log_prompt_interaction("extract_with_openai", log_inputs, json.dumps(result, indent=2), log_filename)
        
        # Validate that we got some results
        if len(result) == 0:
            print(f"    Warning: No records extracted, section might be empty or malformed")
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"    Failed to parse JSON response: {str(e)}")
        # Log the error
        log_inputs = {
            "enhanced_system_prompt": enhanced_system_prompt,
            "schema_definition": schema_definition,
            "section_text": section_text[:500] + "..." if len(section_text) > 500 else section_text,
            "model": "o4-mini-2025-04-16",
            "error": f"JSON parsing error: {str(e)}",
            "raw_response": response_content
        }
        log_prompt_interaction("extract_with_openai_json_error", log_inputs, f"JSON Error: {str(e)}", log_filename)
        return []
    except Exception as e:
        print(f"    Failed with extraction: {str(e)}")
        # Log the error
        log_inputs = {
            "enhanced_system_prompt": enhanced_system_prompt,
            "schema_definition": schema_definition,
            "section_text": section_text[:500] + "..." if len(section_text) > 500 else section_text,
            "model": "o4-mini-2025-04-16",
            "error": str(e)
        }
        log_prompt_interaction("extract_with_openai_error", log_inputs, f"Error: {str(e)}", log_filename)
        return []  # Return empty list instead of raising error


def clean_title(title: str) -> str:
    """Clean the title by replacing non-alphanumeric characters with underscores."""
    # Replace all non-alphanumeric characters (except for underscores) with underscores
    cleaned = re.sub(r'[^\w\d]', '_', title)
    # Replace consecutive underscores with a single underscore
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned


def main():
    # Initialize logging
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Extract structured data from Markdown and export to CSV.")
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
    args = parser.parse_args()

    # 1) Read API key from secrets.toml
    api_key = load_secrets("secrets.toml")
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    # 2) Determine output paths and create cleaned name for consistent naming
    if args.output:
        output_csv_path = args.output
        # Extract directory for schema file
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

    schema_output_path = os.path.join(output_dir, f"{cleaned_name}_schema.json")
    log_filename = os.path.join(output_dir, f"{cleaned_name}_logging.json")
    
    print(f"Logging initialized. All prompt interactions will be saved to {log_filename}")

    # 3) Parse schema input to get field names
    try:
        schema_fields = parse_schema_input(args.schema, client, schema_output_path, log_filename)
    except Exception as e:
        print(f"Schema parsing error: {e}", file=sys.stderr)
        sys.exit(1)

    # 4) Read Markdown and split into H1 sections
    try:
        with open(args.markdown, "r", encoding="utf-8") as f:
            md_text = f.read()
    except FileNotFoundError:
        print(f"Error: Markdown file '{args.markdown}' not found.", file=sys.stderr)
        sys.exit(1)

    sections = split_markdown_into_h1_sections(md_text)
    if not sections:
        print("No H1 (`# `) sections found in the Markdown file.", file=sys.stderr)
        sys.exit(1)

    # 5) Call OpenAI for each section to extract structured JSON
    extracted_list: List[Dict[str, Any]] = []
    
    # Generate schema definition once using the first section as sample
    print("Generating detailed schema definition...")
    schema_definition = Schema_Instruction(client, schema_fields, sections[0], log_filename)
    
    print(f"Found {len(sections)} H1 sections to process")
    
    total_records_extracted = 0
    
    for idx, sec in enumerate(sections, start=1):
        section_size = len(sec)
        print(f"Processing section {idx}/{len(sections)} (Size: {section_size} chars)...")
        print(f"Section title: {sec.split(chr(10))[0][:100]}...")
        
        try:
            parsed_list = extract_with_openai(
                client=client,
                schema_fields=schema_fields,
                section_text=sec,
                schema_definition=schema_definition,
                log_filename=log_filename,
            )
            
            num_records = len(parsed_list)
            print(f"Extracted {num_records} records from section {idx}")
            total_records_extracted += num_records
            extracted_list.extend(parsed_list)
            
        except Exception as ex:
            print(f"Error extracting section {idx}: {ex}", file=sys.stderr)
            print(f"Section content preview: {sec[:200]}...", file=sys.stderr)

    print(f"\nTotal records extracted: {total_records_extracted} from {len(sections)} sections")

    if not extracted_list:
        print("No data extracted; exiting.", file=sys.stderr)
        sys.exit(1)

    # 6) Write results to CSV
    fieldnames = schema_fields
    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in extracted_list:
                row = {k: item.get(k, "") for k in fieldnames}
                writer.writerow(row)
    except Exception as e:
        print(f"Error writing CSV '{output_csv_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extraction complete. CSV written to: {output_csv_path}")
    print(f"All prompt interactions logged to: {log_filename}")


if __name__ == "__main__":
    main()