import argparse
import json
import os
import re
import csv
import sys
import ast
from typing import Dict, Any, List, Type

import toml
from pydantic import BaseModel, create_model, ValidationError
from openai import OpenAI


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


def generate_schema_from_keys(client: OpenAI, keys: List[str]) -> Dict[str, str]:
    """
    Use OpenAI to generate a schema mapping field names to appropriate types
    based on the provided key names.
    """
    system_prompt = """You are a helpful assistant that generates JSON schemas for data extraction.
    Given a list of field names, you should output a JSON object mapping each field name to an appropriate data type.

    Available types:
    - "str" for text fields
    - "int" for integer numbers
    - "float" for decimal numbers
    - "bool" for true/false values
    - "list[str]" for lists of strings
    - "list[int]" for lists of integers

    Consider the semantic meaning of each field name to determine the most appropriate type.
    For example:
    - "Title", "Name", "Description", "Authors", "Organizers", "Participants" → "str"
    - "Age", "Count", "Year" → "int"
    - "Price", "Temperature" → "float"
    - "Active", "Published" → "bool"

    Output only a valid JSON object with no additional text."""

    user_prompt = f"Generate a schema for these field names: {keys}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages,
        response_format={"type": "json_object"},
    )

    schema_text = completion.choices[0].message.content
    try:
        schema = json.loads(schema_text)
        return schema
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse generated schema as JSON: {e}")


def parse_schema_input(schema_input: str, client: OpenAI, schema_output_path: str = None) -> Dict[str, str]:
    """
    Parse schema input which should be a list of field names.
    Generate schema using OpenAI and optionally save it.
    
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
    
    print("Generating schema from field names using OpenAI...")
    generated_schema = generate_schema_from_keys(client, keys)
    
    # Save the generated schema to a file if path is provided
    if schema_output_path:
        try:
            with open(schema_output_path, "w", encoding="utf-8") as f:
                json.dump(generated_schema, f, indent=2, ensure_ascii=False)
            print(f"Generated schema saved to: {schema_output_path}")
        except Exception as e:
            print(f"Warning: Could not save generated schema to '{schema_output_path}': {e}", file=sys.stderr)
    
    return generated_schema


def load_schema(schema_path: str, client: OpenAI = None, schema_output_path: str = None) -> Dict[str, str]:
    """
    Load schema from a JSON file. The file can contain either:
    1. A full schema mapping field names → type strings (original format)
    2. A list of field names that will be converted to a schema using OpenAI

    Example full schema:
    {
      "name": "str",
      "date": "str",
      "participants": "list[str]"
    }

    Example key-only format:
    ["name", "date", "participants"]
    
    If a schema is generated from keys, it will be saved to schema_output_path.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        # Full schema format - return as is
        return data
    elif isinstance(data, list):
        # Key-only format - generate schema using OpenAI
        if client is None:
            raise ValueError("OpenAI client is required when schema file contains only keys")
        print("Generating schema from field names using OpenAI...")
        generated_schema = generate_schema_from_keys(client, data)
        
        # Save the generated schema to a file if path is provided
        if schema_output_path:
            try:
                with open(schema_output_path, "w", encoding="utf-8") as f:
                    json.dump(generated_schema, f, indent=2, ensure_ascii=False)
                print(f"Generated schema saved to: {schema_output_path}")
            except Exception as e:
                print(f"Warning: Could not save generated schema to '{schema_output_path}': {e}", file=sys.stderr)
        
        return generated_schema
    else:
        raise ValueError("Schema file must contain either a JSON object (full schema) or a JSON array (field names only).")


def python_type_from_str(type_str: str):
    """
    Convert a type‐string (e.g. "str", "int", "list[str]") to an actual Python type for Pydantic.
    """
    basic_map = {
        "str": (str, ...),
        "int": (int, ...),
        "float": (float, ...),
        "bool": (bool, ...),
    }
    if type_str in basic_map:
        return basic_map[type_str]

    list_match = re.fullmatch(r"list\[(.+)\]", type_str.strip())
    if list_match:
        inner = list_match.group(1).strip()
        if inner not in basic_map:
            raise ValueError(f"Unsupported inner list type: {inner}")
        py_inner, ell = basic_map[inner]
        return (List[py_inner], ...)
    raise ValueError(f"Unsupported type string in schema: {type_str}")


def make_pydantic_model(model_name: str, schema: Dict[str, str]) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model with name `model_name` from the schema dict.
    """
    fields = {}
    for field_name, type_str in schema.items():
        try:
            fields[field_name] = python_type_from_str(type_str)
        except ValueError as e:
            raise ValueError(f"Error parsing type for field '{field_name}': {e}")
    return create_model(model_name, **fields)  # type: ignore


def Schema_Instruction(
    client: OpenAI,
    generated_schema: Dict[str, str],
    sample_text: str
) -> str:
    """
    Generate a detailed schema definition by analyzing the schema and sample text.
    Returns a string with detailed field descriptions.
    """
    system_prompt = """Create a schema definition by analyzing the schema and sample text. 

## Schema
```json
{generated_schema}
```

## Analyze the type and definition of the schema from the sample text
```text
{sample_text}
```

1. For every **top-level field** in `{generated_schema}`, provide a bullet like:  
   - **`fieldName`**: *description* — expected **type** & precise content to pull from the text.  
   - If the field is an **object** or **array**, briefly outline its inner keys/items.

## Important Notes
- Keep spelling, casing, and punctuation exactly as they appear in the text.  
- Extract **only** information explicitly present.  
- For multi-value fields, return an **array** (unless the schema says otherwise).  
- Do **not** invent or omit keys; follow the schema structure strictly.  
- If a required value is missing, output an empty string (`""`) or empty array (`[]`) as appropriate.

# Output Format

Return the schema definition, where the key is the schema itself, and the value is the description or definition of it. Below is a sample output:

1. `authors`: An array of objects, each with:
   - `name`: Author's full name (string)
   - `affiliations`: Array of affiliation markers (numbers or letters) associated with this author

2. `affiliations`: An array of objects, each with:
   - `id`: The affiliation marker (number or letter)
   - `name`: Name of the institution/organization
  
3. `keywords`: An array of 5-8 important keywords or topics from the abstract"""

    # Format the prompt with actual values
    formatted_prompt = system_prompt.format(
        generated_schema=json.dumps(generated_schema, indent=2),
        sample_text=sample_text
    )

    messages = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": "Generate the detailed schema definition based on the provided schema and sample text."},
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=messages,
    )

    return completion.choices[0].message.content


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
    model_class: Type[BaseModel],
    section_text: str,
    schema_definition: str,
) -> List[Dict[str, Any]]:
    """
    Call OpenAI's chat completion endpoint to parse multiple records from a section.
    Returns a list of Python dicts corresponding to model_class.
    """
    # Include the detailed schema definition
    schema_info = f"""## Schema Definition
{schema_definition}
"""

    enhanced_system_prompt = f"""Extract the structured data according to the given schema.

IMPORTANT: This section may contain MULTIPLE events/sessions/records. Please extract ALL records from the section, not just the first one. Return an array of objects, where each object represents one complete record according to the schema.

If a section contains multiple events, sessions, papers, or items, extract each one as a separate record in the array.

For example, if the section contains:
- Multiple time slots with different events
- Multiple numbered papers or presentations  
- Multiple sessions with different details

Extract each one as a separate object in the response array.

CRITICAL: - Do not truncate or skip any records. Process the ENTIRE section and extract ALL valid records.
          - Maintain the exact spelling and formatting of names.
         - Extract only information explicitly stated in the text"""

    messages = [
        {"role": "system", "content": enhanced_system_prompt + "\n\n" + schema_info},
        {"role": "user", "content": section_text},
    ]

    # Create a wrapper model that contains a list of the original model
    list_model_name = f"List{model_class.__name__}"
    list_fields = {"items": (List[model_class], ...)}
    ListModel = create_model(list_model_name, **list_fields)

    try:
        completion = client.beta.chat.completions.parse(
            model="o4-mini-2025-04-16",
            messages=messages,
            response_format=ListModel
        )
        
        parsed = completion.choices[0].message.parsed
        # Return list of dictionaries instead of a single dictionary
        result = [item.model_dump() for item in parsed.items]
        
        # Validate that we got some results
        if len(result) == 0:
            print(f"    Warning: No records extracted, section might be empty or malformed")
        
        return result
        
    except Exception as e:
        print(f"    Failed with extraction: {str(e)}")
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

    # 3) Parse schema input and build Pydantic model class
    try:
        schema_dict = parse_schema_input(args.schema, client, schema_output_path)
        model_cls = make_pydantic_model("ExtractedData", schema_dict)
    except Exception as e:
        print(f"Schema/model creation error: {e}", file=sys.stderr)
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
    schema_definition = Schema_Instruction(client, schema_dict, sections[0])
    
    print(f"Found {len(sections)} H1 sections to process")
    total_records_extracted = 0
    
    for idx, sec in enumerate(sections, start=1):
        section_size = len(sec)
        print(f"Processing section {idx}/{len(sections)} (Size: {section_size} chars)...")
        print(f"Section title: {sec.split(chr(10))[0][:100]}...")
        
        try:
            parsed_list = extract_with_openai(
                client=client,
                model_class=model_cls,
                section_text=sec,
                schema_definition=schema_definition,
            )
            
            num_records = len(parsed_list)
            print(f"Extracted {num_records} records from section {idx}")
            total_records_extracted += num_records
            extracted_list.extend(parsed_list)
            
        except ValidationError as ve:
            print(f"Validation error in section {idx}: {ve}", file=sys.stderr)
            print(f"Section content preview: {sec[:200]}...", file=sys.stderr)
        except Exception as ex:
            print(f"Error extracting section {idx}: {ex}", file=sys.stderr)
            print(f"Section content preview: {sec[:200]}...", file=sys.stderr)

    print(f"\nTotal records extracted: {total_records_extracted} from {len(sections)} sections")

    if not extracted_list:
        print("No data extracted; exiting.", file=sys.stderr)
        sys.exit(1)

    # 6) Write results to CSV
    fieldnames = list(model_cls.model_fields.keys())
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


if __name__ == "__main__":
    main()