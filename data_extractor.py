import argparse
import json
import os
import re
import csv
import sys
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


def load_schema(schema_path: str) -> Dict[str, str]:
    """
    Load a simple JSON schema file mapping field names → type strings.

    Example `schema.json`:
    {
      "name": "str",
      "date": "str",
      "participants": "list[str]"
    }
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    if not isinstance(schema, dict):
        raise ValueError("Schema file must be a JSON object mapping field names → type strings.")
    return schema


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
    system_prompt: str = "Extract the structured data according to the given schema."
) -> Dict[str, Any]:
    """
    Call OpenAI's chat completion endpoint with response_format=model_class to parse.
    Returns a Python dict corresponding to model_class.
    """
    schema_fields = []
    for name, field_info in model_class.model_fields.items():
        schema_fields.append(f"- `{name}`: {field_info.annotation!s}")
    schema_desc = "Schema:\n" + "\n".join(schema_fields)

    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + schema_desc},
        {"role": "user", "content": section_text},
    ]

    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
        response_format=model_class,
    )
    parsed = completion.choices[0].message.parsed
    return parsed.model_dump()


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
        help="Path to JSON schema file (field names → type strings)."
    )
    parser.add_argument(
        "--model_name", "-n", required=True,
        help="Name for the generated Pydantic model class."
    )
    parser.add_argument(
        "--output", "-o", required=False,
        help="Path to output CSV file. If not provided, derives from markdown filename."
    )
    parser.add_argument(
        "--system_prompt", default="Extract the structured data according to the given schema.",
        help="(Optional) System prompt to use for each API call."
    )
    args = parser.parse_args()

    # 1) Read API key from secrets.toml
    api_key = load_secrets("secrets.toml")
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI()

    # 2) Load JSON schema & build Pydantic model class
    try:
        schema_dict = load_schema(args.schema)
        model_cls = make_pydantic_model(args.model_name, schema_dict)
    except Exception as e:
        print(f"Schema/model creation error: {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Read Markdown and split into H1 sections
    try:
        with open(args.markdown, "r", encoding="utf-8") as f:
            md_text = f.read()
    except FileNotFoundError:
        print(f"Error: Markdown file '{args.markdown}' not found.", file=sys.stderr)
        sys.exit(1)

    # 4) Determine output CSV path
    if args.output:
        output_csv_path = args.output
    else:
        # Derive from markdown filename
        md_filename = os.path.basename(args.markdown)
        md_name_without_ext = os.path.splitext(md_filename)[0]
        cleaned_name = clean_title(md_name_without_ext)
        output_csv_path = f"{cleaned_name}_results.csv"

    sections = split_markdown_into_h1_sections(md_text)
    if not sections:
        print("No H1 (`# `) sections found in the Markdown file.", file=sys.stderr)
        sys.exit(1)

    # 5) Call OpenAI for each section to extract structured JSON
    extracted_list: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections, start=1):
        print(f"Processing section {idx}/{len(sections)}...")
        try:
            parsed_dict = extract_with_openai(
                client=client,
                model_class=model_cls,
                section_text=sec,
                system_prompt=args.system_prompt
            )
            extracted_list.append(parsed_dict)
        except ValidationError as ve:
            print(f"  → Validation error in section {idx}: {ve}", file=sys.stderr)
        except Exception as ex:
            print(f"  → Error extracting section {idx}: {ex}", file=sys.stderr)

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