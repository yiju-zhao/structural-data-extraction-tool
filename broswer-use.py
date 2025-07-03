import asyncio
import argparse
import threading
import csv
import json
import os
import sys
import re
import time
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field, create_model, validator
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import ast

load_dotenv()

# Disable browser_use telemetry and sync to avoid cloud warnings  
os.environ["BROWSER_USE_TELEMETRY"] = "false"
os.environ["BROWSER_USE_SYNC"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from browser_use import Agent, Controller
from langchain_openai import ChatOpenAI

# Thread-safe CSV writing
csv_writer_lock = threading.Lock()

def load_secrets(toml_path: str = "secrets.toml") -> str:
    """Load OPENAI_API_KEY from environment variable or a TOML file."""
    # First try environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Fallback to TOML file
    if os.path.exists(toml_path):
        import toml
        try:
            data = toml.load(toml_path)
            api_key = data.get("OPENAI_API_KEY")
            if api_key:
                return api_key
        except Exception as e:
            print(f"Error loading {toml_path}: {e}")
    
    print("Error: OPENAI_API_KEY not found in environment variables or secrets.toml.")
    sys.exit(1)

def parse_schema_input(schema_input: str) -> List[str]:
    """Parse schema input which should be a list of field names."""
    try:
        # Parse the string as a Python literal (list)
        keys = ast.literal_eval(schema_input)
        if not isinstance(keys, list):
            raise ValueError("Schema input must be a list of field names")
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("All field names must be strings")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid schema format. Expected a list like ['Title','DOI','Authors','Affiliations']. Error: {e}")
    
    return keys

def clean_filename(filename: str) -> str:
    """Clean filename to be alphanumeric only, as required by browser_use library."""
    import re
    # Keep only alphanumeric characters and common separators
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '', filename)
    # Ensure it doesn't start with special characters
    clean_name = re.sub(r'^[^a-zA-Z0-9]+', '', clean_name)
    return clean_name if clean_name else "output"

def generate_browser_task_prompt(client: OpenAI, url: str, schema_fields: List[str], model: str = "gpt-4.1-mini") -> str:
    """Generate a dynamic task prompt for browser-use based on URL and schema fields."""
    
    system_prompt = f"""You are an expert at creating browser automation task prompts. Given a URL and a list of data fields to extract, create a comprehensive and detailed task prompt for browser automation.

URL to scrape: {url}
Data fields to extract: {schema_fields}

Generate a detailed task prompt that:
1. Instructs the browser automation agent to navigate to the URL
2. Uses a SINGLE COMPREHENSIVE EXTRACTION strategy: scroll to load all content, then extract everything at once
3. Identifies and extracts ALL instances/records of the specified data fields from the ENTIRE page
4. Handles different page layouts and structures, including dynamic loading
5. Explicitly mentions scrolling to TOP, then BOTTOM to load all content, then extracting ALL data
6. Extracts the complete dataset from the entire page in ONE operation
7. Ensures all content is loaded before performing the comprehensive extraction
8. Specifies the expected output format with proper field handling
9. MUST return ALL data from the complete page in the exact schema format - NOT create files
10. Must use the 'done' action with ALL extracted data in proper JSON format

CRITICAL: The agent must capture ALL data from the COMPLETE page in ONE comprehensive extraction, not partial sections.

The task prompt should be clear, actionable, and comprehensive for browser automation."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create a comprehensive browser automation task prompt for extracting ALL {', '.join(schema_fields)} from the complete page at {url}. Emphasize PROGRESSIVE extraction starting from the TOP, accumulating data throughout the entire page scrolling process, and returning ALL accumulated structured data (NOT creating files). The output must be returned via the 'done' action in proper JSON schema format containing ALL data from the entire page."}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return completion.choices[0].message.content

def generate_schema_definition(client: OpenAI, url: str, schema_fields: List[str], model: str = "gpt-4.1-mini") -> str:
    """Generate a detailed schema definition based on URL and field names."""
    
    system_prompt = f"""You are an expert at creating data schema definitions. Given a URL and list of field names, create a detailed schema definition that describes what each field should contain.

URL context: {url}
Schema fields: {schema_fields}

For each field, provide:
1. A clear description of what data it should contain
2. The expected data type (all fields should be strings for CSV compatibility)
3. Format specifications (e.g., comma-separated for multiple values)

Important: All fields must be STRING type for proper CSV export. Arrays should be converted to delimited strings.

Format your response as a structured definition that clearly explains each field's purpose and expected content based on the context of the URL."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create a detailed schema definition for fields {', '.join(schema_fields)} in the context of data from {url}. All fields must be strings."}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return completion.choices[0].message.content

def create_dynamic_pydantic_models(schema_fields: List[str]):
    """Create Pydantic models dynamically based on schema fields."""
    
    # Create field definitions for the item model - ALL FIELDS AS STRINGS
    field_definitions = {}
    for field in schema_fields:
        # Ensure all fields are strings for CSV compatibility
        field_definitions[field] = (str, Field(default="", description=f"The {field.lower()} of the item as a string"))
    
    # Create the individual item model with string validation
    class ItemModel(BaseModel):
        class Config:
            extra = "allow"
        
        @validator('*', pre=True)
        def convert_to_string(cls, v):
            """Convert any field to string to ensure CSV compatibility."""
            if isinstance(v, list):
                # Convert lists to comma-separated strings for authors or semicolon for affiliations
                if isinstance(v, list) and len(v) > 0:
                    if any(keyword in str(v[0]).lower() for keyword in ['university', 'institute', 'college', 'lab']):
                        return '; '.join(str(item) for item in v)  # Affiliations with semicolon
                    else:
                        return ', '.join(str(item) for item in v)  # Authors with comma
                return ', '.join(str(item) for item in v)
            elif isinstance(v, dict):
                return str(v)
            elif v is None:
                return ""
            else:
                return str(v)
    
    # Add fields dynamically to the model
    for field, field_type in field_definitions.items():
        setattr(ItemModel, field, field_type[1])
    
    # Recreate with proper annotations
    ItemModel.__annotations__ = {field: str for field in schema_fields}
    
    # Create the container model with a list of items
    class ContainerModel(BaseModel):
        items: List[ItemModel] = Field(description="List of extracted items")
    
    return ItemModel, ContainerModel

class ThreadSafeCSVWriter:
    """Thread-safe CSV writer for real-time data storage during browser automation."""
    
    def __init__(self, filename: str, fieldnames: List[str]):
        self.filename = filename
        self.fieldnames = fieldnames
        self.lock = threading.Lock()
        self.file_initialized = False
        
    def write_header(self):
        """Write CSV header if not already written."""
        with self.lock:
            if not self.file_initialized:
                with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()
                self.file_initialized = True
                print(f"📝 CSV header written to {self.filename}")
    
    def write_row(self, row_data: Dict[str, Any]):
        """Write a single row to CSV in a thread-safe manner."""
        with self.lock:
            try:
                with open(self.filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writerow(row_data)
                print(f"✅ Row written to CSV: {list(row_data.keys())}")
            except Exception as e:
                print(f"❌ Error writing row to CSV: {e}")
    
    def write_batch(self, rows: List[Dict[str, Any]]):
        """Write multiple rows to CSV in a thread-safe manner."""
        with self.lock:
            try:
                with open(self.filename, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    for row in rows:
                        writer.writerow(row)
                print(f"✅ {len(rows)} rows written to CSV")
            except Exception as e:
                print(f"❌ Error writing batch to CSV: {e}")

def process_result_data(result, schema_fields: List[str]) -> List[Dict[str, str]]:
    """Process result data and ensure all fields are strings."""
    items = []
    
    # Handle different result formats
    if hasattr(result, 'items') and result.items:
        # Direct pydantic model
        items = result.items
    elif isinstance(result, dict) and 'items' in result:
        # Dictionary format
        items = result['items']
    elif isinstance(result, str):
        # String that needs JSON parsing
        try:
            parsed_result = json.loads(result)
            if 'items' in parsed_result:
                items = parsed_result['items']
            elif isinstance(parsed_result, list):
                items = parsed_result
        except json.JSONDecodeError:
            # Try to extract from markdown-like format
            try:
                # Look for JSON-like structures in the string
                json_match = re.search(r'\[[\s\S]*\]', result)
                if json_match:
                    json_str = json_match.group(0)
                    items = json.loads(json_str)
            except:
                pass
    elif isinstance(result, list):
        items = result
    
    # If still no items, try to extract from the raw result string
    if not items and hasattr(result, '__dict__'):
        result_dict = result.__dict__
        if 'items' in result_dict:
            items = result_dict['items']
    
    # Debug print to see what we got
    print(f"🔍 Processing result type: {type(result)}")
    print(f"🔍 Found {len(items) if items else 0} items")
    if items and len(items) > 0:
        print(f"🔍 First item type: {type(items[0])}")
        print(f"🔍 First item preview: {str(items[0])[:200]}...")
    
    # Convert items to CSV-compatible format
    csv_rows = []
    
    # Handle nested structure where papers are inside items
    papers_list = []
    for item in items:
        if isinstance(item, dict) and 'papers' in item:
            papers_list.extend(item['papers'])
        elif hasattr(item, 'papers'):
            papers_list.extend(item.papers)
        else:
            # If item doesn't have 'papers', treat it as a direct paper entry
            papers_list.append(item)
    
    # If no papers found in nested structure, use items directly
    if not papers_list:
        papers_list = items
    
    print(f"🔍 Found {len(papers_list)} papers to process")
    
    for paper in papers_list:
        row_data = {}
        for field in schema_fields:
            value = ""
            
            # Handle both dict and object access
            if isinstance(paper, dict):
                # Try different field name variations
                value = paper.get(field, "")
                if not value:
                    # Try variations like "Authors" vs "Author"
                    for key in paper.keys():
                        if key.lower() == field.lower():
                            value = paper[key]
                            break
            else:
                value = getattr(paper, field, "")
                if not value:
                    # Try variations
                    for attr in dir(paper):
                        if attr.lower() == field.lower():
                            value = getattr(paper, attr, "")
                            break
            
            # Ensure value is a string
            if isinstance(value, list):
                # Convert lists to appropriate string format
                if field.lower() in ['affiliations', 'affiliation']:
                    value = '; '.join(str(v) for v in value)
                else:
                    value = ', '.join(str(v) for v in value)
            elif value is None:
                value = ""
            else:
                value = str(value)
            
            row_data[field] = value
        
        # Only add rows that have at least one non-empty field
        if any(v.strip() for v in row_data.values()):
            csv_rows.append(row_data)
    
    return csv_rows

async def main():
    parser = argparse.ArgumentParser(description="Browser automation with dynamic schema and real-time CSV writing")
    parser.add_argument("--url", "-u", required=True, help="URL to scrape")
    parser.add_argument("--schema", "-s", required=True, 
                       help="List of field names as a Python list string, e.g., \"['Title','DOI','Authors','Affiliations']\"")
    parser.add_argument("--output", "-o", help="Output CSV filename (optional)")
    parser.add_argument("--model", "-m", default="gpt-4.1-mini", help="OpenAI model for prompt generation")
    parser.add_argument("--max-retries", "-r", type=int, default=2, help="Maximum number of retry attempts (default: 2)")
    
    args = parser.parse_args()
    
    # Parse inputs
    try:
        schema_fields = parse_schema_input(args.schema)
        url = args.url
    except Exception as e:
        print(f"❌ Input parsing error: {e}")
        sys.exit(1)
    
    # Setup OpenAI client
    api_key = load_secrets()
    client = OpenAI(api_key=api_key)
    
    # Generate output filename if not provided
    if args.output:
        csv_filename = clean_filename(args.output)
    else:
        # Create filename from URL domain and timestamp - make it strictly alphanumeric
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        # Clean domain to be strictly alphanumeric
        clean_domain = re.sub(r'[^a-zA-Z0-9]', '', domain)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Remove underscores
        csv_filename = f"{clean_domain}results{timestamp}.csv"
    
    print(f"🎯 Target URL: {url}")
    print(f"📋 Schema fields: {schema_fields}")
    print(f"📄 Output file: {csv_filename}")
    print(f"🔄 Max retries: {args.max_retries}")
    
    # Generate dynamic prompts
    print("\n🤖 Generating browser task prompt...")
    task_prompt = generate_browser_task_prompt(client, url, schema_fields, args.model)
    
    print("\n📚 Generating schema definition...")
    schema_definition = generate_schema_definition(client, url, schema_fields, args.model)
    
    # Display generated prompts
    print("\n" + "="*80)
    print("GENERATED BROWSER TASK PROMPT:")
    print("="*80)
    print(task_prompt)
    print("="*80)
    
    print("\n" + "="*80)
    print("GENERATED SCHEMA DEFINITION:")
    print("="*80)
    print(schema_definition)
    print("="*80)
    
    # Create dynamic Pydantic models
    ItemModel, ContainerModel = create_dynamic_pydantic_models(schema_fields)
    
    # Initialize thread-safe CSV writer
    csv_writer = ThreadSafeCSVWriter(csv_filename, schema_fields)
    csv_writer.write_header()
    
    # Retry logic for browser automation
    for attempt in range(args.max_retries):
        try:
            print(f"\n🚀 Starting browser automation (attempt {attempt + 1}/{args.max_retries})...")
            
            # Setup browser automation
            controller = Controller(output_model=ContainerModel)
            
            # File-based extraction to bypass tool limitations
            enhanced_prompt = f"""
            {task_prompt}
            
            ALTERNATIVE FILE-BASED EXTRACTION STRATEGY:
            
            APPROACH: BYPASS EXTRACTION TOOL LIMITATIONS
            The structured extraction tool appears to have limitations. Use file output to capture ALL data.
            
            EXECUTION STEPS:
            1. Navigate to the page and ensure full loading
            2. Scroll to the very TOP of the page 
            3. Perform ONE scroll to the very BOTTOM to trigger loading of ALL content
            4. Extract ALL paper entries from the COMPLETE page in small batches
            5. Write each batch to a temporary text file as you go
            6. Continue until the entire page is processed
            
            CRITICAL EXTRACTION REQUIREMENTS:
            - Extract ALL paper entries from the ENTIRE page, not just what's visible
            - Process the page systematically from top to bottom
            - For each paper entry, extract: {', '.join(schema_fields)}
            - Write results to a file named "extracted_papers.txt" in JSON format
            - Each line should be a JSON object with the required fields
            - Continue processing until ALL entries are captured and written
            
            BATCH PROCESSING STRATEGY:
            - Process papers in small groups (5-10 at a time)
            - Write each batch to the file immediately
            - Continue until the complete page is processed
            - The file should contain dozens or hundreds of entries when complete
            
            FILE FORMAT:
            Each line in extracted_papers.txt should be a JSON object like:
            {{"Title": "Paper Title", "DOI": "DOI URL", "Author": "Author Names", "Affiliation": "Institutions"}}
            
            VERIFICATION: The extracted_papers.txt file should reflect the COMPLETE dataset from the entire page.
            """
            
            agent = Agent(
                task=enhanced_prompt.replace('\n', ' ').strip(),
                llm=ChatOpenAI(model="gpt-4.1"),
                controller=controller,
            )
            
            print(f"📁 Working directory: {os.getcwd()}")
            print(f"📂 CSV filename: {csv_filename}")
            
            # Ensure working directory exists and is writable
            os.makedirs(os.path.dirname(os.path.abspath(csv_filename)), exist_ok=True)
            
            # Run the agent and capture the history
            history = await agent.run()
            
            # Get the final result from the agent
            result = history.final_result()

            if result:
                print("✅ Agent completed successfully!")
                
                # Process and write to CSV
                try:
                    # First check for file-based extraction
                    extracted_file = "extracted_papers.txt"
                    csv_rows = []
                    
                    if os.path.exists(extracted_file):
                        print(f"📄 Found file-based extraction: {extracted_file}")
                        try:
                            import json
                            with open(extracted_file, 'r', encoding='utf-8') as f:
                                for line_num, line in enumerate(f, 1):
                                    line = line.strip()
                                    if line:
                                        try:
                                            paper_data = json.loads(line)
                                            # Convert to CSV row format
                                            row = {}
                                            for field in schema_fields:
                                                value = paper_data.get(field, "")
                                                if isinstance(value, list):
                                                    value = ", ".join(str(v) for v in value)
                                                row[field] = str(value)
                                            csv_rows.append(row)
                                        except json.JSONDecodeError as e:
                                            print(f"⚠️ Line {line_num}: Invalid JSON - {e}")
                                        except Exception as e:
                                            print(f"⚠️ Line {line_num}: Error processing - {e}")
                            
                            if csv_rows:
                                print(f"📊 Extracted {len(csv_rows)} items from file")
                            else:
                                print(f"❌ File exists but no valid data extracted")
                                
                        except Exception as e:
                            print(f"❌ Error processing file-based extraction: {e}")
                    
                    # Fall back to structured extraction if no file-based data
                    if not csv_rows:
                        csv_rows = process_result_data(result, schema_fields)
                        
                        # If still no direct CSV rows, try to read from any files the agent created
                        if not csv_rows:
                            print("🔍 No direct CSV data found, checking for created files...")
                            
                            # Check for common file names the agent might create
                            potential_files = ['results.md', 'output.json', 'data.json', 'extracted_data.json']
                            for filename in potential_files:
                                if os.path.exists(filename):
                                    print(f"📄 Found file: {filename}")
                                    try:
                                        with open(filename, 'r', encoding='utf-8') as f:
                                            file_content = f.read()
                                        
                                        # Try to parse as JSON
                                        try:
                                            file_data = json.loads(file_content)
                                            csv_rows = process_result_data(file_data, schema_fields)
                                            if csv_rows:
                                                print(f"✅ Successfully parsed data from {filename}")
                                                break
                                        except json.JSONDecodeError:
                                            # Try to extract JSON from markdown content
                                            json_match = re.search(r'\[[\s\S]*\]', file_content)
                                            if json_match:
                                                json_str = json_match.group(0)
                                                file_data = json.loads(json_str)
                                                csv_rows = process_result_data(file_data, schema_fields)
                                                if csv_rows:
                                                    print(f"✅ Successfully extracted JSON from {filename}")
                                                    break
                                    except Exception as e:
                                        print(f"❌ Error reading {filename}: {e}")
                    
                    if csv_rows:
                        print(f"📊 Extracted {len(csv_rows)} items")
                        
                        # Write all rows to CSV
                        csv_writer.write_batch(csv_rows)
                        
                        print(f"\n✅ Data extraction completed!")
                        print(f"📄 CSV file: {csv_filename}")
                        print(f"📊 Total items extracted: {len(csv_rows)}")
                        
                        # Show a sample of the first few items for verification
                        if len(csv_rows) > 0:
                            print(f"\n📋 Sample extracted data (first 3 items):")
                            for i, row in enumerate(csv_rows[:3]):
                                print(f"  Item {i+1}: {list(row.values())[:2]}...")  # Show first 2 fields
                        
                        # Clean up any temporary files created by the agent
                        temp_files = ['results.md', 'output.json', 'data.json', 'extracted_data.json', 'extracted_papers.txt']
                        for temp_file in temp_files:
                            if os.path.exists(temp_file):
                                try:
                                    os.remove(temp_file)
                                    print(f"🧹 Cleaned up temporary file: {temp_file}")
                                except Exception as e:
                                    print(f"⚠️ Could not remove {temp_file}: {e}")
                        
                        # Success - break out of retry loop
                        break
                        
                    else:
                        print("⚠️ No items found in the result")
                        print(f"🔍 Result type: {type(result)}")
                        print(f"🔍 Result preview: {str(result)[:200]}...")
                        
                        if attempt < args.max_retries - 1:
                            print("🔄 Retrying extraction...")
                            time.sleep(2)
                            continue
                        
                except Exception as e:
                    print(f"❌ Error processing results: {e}")
                    print("💾 CSV processing failed")
                    import traceback
                    traceback.print_exc()
                    
                    if attempt < args.max_retries - 1:
                        print("🔄 Retrying due to processing error...")
                        time.sleep(2)
                        continue
            
            else:
                print("❌ No result returned from agent")
                if attempt < args.max_retries - 1:
                    print("🔄 Retrying due to no result...")
                    time.sleep(2)
                    continue
            
            # If we get here without breaking, the attempt failed
            if attempt == args.max_retries - 1:
                print(f"❌ All {args.max_retries} attempts failed")
                sys.exit(1)
        
        except Exception as e:
            print(f"❌ Browser automation failed (attempt {attempt + 1}): {e}")
            if attempt < args.max_retries - 1:
                print("🔄 Retrying due to exception...")
                time.sleep(2)
                continue
            else:
                print(f"❌ All {args.max_retries} attempts failed")
                sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())