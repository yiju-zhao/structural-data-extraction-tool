import json
import re
import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def clean_html(html_content):
    """Helper function to clean HTML content"""
    text = re.sub(r'<[^>]+>', ' ', html_content).strip()
    text = re.sub(r'<a href="[^"]*">[^<]*</a>', '', text)
    return re.sub(r'\s+', ' ', text)

def extract_metadata_from_json(json_file_path):
    """
    Extract metadata from the JSON file with a simplified approach.
    Extracts content before abstract, abstract text, and footnotes.
    """
    # Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Initialize collections
    content_before_abstract = []
    footnotes = []
    abstract = ""
    title = ""
    
    # Process the document
    found_abstract_header = False
    got_abstract = False
    got_title = False
    
    # Process all children in one pass
    for child in data["children"][0]["children"]:
        block_type = child.get("block_type")
        
        # Process footnotes regardless of position
        if block_type == "Footnote":
            footnotes.append(clean_html(child.get("html", "")))
            continue

        # Handle title detection
        if not got_title and block_type == "SectionHeader":
            clean_text = clean_html(child.get("html", ""))
            got_title = True
            title = clean_html(child.get("html", ""))
            continue
        
        # Handle abstract header detection
        if not found_abstract_header and block_type == "SectionHeader":
            clean_text = clean_html(child.get("html", ""))
            if "ABSTRACT" in clean_text.upper():
                found_abstract_header = True
                continue
        
        # Collect content before abstract
        if not found_abstract_header and block_type in ["SectionHeader", "Text"]:
            content_before_abstract.append(clean_html(child.get("html", "")))
        
        # Extract abstract text (first Text block after abstract header)
        elif found_abstract_header and block_type == "Text" and not got_abstract:
            abstract = clean_html(child.get("html", ""))
            got_abstract = True
    
    # Return the collected sections for LLM to analyze
    return {
        "title": title,
        "content_before_abstract": content_before_abstract,
        "footnotes": footnotes,
        "abstract": abstract
    }

def get_structured_metadata(extracted_data):
    # OpenAI client setup with API key from .env
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Prepare data for the LLM
    content_text = "\n\n".join(extracted_data['content_before_abstract'])
    footnotes_text = "\n\n".join(extracted_data['footnotes'])
    abstract_text = extracted_data['abstract']
    
    # Create optimized prompt for OpenAI
    prompt = f"""
    # Research Paper Metadata Extraction Task
    
    Analyze the following research paper content and extract the specified metadata elements into a structured JSON format.
    
    ## Paper Content
    
    ### Content Before Abstract:
    ```
    {content_text}
    ```
    
    ### Footnotes:
    ```
    {footnotes_text}
    ```
    
    ### Abstract:
    ```
    {abstract_text}
    ```
    
    ## Extraction Requirements
    
    Return a JSON object with the following fields:
    
    1. `authors`: An array of objects, each with:
       - `name`: Author's full name (string)
       - `affiliations`: Array of affiliation markers (numbers or letters) associated with this author
    
    2. `affiliations`: An array of objects, each with:
       - `id`: The affiliation marker (number or letter)
       - `name`: Name of the institution/organization
    
    3. `keywords`: An array of 5-8 important keywords or topics from the abstract
    
    ## Important Notes
    - Maintain the exact spelling and formatting of names
    - If an author has multiple affiliations, include all markers with a comma separator
    - Extract only information explicitly stated in the text
    - For affiliations, only include the highest level of the institution/organization no department information no geographic information, e.g. "University of Chicago"
    """
    
    # Make a single LLM call to extract all metadata
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are an expert at extracting structured metadata from research papers. Your task is to extract specific fields and return them in a clean, consistent JSON format."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
    )
    
    extracted = completion.choices[0].message.content
    
    # Add the extracted abstract to the result
    result = json.loads(extracted)
    result["title"] = extracted_data["title"]
    result["abstract"] = abstract_text
    
    return result

def main():
    # Extract raw metadata from JSON
    extracted_data = extract_metadata_from_json("AdaParse.json")
    
    # Get structured metadata using OpenAI
    structured_metadata = get_structured_metadata(extracted_data)
    
    # Output results
    print(json.dumps(structured_metadata, indent=2))
    
    # Optionally save to file
    with open("paper_metadata.json", "w") as f:
        json.dump(structured_metadata, indent=2, fp=f)

if __name__ == "__main__":
    main()
