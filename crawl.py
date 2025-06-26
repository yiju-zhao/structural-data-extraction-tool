import os
import asyncio
import json
import csv
import argparse
import sys
from datetime import datetime
from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Any
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.chunking_strategy import ChunkingStrategy
from dotenv import load_dotenv
import openai
from nltk.tokenize import TextTilingTokenizer

load_dotenv()

def create_dynamic_model(schema_fields: List[str], model_name: str = "DynamicModel") -> BaseModel:
    """Create a dynamic Pydantic model based on user-provided schema fields"""
    field_definitions = {}
    for field in schema_fields:
        # All fields are strings for simplicity, could be enhanced to support different types
        field_definitions[field] = (str, Field(description=f"The {field} field"))
    
    return create_model(model_name, **field_definitions)

class TopicSegmentationChunking(ChunkingStrategy):
    def __init__(self, w=20, k=10, similarity_method='cosine'):
        """
        Initialize the Topic Segmentation Chunking strategy using TextTiling.
        
        Args:
            w (int): Window size for text tiling (default: 20)
            k (int): Size of the window for computing similarity (default: 10) 
            similarity_method (str): Method for computing similarity (default: 'cosine')
        """
        super().__init__()
        try:
            import nltk
            # Download required NLTK data if not already present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            self.tokenizer = TextTilingTokenizer(w=w, k=k, similarity_method=similarity_method)
        except ImportError:
            raise ImportError("NLTK is required for TopicSegmentationChunking. Install it with: pip install nltk")

    def chunk(self, text: str) -> list:
        """
        Chunk text using TextTiling algorithm based on topic segmentation.
        
        Args:
            text (str): Input text to be chunked
            
        Returns:
            list: List of text chunks based on topic boundaries
        """
        if not text or not text.strip():
            return []
            
        try:
            # Use TextTiling to segment text into topic-based chunks
            segments = self.tokenizer.tokenize(text)
            
            # Filter out empty segments and strip whitespace
            chunks = [segment.strip() for segment in segments if segment.strip()]
            
            # If no segments found, return the original text as a single chunk
            if not chunks:
                chunks = [text.strip()]
                
            return chunks
            
        except Exception as e:
            print(f"Error in topic segmentation chunking: {e}")
            # Fallback to simple paragraph splitting if TextTiling fails
            chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
            return chunks if chunks else [text.strip()]


async def generate_extraction_instruction(schema_fields: List[str]) -> str:
    """Generate detailed extraction instruction using LLM based on schema fields"""
    
    client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""
You are an expert at creating detailed extraction instructions for web content parsing.

Given the following schema fields: {', '.join(schema_fields)}

Create a comprehensive and detailed extraction instruction that will help an LLM extract data from web content accurately. The instruction should:

1. Clearly specify what information to extract for each field and definition of the field.
2. Handle cases where fields might be missing (mark as 'N/A'), do not omit any fields.
3. Explicitly state that you will extract ALL records from the content, not just the first few ones.
4. Be specific about the expected field format and structure.
5. Include the instuctions to handle the case where fileds containing multiple values (such as lists), MUST NOT split them into multiple records or multiple rows, the fields of the same entity should be stored in a single record.
6. Include the instuctions to ask the llm must first understand the structure of the content, where the expected fileds usually located, and then extract the data according to the structure.
7. Do not output examples.

Return only the instruction text, no additional formatting or explanation.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an expert at creating detailed extraction instructions for web content parsing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        
        instruction = response.choices[0].message.content.strip()
        print(f"Generated instruction: {instruction}")
        return instruction
        
    except Exception as e:
        print(f"Error generating instruction: {e}")
        # Fallback to a generic instruction
        field_list = "', '".join(schema_fields)
        return f"Extract all records with '{field_list}' from the content. If a field is missing, mark it as 'N/A'. This section may contain MULTIPLE records. Please extract ALL records from the section, not just the first few ones."

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Web content extraction with dynamic schema')
    parser.add_argument('url', help='URL to crawl and extract data from')
    parser.add_argument('--schema', nargs='+', required=True, 
                       help='Schema fields to extract (e.g., --schema title doi author affiliation)')
    parser.add_argument('--output', default=None, 
                       help='Output CSV filename (default: auto-generated based on timestamp)')
    
    args = parser.parse_args()
    
    url = args.url
    schema_fields = args.schema
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = url.split('/')[2].replace('.', '_')
        output_file = f"{domain}_extraction_{timestamp}.csv"
    
    print(f"URL to crawl: {url}")
    print(f"Schema fields: {schema_fields}")
    print(f"Output file: {output_file}")
    
    # Create dynamic Pydantic model
    DynamicModel = create_dynamic_model(schema_fields, "ExtractedData")
    
    # Generate detailed extraction instruction using LLM
    print("Generating extraction instruction...")
    extraction_instruction = await generate_extraction_instruction(schema_fields)
    
    # Create custom topic segmentation chunking strategy
    topic_chunker = TopicSegmentationChunking(w=20, k=10, similarity_method='cosine')
    
    # 1. Define the LLM extraction strategy with custom chunking
    llm_strategy = LLMExtractionStrategy(
        llm_config = LLMConfig(provider="openai/gpt-4.1-mini", api_token=os.getenv('OPENAI_API_KEY')),
        schema=DynamicModel.model_json_schema(),
        extraction_type="schema",
        instruction=extraction_instruction,
        chunking_strategy=topic_chunker,  # Use custom chunking strategy
        apply_chunking=True,              # Enable chunking
        input_format="markdown",          # or "html", "fit_markdown"
        extra_args={"temperature": 0.0}
    )

    # 2. Build the crawler config
    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS
    )

    # 3. Create a browser config if needed
    browser_cfg = BrowserConfig(headless=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        # 4. Crawl the provided URL
        print(f"Crawling {url}...")
        result = await crawler.arun(
            url=url,
            config=crawl_config
        )

        if result.success:
            # 5. The extracted content is presumably JSON
            data = json.loads(result.extracted_content)
            print("Extracted items:", len(data) if isinstance(data, list) else "Not a list")

            # 6. Save results to CSV file
            if data and isinstance(data, list) and len(data) > 0:
                # Get field names from the dynamic model
                fieldnames = list(DynamicModel.model_fields.keys())
                
                # Post-processing: filter out fields not in the model
                filtered_data = []
                for item in data:
                    if isinstance(item, dict):
                        filtered_item = {key: value for key, value in item.items() if key in fieldnames}
                        # Ensure all required fields are present
                        for field in fieldnames:
                            if field not in filtered_item:
                                filtered_item[field] = 'N/A'
                        filtered_data.append(filtered_item)
                
                # Post-processing: remove entirely empty rows
                def is_empty_row(row):
                    """Check if a row is entirely empty (all values are None, empty string, or 'N/A')"""
                    for value in row.values():
                        if value and str(value).strip() and str(value).strip().upper() != 'N/A':
                            return False
                    return True
                
                non_empty_data = [row for row in filtered_data if not is_empty_row(row)]
                
                # Post-processing: remove duplicate rows
                seen = set()
                unique_data = []
                for row in non_empty_data:
                    # Create a tuple of sorted items for comparison, handling unhashable types
                    def make_hashable(value):
                        """Convert unhashable types to hashable ones"""
                        if isinstance(value, (list, dict)):
                            return str(value)  # Convert to string representation
                        return value
                    
                    row_tuple = tuple(sorted((k, make_hashable(v)) for k, v in row.items()))
                    if row_tuple not in seen:
                        seen.add(row_tuple)
                        unique_data.append(row)
                
                print(f"Original records: {len(data)}")
                print(f"After filtering empty rows: {len(non_empty_data)}")
                print(f"After removing duplicates: {len(unique_data)}")
                
                # Write to CSV
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Write header
                    writer.writeheader()
                    
                    # Write processed data rows
                    for item in unique_data:
                        writer.writerow(item)
                
                print(f"Results saved to {output_file}")
                print(f"Total papers extracted: {len(unique_data)}")
            else:
                print("No data to save or data format is incorrect")
                with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(DynamicModel.model_fields.keys()))
                    writer.writeheader()
                print(f"Empty CSV file created: {output_file}")

            # 7. Show usage stats
            llm_strategy.show_usage()  # prints token usage
        else:
            print("Error:", result.error_message)
            return 1

    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
