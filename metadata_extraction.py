import json
import re
import os
import logging
from pathlib import Path
import hashlib
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class MetadataExtractor:
    """Class for extracting metadata from document JSON files"""

    def __init__(self, cache_dir: Union[str, Path] = ".cache"):
        """
        Initialize the metadata extractor

        Args:
            cache_dir: Directory to store cached results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Check for API key early
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY environment variable is not set")

    @staticmethod
    def clean_html(html_content: Optional[str]) -> str:
        """
        Helper function to clean HTML content

        Args:
            html_content: HTML content to clean

        Returns:
            Cleaned text
        """
        if not html_content:
            return ""
        text = re.sub(r"<[^>]+>", " ", html_content).strip()
        text = re.sub(r'<a href="[^"]*">[^<]*</a>', "", text)
        return re.sub(r"\s+", " ", text)

    def extract_from_json(self, json_file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from the JSON file with a simplified approach.
        Extracts content before abstract, abstract text, and footnotes.

        Args:
            json_file_path: Path to the JSON file

        Returns:
            Dictionary containing extracted metadata
        """
        try:
            # Load JSON data
            with open(json_file_path, "r") as file:
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

            # Check if data has expected structure
            if (
                not data.get("children")
                or not data["children"]
                or not data["children"][0].get("children")
            ):
                logger.warning(f"Invalid JSON structure in {json_file_path}")
                return {
                    "title": title,
                    "content_before_abstract": content_before_abstract,
                    "footnotes": footnotes,
                    "abstract": abstract,
                }

            # Process all children in one pass
            for child in data["children"][0]["children"]:
                block_type = child.get("block_type")

                # Process footnotes regardless of position
                if block_type == "Footnote":
                    footnotes.append(self.clean_html(child.get("html", "")))
                    continue

                # Handle title detection
                if not got_title and block_type == "SectionHeader":
                    got_title = True
                    title = self.clean_html(child.get("html", ""))
                    continue

                # Handle abstract header detection
                if not found_abstract_header and block_type == "SectionHeader":
                    clean_text = self.clean_html(child.get("html", ""))
                    if "ABSTRACT" in clean_text.upper():
                        found_abstract_header = True
                        continue

                # Collect content before abstract
                if not found_abstract_header and block_type in [
                    "SectionHeader",
                    "Text",
                ]:
                    content_before_abstract.append(
                        self.clean_html(child.get("html", ""))
                    )

                # Extract abstract text (first Text block after abstract header)
                elif (
                    found_abstract_header and block_type == "Text" and not got_abstract
                ):
                    abstract = self.clean_html(child.get("html", ""))
                    got_abstract = True

            # Return the collected sections for LLM to analyze
            return {
                "title": title,
                "content_before_abstract": content_before_abstract,
                "footnotes": footnotes,
                "abstract": abstract,
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {json_file_path}: {str(e)}")
            return {
                "title": "",
                "content_before_abstract": [],
                "footnotes": [],
                "abstract": "",
            }

    def _get_cache_key(self, extracted_data: Dict[str, Any]) -> str:
        """
        Generate a cache key from extracted data

        Args:
            extracted_data: The extracted metadata to hash

        Returns:
            MD5 hash to use as cache key
        """
        # Create a string representation of the extracted data
        data_str = json.dumps(extracted_data, sort_keys=True)
        # Generate a hash
        return hashlib.md5(data_str.encode()).hexdigest()

    def get_structured_metadata(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get structured metadata from extracted data using OpenAI.
        Uses caching to avoid duplicate LLM calls.

        Args:
            extracted_data: The extracted metadata to structure

        Returns:
            Dictionary of structured metadata
        """
        try:
            # Check if we have a valid API key
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")

            # Check for minimal data validity
            if not extracted_data.get("title") and not extracted_data.get("abstract"):
                logger.warning("Not enough data for metadata extraction")
                return {
                    "title": extracted_data.get("title", ""),
                    "abstract": extracted_data.get("abstract", ""),
                    "authors": [],
                    "affiliations": [],
                    "keywords": [],
                }

            # Generate cache key and check cache
            cache_key = self._get_cache_key(extracted_data)
            cache_file = self.cache_dir / f"{cache_key}.json"

            # If cache exists, return cached result
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    logger.info("Using cached metadata result")
                    return json.load(f)

            # OpenAI client setup with API key from .env
            client = OpenAI(api_key=self.api_key)

            # Prepare data for the LLM
            content_text = "\n\n".join(
                extracted_data.get("content_before_abstract", [])
            )
            footnotes_text = "\n\n".join(extracted_data.get("footnotes", []))
            abstract_text = extracted_data.get("abstract", "")

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
            logger.info("Calling OpenAI API for metadata extraction")
            completion = client.beta.chat.completions.parse(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting structured metadata from research papers. Your task is to extract specific fields and return them in a clean, consistent JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            extracted = completion.choices[0].message.content

            # Add the extracted abstract and title to the result
            result = json.loads(extracted)
            result["title"] = extracted_data["title"]
            result["abstract"] = abstract_text

            # Save to cache
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2)

            logger.info("Successfully extracted structured metadata")
            return result
        except Exception as e:
            logger.error(f"Error getting structured metadata: {str(e)}")
            return {
                "title": extracted_data.get("title", ""),
                "abstract": extracted_data.get("abstract", ""),
                "authors": [],
                "affiliations": [],
                "keywords": [],
            }


def extract_metadata_from_json(json_file_path):
    """Legacy function for backward compatibility"""
    extractor = MetadataExtractor()
    return extractor.extract_from_json(json_file_path)


def get_structured_metadata(extracted_data):
    """Legacy function for backward compatibility"""
    extractor = MetadataExtractor()
    return extractor.get_structured_metadata(extracted_data)


def main():
    """Test the metadata extraction functionality"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python metadata_extraction.py <json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    print(f"Processing {json_file}...")

    # Extract raw metadata from JSON
    extractor = MetadataExtractor()
    extracted_data = extractor.extract_from_json(json_file)

    # Get structured metadata using OpenAI
    structured_metadata = extractor.get_structured_metadata(extracted_data)

    # Output results
    print(json.dumps(structured_metadata, indent=2))

    # Save to file
    output_file = Path(json_file).stem + "_metadata.json"
    with open(output_file, "w") as f:
        json.dump(structured_metadata, indent=2, fp=f)

    print(f"Saved metadata to {output_file}")


if __name__ == "__main__":
    main()
