import sys
import os
import re
import fitz  # PyMuPDF
import argparse

def clean_title(title: str) -> str:
    """Clean the title by replacing non-alphanumeric characters with underscores."""
    cleaned = re.sub(r'[^\w\d]', '_', title)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned

def truncate_before_section(text: str) -> str:
    """
    Truncates the text to keep only content before a specified section.
    Searches first for "Abstract", then for "Introduction" (case-insensitively).
    """
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

def extract_first_page_text_to_md(pdf_path: str, should_truncate: bool):
    """
    Extracts text from the first page of a PDF and saves it to a markdown file.
    Optionally truncates the text before the 'Abstract' or 'Introduction' section.

    Args:
        pdf_path (str): The path to the input PDF file.
        should_truncate (bool): If True, apply the truncation logic.
    """
    try:
        # Generate output path
        base_name = os.path.basename(pdf_path)
        title_without_ext = os.path.splitext(base_name)[0]
        cleaned_filename = clean_title(title_without_ext)
        md_path = f"{cleaned_filename}.md"

        # Open PDF and extract text from the first page
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        text = page.get_text()
        doc.close()

        # Process text based on the truncate flag
        if should_truncate:
            processed_text = truncate_before_section(text)
            action_msg = "extracted and truncated"
        else:
            processed_text = text
            action_msg = "extracted"

        # Write the result to the markdown file
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(processed_text)

        print(f"Successfully {action_msg} the first page of '{pdf_path}' to '{md_path}'")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract text from the first page of a PDF to a Markdown file."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="The path to the input PDF file."
    )
    parser.add_argument(
        "--truncate_section",
        default=True,
        help="If set, remove the 'Abstract' or 'Introduction' section and all subsequent content."
    )
    
    # Parse arguments and run the main function
    args = parser.parse_args()
    extract_first_page_text_to_md(args.pdf_path, args.truncate_section)