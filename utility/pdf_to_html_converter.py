#!/usr/bin/env python3
"""
PDF to HTML Converter Utility

This utility uses the marker library to convert PDF files to HTML format,
making them suitable for web content extraction workflows.

Author: Assistant
License: MIT
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urljoin

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.config.parser import ConfigParser
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False


class PDFToHTMLConverter:
    """Convert PDF files to HTML using the marker library"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the PDF to HTML converter
        
        Args:
            config: Optional configuration dict for marker
        """
        if not MARKER_AVAILABLE:
            raise ImportError(
                "Marker library is not available. Install it with: pip install marker-pdf"
            )
        
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize marker converter
        self._setup_converter()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the converter"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_converter(self) -> None:
        """Setup the marker PDF converter"""
        try:
            # Default configuration for HTML output
            default_config = {
                "output_format": "html",
                "extract_images": False,  # Ignore images, only keep text
                "force_ocr": False,
                "use_llm": False
            }
            
            # Merge with user config
            merged_config = {**default_config, **self.config}
            
            if merged_config.get("use_llm", False):
                # Check if API key is available for LLM usage
                if not os.getenv('OPENAI_API_KEY') and not os.getenv('GOOGLE_API_KEY'):
                    self.logger.warning("No API key found for LLM usage. Disabling LLM.")
                    merged_config["use_llm"] = False
            
            # Create config parser
            config_parser = ConfigParser(merged_config)
            
            # Initialize converter
            self.converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service() if merged_config.get("use_llm") else None
            )
            
            self.logger.info("✅ Marker PDF converter initialized successfully")
            
        except Exception as e:
            # Fallback to basic converter
            self.logger.warning(f"Advanced configuration failed: {e}")
            self.logger.info("🔄 Falling back to basic converter...")
            
            self.converter = PdfConverter(
                artifact_dict=create_model_dict(),
            )
    
    def convert_pdf_to_html(self, pdf_path: str, output_dir: Optional[str] = None) -> Tuple[str, str]:
        """
        Convert PDF file to HTML format
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional output directory. If None, uses temp directory
            
        Returns:
            Tuple of (html_content, html_file_path)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        self.logger.info(f"🔄 Converting PDF to HTML: {pdf_path.name}")
        
        try:
            # Convert PDF using marker
            rendered = self.converter(str(pdf_path))
            
            # Extract only text from rendered output (ignore images)
            text, _, images = text_from_rendered(rendered)
            
            # Create HTML content (without images)
            html_content = self._create_html_content(text, None, pdf_path.stem)
            
            # Save HTML file
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path = Path(tempfile.gettempdir())
            
            html_file_path = output_path / f"{pdf_path.stem}_converted.html"
            
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"✅ PDF converted successfully: {html_file_path}")
            
            return html_content, str(html_file_path)
            
        except Exception as e:
            self.logger.error(f"❌ Error converting PDF: {e}")
            raise
    
    def _create_html_content(self, markdown_text: str, images: dict, title: str) -> str:
        """
        Create HTML content from markdown text (text only, no images)
        
        Args:
            markdown_text: Markdown text from marker
            images: Dictionary of images from marker (ignored)
            title: Title for the HTML document
            
        Returns:
            Complete HTML content
        """
        try:
            # Try to use markdown library for better HTML conversion
            import markdown
            from markdown.extensions import tables, codehilite
            
            # Configure markdown with extensions
            md = markdown.Markdown(
                extensions=['tables', 'codehilite', 'fenced_code', 'toc'],
                extension_configs={
                    'codehilite': {
                        'use_pygments': False,
                        'noclasses': True
                    }
                }
            )
            
            # Convert markdown to HTML
            html_body = md.convert(markdown_text)
            
        except ImportError:
            self.logger.warning("Markdown library not available. Using basic conversion.")
            # Basic markdown to HTML conversion
            html_body = self._basic_markdown_to_html(markdown_text)
        
        # Skip image embedding - text only conversion
        
        # Create complete HTML document
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        .pdf-converted {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
        }}
        blockquote {{
            border-left: 4px solid #007bff;
            margin: 20px 0;
            padding-left: 20px;
            color: #6c757d;
        }}
    </style>
</head>
<body>
            <div class="pdf-converted">
            <strong>📄 Converted from PDF using marker library (text only)</strong>
        </div>
    
    <h1>{title}</h1>
    
    {html_body}
</body>
</html>"""
        
        return html_content
    
    def _basic_markdown_to_html(self, markdown_text: str) -> str:
        """
        Basic markdown to HTML conversion without external dependencies
        
        Args:
            markdown_text: Markdown text to convert
            
        Returns:
            Basic HTML content
        """
        import re
        
        # Simple markdown to HTML conversion
        html = markdown_text
        
        # Convert headers
        html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
        
        # Convert bold and italic
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
        
        # Convert links
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
        
        # Convert line breaks
        html = html.replace('\n\n', '</p><p>')
        html = html.replace('\n', '<br>')
        
        # Wrap in paragraphs
        html = f'<p>{html}</p>'
        
        # Clean up empty paragraphs
        html = re.sub(r'<p></p>', '', html)
        html = re.sub(r'<p><br></p>', '', html)
        
        return html
    
    def _embed_images_html(self, images: dict) -> str:
        """
        Create HTML for embedded images
        
        Args:
            images: Dictionary of images from marker
            
        Returns:
            HTML content for images
        """
        if not images:
            return ""
        
        image_html = '<div class="images-section">\n<h2>📷 Extracted Images</h2>\n'
        
        for i, (image_name, image_data) in enumerate(images.items()):
            # For now, just create placeholders
            # In a full implementation, you'd save images and reference them
            image_html += f'''
            <div class="image-container">
                <p><strong>Image {i+1}:</strong> {image_name}</p>
                <p><em>Image data available but not displayed in this HTML conversion</em></p>
            </div>
            '''
        
        image_html += '</div>\n\n'
        return image_html
    
    def convert_and_serve(self, pdf_path: str, output_dir: Optional[str] = None) -> str:
        """
        Convert PDF to HTML and return a file:// URL that can be used with crawl4ai
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional output directory
            
        Returns:
            file:// URL pointing to the converted HTML file
        """
        html_content, html_file_path = self.convert_pdf_to_html(pdf_path, output_dir)
        
        # Convert to file:// URL
        html_file_path = Path(html_file_path).resolve()
        file_url = f"file://{html_file_path}"
        
        self.logger.info(f"🌐 HTML file available at: {file_url}")
        
        return file_url


def is_pdf_file(file_path: str) -> bool:
    """
    Check if a given file path is a PDF file
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file is a PDF, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.suffix.lower() == '.pdf'
    except Exception:
        return False


def convert_pdf_for_crawl4ai(pdf_path: str, output_dir: Optional[str] = None, config: Optional[dict] = None) -> str:
    """
    Convenience function to convert PDF for use with crawl4ai
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional output directory
        config: Optional configuration for marker
        
    Returns:
        file:// URL that can be used with crawl4ai
    """
    converter = PDFToHTMLConverter(config)
    return converter.convert_and_serve(pdf_path, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PDF to HTML using marker library")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", help="Output directory for HTML file")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for improved quality")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on the document")
    parser.add_argument("--extract-images", action="store_true", help="Extract images from PDF (default: text only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        "use_llm": args.use_llm,
        "force_ocr": args.force_ocr,
        "extract_images": args.extract_images
    }
    
    try:
        file_url = convert_pdf_for_crawl4ai(args.pdf_path, args.output_dir, config)
        print(f"✅ PDF converted successfully!")
        print(f"🌐 HTML URL: {file_url}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)