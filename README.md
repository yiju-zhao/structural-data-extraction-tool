# Structural Data Extraction Tool

A comprehensive tool for extracting structured data from various file formats using marker and LLM services. This tool supports PDFs, images, PowerPoint presentations, Word documents, Excel files, HTML, and EPUB files.

## Features

- **Multi-format Support**: Process PDF, images (PNG, JPG, TIFF, etc.), PPT/PPTX, DOC/DOCX, XLS/XLSX, HTML, and EPUB files
- **Flexible LLM Integration**: Support for OpenAI, Claude, Gemini, Vertex AI, and Ollama services
- **Custom Extraction Schemas**: Define custom Pydantic schemas for specific extraction needs
- **Batch Processing**: Efficient processing of multiple files with progress tracking
- **Multiple Output Formats**: Save results as CSV or JSON
- **GPU Support**: Optional GPU acceleration for faster processing
- **Caching**: Built-in caching to reduce API costs and processing time

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd structural-data-extraction-tool
```

2. Install dependencies:
```bash
pip install -e .
```

**Note:** For non-PDF files (PPTX, DOCX, XLSX, HTML, EPUB), ensure you have the full marker installation which is included in our dependencies. The base marker installation only supports PDFs and images.

3. Set up your API keys (choose one or more):
```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Claude
export CLAUDE_API_KEY="your-claude-api-key"

# Gemini
export GEMINI_API_KEY="your-gemini-api-key"
```

## Supported File Formats

- **PDFs**: Academic papers, reports, documents
- **Images**: PNG, JPG, JPEG, TIFF, BMP, GIF
- **Presentations**: PPT, PPTX
- **Documents**: DOC, DOCX
- **Spreadsheets**: XLS, XLSX
- **Web**: HTML, HTM
- **E-books**: EPUB

## Quick Start

### Basic Usage

```python
from structural_extractor import StructuralDataExtractor
import os

# Initialize extractor
extractor = StructuralDataExtractor(
    llm_config={"openai_api_key": os.getenv("OPENAI_API_KEY")}
)

# Extract from single file
result = extractor.extract_from_file("document.pdf")

# Extract from directory
results = extractor.extract_from_directory("input_folder")

# Save to CSV
extractor.save_to_csv(results, "output.csv")
```

### Command Line Usage

```bash
# Extract academic papers
python structural_extractor.py --input ./papers --schema academic --output results.csv

# Extract general content
python structural_extractor.py --input ./documents --schema general --output content.csv

# Show file statistics
python structural_extractor.py --input ./documents --stats
```

### Convenience Functions

```python
from structural_extractor import extract_academic_papers, extract_general_content

# Extract academic papers with automatic schema
csv_file = extract_academic_papers(
    input_dir="papers",
    output_csv="academic_results.csv",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Extract general content from various formats
csv_file = extract_general_content(
    input_dir="documents", 
    output_csv="general_results.csv",
    llm_service="marker.services.claude.ClaudeService",
    api_key=os.getenv("CLAUDE_API_KEY")
)
```

## Custom Extraction Schemas

Define custom schemas for specific extraction needs:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class PresentationMetadata(BaseModel):
    title: Optional[str] = Field(description="Presentation title")
    presenter: Optional[str] = Field(description="Presenter name")
    topics: List[str] = Field(default_factory=list, description="Main topics")
    slide_count: Optional[int] = Field(description="Number of slides")

# Use custom schema
extractor = StructuralDataExtractor(
    extraction_schema=PresentationMetadata,
    llm_config={"openai_api_key": os.getenv("OPENAI_API_KEY")}
)
```

## LLM Service Configuration

### OpenAI
```python
extractor = StructuralDataExtractor(
    llm_service="marker.services.openai.OpenAIService",
    llm_config={
        "openai_api_key": "your-key",
        "openai_model": "gpt-4",
        "openai_base_url": "https://api.openai.com/v1"  # optional
    }
)
```

### Claude
```python
extractor = StructuralDataExtractor(
    llm_service="marker.services.claude.ClaudeService",
    llm_config={
        "claude_api_key": "your-key",
        "claude_model_name": "claude-3-sonnet-20240229"
    }
)
```

### Gemini
```python
extractor = StructuralDataExtractor(
    llm_service="marker.services.gemini.GeminiService",
    llm_config={"gemini_api_key": "your-key"}
)
```

### Vertex AI
```python
extractor = StructuralDataExtractor(
    llm_service="marker.services.vertex.GoogleVertexService",
    llm_config={"vertex_project_id": "your-project-id"}
)
```

### Ollama (Local)
```python
extractor = StructuralDataExtractor(
    llm_service="marker.services.ollama.OllamaService",
    llm_config={
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama2"
    }
)
```

## Advanced Configuration

```python
extractor = StructuralDataExtractor(
    llm_service="marker.services.openai.OpenAIService",
    llm_config={"openai_api_key": "your-key"},
    extraction_schema=CustomSchema,
    output_dir="./results",
    use_gpu=True,           # Enable GPU acceleration
    max_pages=10,           # Limit pages processed
    cache_dir="./.cache"    # Cache directory
)
```

## Output Formats

### CSV Output
The tool flattens nested data structures for CSV compatibility:
- Lists are converted to semicolon-separated strings
- Dictionaries are converted to JSON strings
- Metadata includes source file information

### JSON Output
Preserves full data structure with nested objects and arrays.

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic usage patterns
- Custom schema definitions
- Multiple LLM service configurations
- Batch processing workflows
- Financial document extraction
- Presentation metadata extraction

## Performance Optimization

1. **GPU Usage**: Enable GPU for faster processing of large documents
2. **Batch Size**: Adjust batch size based on memory constraints
3. **Page Limits**: Set `max_pages` for documents where metadata is in early pages
4. **Caching**: Results are automatically cached to avoid duplicate API calls

## Error Handling

The tool includes robust error handling:
- Unsupported files are skipped with warnings
- API errors are logged and processing continues
- Detailed logging for troubleshooting

## Command Line Options

```bash
python structural_extractor.py [OPTIONS]

Options:
  --input, -i          Input directory (required)
  --output, -o         Output CSV file
  --llm-service        LLM service to use
  --api-key            API key for LLM service
  --schema             Extraction schema (academic/general)
  --stats              Show file statistics only
```

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `CLAUDE_API_KEY`: Anthropic Claude API key
- `GEMINI_API_KEY`: Google Gemini API key

## Dependencies

- `marker-pdf`: Document processing and conversion
- `pydantic`: Schema definition and validation
- `openai`: OpenAI API client
- `tqdm`: Progress tracking
- `pandas`: Data manipulation (optional)

## License

[Add your license information here]

## Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests for any improvements.
