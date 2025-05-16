# PDF Metadata Extraction Tool

A command-line tool for extracting structured metadata from academic PDFs. This tool uses the Marker PDF library for PDF parsing and OpenAI API for intelligent metadata extraction.

## Features

- Batch process multiple PDF files
- Extract structured metadata including authors, affiliations, and keywords
- GPU acceleration support (optional)
- Multi-GPU parallel processing for high-throughput processing
- Caching mechanism to avoid duplicate API calls
- Progress tracking with tqdm
- Configurable processing parameters

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd metadata-extraction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
Create a `.env` file in the project root with your API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

```bash
python main.py --input /path/to/pdfs
```

### Advanced Options

```bash
python main.py --input /path/to/pdfs \
               --json-dir /path/to/json/cache \
               --output /path/to/output \
               --batch-size 4 \
               --use-gpu
```

### Multi-GPU Parallel Processing

For systems with multiple GPUs, you can leverage parallel processing:

```bash
python main.py --input /path/to/pdfs \
               --use-gpu \
               --num-gpus 4
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input`, `-i` | Path to folder containing PDF files | (Required) |
| `--json-dir`, `-j` | Path for intermediate JSON files | Same as input |
| `--output`, `-o` | Path for extracted metadata | Same as input |
| `--skip-pdf-parsing` | Skip PDF parsing and use existing JSON | False |
| `--batch-size` | Number of PDFs to process in each batch | 10 |
| `--max-pages` | Maximum number of pages to process per PDF | 1 |
| `--use-gpu` | Use GPU for processing if available | False |
| `--num-gpus` | Number of GPUs to use for parallel processing | 1 |

## Metadata Output

The tool extracts the following metadata from each PDF:

- Title
- Abstract text
- Authors
- Affiliations
- Keywords

Results are saved as JSON files with the naming format `{original_filename}_metadata.json`.

## Requirements

- Python 3.10+
- OpenAI API key
- Marker PDF library

## Processing a Single PDF

To process a single PDF file directly:

```bash
python metadata_extraction.py /path/to/paper.pdf
```

## Project Structure

- `main.py`: Main entry point and command-line interface
- `processor.py`: PDF processing and batch handling
- `metadata_extraction.py`: Text extraction and OpenAI API integration

## Performance Optimization

- PDF parsing model is loaded once per batch for improved efficiency
- Multi-GPU parallel processing for high throughput
- Results are cached to avoid duplicate API calls
- Processing is done in configurable batch sizes
