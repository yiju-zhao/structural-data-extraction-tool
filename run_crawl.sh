#!/bin/bash

# Simple Web Content Extraction Script
# Usage: ./run_crawl.sh [URL] [SCHEMA_FIELDS]

echo "🚀 Web Content Extraction Tool"
echo "=============================="

# Default settings
DEFAULT_URL="https://kdd2025.kdd.org/research-track-papers-2"
DEFAULT_SCHEMA="title author doi affiliation"

# Use provided arguments or defaults
URL=${1:-$DEFAULT_URL}
SCHEMA=${2:-$DEFAULT_SCHEMA}

echo "📍 URL: $URL"
echo "📋 Schema: $SCHEMA"
echo ""

# Run extraction
echo "🔄 Starting extraction..."
python crawl.py "$URL" --schema $SCHEMA --strategy two-pass --verbose

echo ""
echo "✅ Done! Check the output CSV file."
echo ""
echo "💡 Usage examples:"
echo "   ./run_crawl.sh"
echo "   ./run_crawl.sh \"https://example.com\" \"title author date\""
echo "   ./run_crawl.sh \"https://news-site.com\" \"headline summary author\""