#!/bin/bash

# Simple Web Content Extraction Script
# Usage: ./run_crawl.sh [URL] [SCHEMA_FIELDS] [PRIMARY_KEYS]

echo "🚀 Web Content Extraction Tool"
echo "=============================="

# Default settings
DEFAULT_URL="https://kdd2025.kdd.org/research-track-papers-2"
DEFAULT_SCHEMA="title doi author affiliation"
DEFAULT_PRIMARY_KEYS="title doi"

# Use provided arguments or defaults
URL=${1:-$DEFAULT_URL}
SCHEMA=${2:-$DEFAULT_SCHEMA}
PRIMARY_KEYS=${3:-$DEFAULT_PRIMARY_KEYS}

echo "📍 URL: $URL"
echo "📋 Schema: $SCHEMA"
echo "🗝️  Primary Keys: $PRIMARY_KEYS"
echo ""

# Run extraction
echo "🔄 Starting extraction..."
python crawl.py "$URL" --schema $SCHEMA --primary-keys $PRIMARY_KEYS --strategy two-pass --verbose

echo ""
echo "✅ Done! Check the output CSV file in the ./output directory."
echo ""
echo "💡 Usage examples:"
echo "   ./run_crawl.sh"
echo "   ./run_crawl.sh \"https://example.com\" \"title author date\""
echo "   ./run_crawl.sh \"https://news-site.com\" \"headline summary author\" \"headline\""
echo ""
echo "🗝️  Primary keys determine which fields identify duplicates."
echo "   If not specified, the first field + common identifiers (title, doi, etc.) are used."
