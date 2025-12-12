import re
import os

def clean_transcript(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    cleaned_text = []
    # Regex for VTT timestamp line: 00:00:00.000 --> 00:00:00.000
    # Also handles format: 002357.755 -- 002359.794
    timestamp_pattern = re.compile(r'^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}|^\d+\.\d+\s+--\s+\d+\.\d+')

    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Skip WEBVTT header
        if line == 'WEBVTT':
            continue
            
        # Skip timestamp lines
        if timestamp_pattern.match(line):
            continue
            
        # Skip lines that are just numbers (sometimes happen in srt, less likely in simple vtt but good safety)
        if line.isdigit():
            continue

        cleaned_text.append(line)

    # Merge text
    # Join with newlines. 
    full_text = '\n'.join(cleaned_text)
    
    # Write to output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Successfully cleaned transcript. Output saved to: {output_path}")

if __name__ == "__main__":
    input_file = "/Users/eason/Documents/HW Project/Agent/Tools/structural-data-extraction-tool/neurips2025/VLA workshop字幕.txt"
    output_file = "VLA_workshop_字幕_cleaned.txt"
    
    print(f"Processing {input_file}...")
    clean_transcript(input_file, output_file)
