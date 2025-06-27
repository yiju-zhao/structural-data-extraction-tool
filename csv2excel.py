import pandas as pd
import os
from pathlib import Path
import argparse
import sys
import ast
import re

def process_list_strings(df):
    """
    Convert list-like strings in DataFrame to comma-separated values
    
    Args:
        df (pandas.DataFrame): DataFrame to process
        
    Returns:
        pandas.DataFrame: Processed DataFrame
    """
    df_copy = df.copy()
    
    for column in df_copy.columns:
        # Check if column contains list-like strings
        sample_values = df_copy[column].dropna().astype(str).head(10)
        
        if len(sample_values) > 0:
            # Check if values look like lists (start with [ and end with ])
            list_like_pattern = re.compile(r'^\s*\[.*\]\s*$')
            list_like_count = sum(1 for val in sample_values if list_like_pattern.match(str(val)))
            
            # If more than half of the sample values look like lists, process the column
            if list_like_count > len(sample_values) * 0.5:
                print(f"    Processing list-like column: {column}")
                
                def convert_list_string(value):
                    if pd.isna(value):
                        return value
                    
                    value_str = str(value).strip()
                    
                    # Skip if it doesn't look like a list
                    if not list_like_pattern.match(value_str):
                        return value
                    
                    try:
                        # Try to parse as a Python literal
                        parsed_list = ast.literal_eval(value_str)
                        
                        # If it's a list, join with commas
                        if isinstance(parsed_list, list):
                            # Convert all items to strings and join
                            return ', '.join(str(item).strip("'\"") for item in parsed_list)
                        else:
                            return value
                    except (ValueError, SyntaxError):
                        # If parsing fails, try simple regex approach
                        try:
                            # Remove brackets and split by comma
                            clean_str = value_str.strip('[]')
                            items = [item.strip().strip("'\"") for item in clean_str.split(',')]
                            return ', '.join(items)
                        except:
                            # If all else fails, return original value
                            return value
                
                df_copy[column] = df_copy[column].apply(convert_list_string)
    
    return df_copy

def convert_csvs_to_excel(input_dir):
    """
    Convert all CSV files in input_dir to a single Excel file with multiple sheets
    
    Args:
        input_dir (str): Directory containing CSV files
    """
    input_path = Path(input_dir)
    
    # Check if directory exists
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return False
    
    # Find all CSV files in the directory
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return False
    
    print(f"Found {len(csv_files)} CSV files to convert:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    # Create output Excel file path
    output_file = input_path / f"{input_path.name}_converted.xlsx"
    
    # Create ExcelWriter object
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            for csv_file in csv_files:
                try:
                    # Read CSV file with UTF-8 encoding and comma separator
                    df = pd.read_csv(csv_file, encoding='utf-8', sep=',')
                    
                    # Remove duplicates based on first column, keeping the last occurrence
                    if len(df.columns) > 0 and len(df) > 0:
                        first_column = df.columns[0]
                        original_count = len(df)
                        df = df.drop_duplicates(subset=[first_column], keep='last')
                        new_count = len(df)
                        
                        if original_count != new_count:
                            removed_count = original_count - new_count
                            print(f"    Removed {removed_count} duplicate rows based on column '{first_column}'")
                    
                    # Process list-like strings in the DataFrame
                    df = process_list_strings(df)
                    
                    # Use filename (without extension) as sheet name
                    sheet_name = csv_file.stem
                    
                    # Excel sheet names have limitations (max 31 characters), truncate if necessary
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31]
                        print(f"  Warning: Sheet name truncated to '{sheet_name}'")
                    
                    # Remove invalid characters for Excel sheet names
                    invalid_chars = ['[', ']', '*', '?', ':', '/', '\\']
                    for char in invalid_chars:
                        sheet_name = sheet_name.replace(char, '_')
                    
                    # Write to Excel sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    print(f"✓ Converted {csv_file.name} to sheet '{sheet_name}' ({len(df)} rows)")
                    
                except Exception as e:
                    print(f"✗ Error processing {csv_file.name}: {str(e)}")
        
        print(f"\n✓ Excel file saved as: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error creating Excel file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV files to Excel with multiple sheets',
        epilog='Example: python csv2excel.py ./output'
    )
    parser.add_argument('input_dir', help='Directory containing CSV files')
    
    args = parser.parse_args()
    
    success = convert_csvs_to_excel(args.input_dir)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
