import pandas as pd
import argparse
import os

def deduplicate_csv(input_file, output_file=None, column_name=None):
    """
    Deduplicate CSV rows based on specified column name.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str, optional): Path to save deduplicated CSV. 
                                   If None, generates filename as input_name_dedup_columnname.csv
        column_name (str, optional): Column name to use for deduplication.
                                   If None, uses the first column.
    """
    try:
        df = pd.read_csv(input_file)
        
        # Use first column if no column name specified
        if column_name is None:
            column_name = df.columns[0]
            print(f"No column specified, using first column: '{column_name}'")
        
        # Generate output filename if not provided
        if output_file is None:
            input_dir = os.path.dirname(input_file) or '.'
            input_basename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(input_dir, f"{input_basename}_dedup_{column_name}.csv")
            print(f"Generated output filename: {output_file}")
        
        # Perform deduplication
        original_count = len(df)
        df.drop_duplicates(subset=[column_name], keep='first', inplace=True)
        deduplicated_count = len(df)
        
        df.to_csv(output_file, index=False)
        print(f"Successfully deduplicated {input_file} based on column '{column_name}'")
        print(f"Removed {original_count - deduplicated_count} duplicates ({original_count} → {deduplicated_count} rows)")
        print(f"Saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except KeyError:
        print(f"Error: Column '{column_name}' not found in CSV. Available columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deduplicate CSV rows based on specified column')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', nargs='?', default=None, 
                       help='Output CSV file path (optional, auto-generated if not provided)')
    parser.add_argument('column_name', nargs='?', default=None,
                       help='Column name to use for deduplication (optional, uses first column if not provided)')
    
    args = parser.parse_args()
    deduplicate_csv(args.input_file, args.output_file, args.column_name)