
import csv
import sys

def remove_newlines_from_csv(input_file_path, output_file_path):
    """
    Reads a CSV file, removes newline characters from each cell,
    and writes the result to a new CSV file.
    """
    try:
        with open(input_file_path, 'r', newline='', encoding='utf-8') as infile, \
             open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row in reader:
                # Replace newlines in each cell of the row
                cleaned_row = [cell.replace('\n', ' ').replace('\r', ' ') for cell in row]
                writer.writerow(cleaned_row)
        print(f"Successfully removed newlines and saved to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_csv.py <input_csv_path> <output_csv_path>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    remove_newlines_from_csv(input_path, output_path)
