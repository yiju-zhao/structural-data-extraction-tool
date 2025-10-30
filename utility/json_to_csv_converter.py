import csv
import json
from typing import List, Dict, Any, Type
from pydantic import BaseModel
import pandas as pd


def json_to_csv_with_pydantic(
    json_data: str,
    pydantic_model: Type[BaseModel],
    csv_filename: str,
    remove_duplicates: bool = True,
    filter_empty: bool = True,
    preserve_all_data: bool = False,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Convert JSON data to CSV using Pydantic model schema as column structure.

    Args:
        json_data: Raw JSON string from browser-use output
        pydantic_model: Pydantic model class that defines the structure
        csv_filename: Output CSV filename
        remove_duplicates: Whether to remove duplicate rows
        filter_empty: Whether to filter out empty/N/A values
        preserve_all_data: If True, keeps all data with minimal filtering
        encoding: File encoding (default: utf-8)

    Returns:
        Dictionary with statistics about the conversion
    """
    try:
        # Parse the JSON data using the Pydantic model
        parsed_data = pydantic_model.model_validate_json(json_data)

        # Extract the list of items (assuming the model has a list field)
        items = []
        for field_name in pydantic_model.model_fields.keys():
            field_value = getattr(parsed_data, field_name)
            if isinstance(field_value, list):
                items = field_value
                break

        if not items:
            raise ValueError("No list field found in the Pydantic model")

        # Get field names from the first item's model
        if items:
            item_model = type(items[0])
            fieldnames = list(item_model.model_fields.keys())
        else:
            raise ValueError("No items found to extract field names")

        # Remove duplicates if requested
        unique_items = []
        seen_items = set()

        for item in items:
            # Create a unique identifier for each item
            item_values = tuple(getattr(item, field) for field in fieldnames)

            if remove_duplicates:
                if item_values not in seen_items:
                    seen_items.add(item_values)
                    unique_items.append(item)
            else:
                unique_items.append(item)

        # Filter empty/invalid items if requested
        valid_items = []
        if preserve_all_data:
            # When preserving all data, only remove completely empty rows
            valid_items = unique_items
            print(
                f"🔒 Preserve all data mode: keeping all {len(unique_items)} unique items"
            )
        elif filter_empty:
            for item in unique_items:
                # Less aggressive filtering - only filter if ALL fields are empty/N/A
                all_empty = True
                for field in fieldnames:
                    value = getattr(item, field)
                    if (
                        value
                        and str(value).strip()
                        and str(value).strip().upper() != "N/A"
                    ):
                        all_empty = False
                        break

                if not all_empty:  # Keep item if at least one field has valid data
                    valid_items.append(item)
        else:
            valid_items = unique_items

        # Write to CSV
        with open(csv_filename, "w", newline="", encoding=encoding) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # Write data rows
            for item in valid_items:
                row_data = {}
                for field in fieldnames:
                    row_data[field] = getattr(item, field)
                writer.writerow(row_data)

        # Calculate statistics
        stats = {
            "total_items": len(items),
            "unique_items": len(unique_items),
            "valid_items_written": len(valid_items),
            "duplicates_removed": len(items) - len(unique_items),
            "empty_filtered": len(unique_items) - len(valid_items),
            "csv_filename": csv_filename,
            "fieldnames": fieldnames,
        }

        return stats

    except Exception as e:
        raise Exception(f"Error converting JSON to CSV: {str(e)}")


def json_to_csv_fallback(
    raw_text: str,
    csv_filename: str,
    expected_columns: List[str],
    delimiter: str = "|",
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Fallback method to convert raw text output to CSV when JSON parsing fails.

    Args:
        raw_text: Raw text output from browser-use
        csv_filename: Output CSV filename
        expected_columns: List of expected column names
        delimiter: Delimiter used in the raw text (default: '|')
        encoding: File encoding (default: utf-8)

    Returns:
        Dictionary with statistics about the conversion
    """
    lines = raw_text.strip().split("\n")
    csv_data = []

    # Add header
    csv_data.append(expected_columns)

    # Process each line
    for line in lines:
        if line.strip() and delimiter in line:
            row = [cell.strip() for cell in line.split(delimiter)]
            row = [cell for cell in row if cell]  # Remove empty cells

            if row and len(row) >= len(expected_columns):
                # Take only the expected number of columns
                csv_data.append(row[: len(expected_columns)])

    # Remove duplicates
    unique_rows = []
    seen_rows = set()

    for row in csv_data[1:]:  # Skip header
        row_tuple = tuple(row)
        if row_tuple not in seen_rows:
            seen_rows.add(row_tuple)
            unique_rows.append(row)

    # Filter empty rows
    valid_rows = []
    for row in unique_rows:
        if all(cell and cell.strip() and cell.strip().upper() != "N/A" for cell in row):
            valid_rows.append(row)

    # Write to CSV
    final_data = [csv_data[0]] + valid_rows  # Header + valid rows

    with open(csv_filename, "w", newline="", encoding=encoding) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(final_data)

    stats = {
        "total_rows": len(csv_data) - 1,  # Exclude header
        "unique_rows": len(unique_rows),
        "valid_rows_written": len(valid_rows),
        "duplicates_removed": len(csv_data) - 1 - len(unique_rows),
        "empty_filtered": len(unique_rows) - len(valid_rows),
        "csv_filename": csv_filename,
        "fieldnames": expected_columns,
    }

    return stats


# Example usage function
def convert_browser_use_output_to_csv(
    raw_result: str,
    pydantic_model: Type[BaseModel],
    csv_filename: str,
    preserve_all_data: bool = False,
) -> Dict[str, Any]:
    """
    Main function to convert browser-use output to CSV.
    Tries JSON parsing first, falls back to text parsing if needed.

    Args:
        raw_result: Raw result string from browser-use
        pydantic_model: Pydantic model defining the data structure
        csv_filename: Output CSV filename
        preserve_all_data: If True, preserves all data with minimal filtering
    """
    try:
        # Try JSON parsing first
        stats = json_to_csv_with_pydantic(
            json_data=raw_result,
            pydantic_model=pydantic_model,
            csv_filename=csv_filename,
            preserve_all_data=preserve_all_data,
        )

        print(f"✅ Successfully converted JSON to CSV: {csv_filename}")
        print(f"📊 Statistics:")
        print(f"   Total items: {stats['total_items']}")
        print(f"   Unique items: {stats['unique_items']}")
        print(f"   Valid items written: {stats['valid_items_written']}")
        print(f"   Duplicates removed: {stats['duplicates_removed']}")
        print(f"   Empty/invalid filtered: {stats['empty_filtered']}")

        return stats

    except Exception as e:
        print(f"⚠️  JSON parsing failed: {e}")
        print("🔄 Attempting fallback text parsing...")

        # Extract field names from Pydantic model
        fieldnames = list(pydantic_model.model_fields.keys())
        if hasattr(pydantic_model, "__annotations__"):
            # If it's a container model, get the inner model's fields
            for field_name, field_info in pydantic_model.model_fields.items():
                if hasattr(field_info.annotation, "__args__"):
                    inner_type = field_info.annotation.__args__[0]
                    if hasattr(inner_type, "model_fields"):
                        fieldnames = list(inner_type.model_fields.keys())
                        break

        fallback_filename = csv_filename.replace(".csv", "_fallback.csv")

        stats = json_to_csv_fallback(
            raw_text=raw_result,
            csv_filename=fallback_filename,
            expected_columns=fieldnames,
        )

        print(f"✅ Fallback conversion completed: {fallback_filename}")
        print(f"📊 Statistics:")
        print(f"   Total rows: {stats['total_rows']}")
        print(f"   Unique rows: {stats['unique_rows']}")
        print(f"   Valid rows written: {stats['valid_rows_written']}")
        print(f"   Duplicates removed: {stats['duplicates_removed']}")
        print(f"   Empty/invalid filtered: {stats['empty_filtered']}")

        return stats
