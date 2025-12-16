import csv
import json
import re
from datetime import datetime

INPUT_FILE = '/Users/eason/Documents/HW Project/Agent/Tools/structural-data-extraction-tool/neurips2025/neurips_2025_sessions_MexicoCity_detail.csv'
OUTPUT_FILE = 'neurips_2025_sessions_mexicoCity.json'
YEAR = 2025

def parse_date(date_str):
    # Format: "Sunday, Nov 30, 2025" or "TUE 2 DEC"
    if not date_str:
        return None
    try:
        # Try explicit format with year first: "Sunday, Nov 30, 2025"
        # %A: Weekday, %b: Abbr Month, %d: Day, %Y: Year
        dt = datetime.strptime(date_str, "%A, %b %d, %Y")
        return dt.date()
    except ValueError:
        pass

    try:
        # Fallback to old logic: "TUE 2 DEC"
        # We ignore the day of week and parse "2 DEC"
        parts = date_str.split()
        if len(parts) >= 3:
            day_month_str = f"{parts[1]} {parts[2]}"
            # Parse "2 DEC"
            dt = datetime.strptime(day_month_str, "%d %b")
            return dt.replace(year=YEAR).date()
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
    return None

def parse_time(time_str):
    # Format: "8:30 a.m.", "9:30 AM", "16:00", "noon", "4 p.m.", "10:00-11:00"
    if not time_str:
        return None, None

    normalized = time_str.lower().replace('.', '').strip()
    
    # Handle "noon"
    if 'noon' in normalized:
        return datetime.strptime("12:00 pm", "%I:%M %p").time(), None

    # Handle ranges like "10:00-11:00" or "3:30-4:30"
    # Check for '-' or '–'
    if '-' in normalized:
        parts = normalized.split('-')
    elif '–' in normalized: # en-dash
        parts = normalized.split('–')
    else:
        parts = [normalized]
    
    start_part = parts[0].strip()
    end_part = parts[1].strip() if len(parts) > 1 else None
    
    def parse_single_time(t_str):
        if not t_str: return None
        # Add logic to handle "4 pm" (no minutes)
        # Formats to try
        formats = [
            "%I:%M %p", # 8:30 am
            "%H:%M",    # 16:00
            "%I %p",    # 4 pm, 8 am
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(t_str, fmt).time()
            except ValueError:
                continue
        return None

    start_time = parse_single_time(start_part)
    end_time = parse_single_time(end_part)
    
    return start_time, end_time

def main():
    sessions = []
    
    # Get location from user input
    print("Please enter the location for these sessions (e.g. San Diego):")
    location = input().strip()
    

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse Date
                date_obj = parse_date(row.get('date', ''))
                
                # Parse Times
                # parse_time now returns (start, end_from_range)
                start_time_obj, end_time_from_range = parse_time(row.get('time', ''))
                
                # explicit end_time column overrides range extraction if present (usually)
                # but if parse_time failed for end_time column, we might check extraction?
                # Let's parse the explicit end_time column separately.
                # It likely doesn't have ranges, but we can use the same function and take the first return.
                explicit_end_time_str = row.get('end_time', '')
                explicit_end_time_obj, _ = parse_time(explicit_end_time_str)
                
                # Use explicit if available, else use extracted
                end_time_obj = explicit_end_time_obj if explicit_end_time_obj else end_time_from_range
                
                start_iso = None
                end_iso = None
                
                if date_obj and start_time_obj:
                    start_dt = datetime.combine(date_obj, start_time_obj)
                    start_iso = start_dt.isoformat()
                    
                if date_obj and end_time_obj:
                    end_dt = datetime.combine(date_obj, end_time_obj)
                    if start_time_obj and end_time_obj < start_time_obj:
                         # Handle PM -> AM crossover? Or just assume it's next day?
                         # For now, let's assume same day unless we see extracted ranges that imply otherwise
                         pass
                    end_iso = end_dt.isoformat()

                # Clean up keys and create new structure
                session = {
                    "title": row.get('title'),
                    "type": row.get('type'),
                    "date": date_obj.isoformat() if date_obj else None,
                    "start_datetime": start_iso,
                    "end_datetime": end_iso,
                    "url": row.get('url'),
                    "speaker": row.get('speaker'),
                    "abstract": row.get('abstract'),
                    "overview": row.get('overview'),
                    "location": location
                }
                sessions.append(session)
                
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, indent=4)
            
        print(f"Successfully converted {len(sessions)} sessions to {OUTPUT_FILE}")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {INPUT_FILE}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
