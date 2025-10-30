import json
import csv
import argparse
from typing import List, Dict

# ============================================
# CONFIGURE TYPES TO INCLUDE HERE
# ============================================
INCLUDED_TYPES = {
    "Workshop",
    "Tutorial",
    "Invited Talk",
    "Oral",
    "Expo Talk Panel",
    "Expo Demonstration",
    "Expo Workshop",
}


def load_events(json_file: str) -> List[Dict]:
    """Load events from JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("events", [])


def filter_events(events: List[Dict]) -> List[Dict]:
    """Filter events to include only specified types."""
    return [
        event for event in events if event.get("type", "").strip() in INCLUDED_TYPES
    ]


def clean_end_time(end_time: str) -> str:
    """Clean end_time field by removing 'ends' prefix."""
    return end_time.replace("ends", "").strip()


def save_to_csv(events: List[Dict], output_file: str):
    """Save events to CSV file."""
    if not events:
        print("No events to save!")
        return

    fieldnames = ["date", "time", "type", "title", "url", "end_time"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for event in events:
            event_copy = event.copy()
            event_copy["end_time"] = clean_end_time(event_copy.get("end_time", ""))
            row = {k: event_copy.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"Saved {len(events)} events to {output_file}")


def print_statistics(events: List[Dict]):
    """Print statistics about event types."""
    type_counts = {}
    for event in events:
        event_type = event.get("type", "Unknown")
        type_counts[event_type] = type_counts.get(event_type, 0) + 1

    print("\nEvent Type Statistics:")
    print("-" * 40)
    for event_type, count in sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{event_type:20s}: {count:4d}")
    print("-" * 40)
    print(f"{'Total':20s}: {len(events):4d}\n")


def main():
    parser = argparse.ArgumentParser(
        description=f"Filter conference events and export to CSV. Included types: {', '.join(INCLUDED_TYPES)}",
        epilog="""
Examples:
  python post_process_events.py events.json -o filtered_events.csv
        """,
    )

    parser.add_argument("input", type=str, help="Input JSON file")
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output CSV file"
    )

    args = parser.parse_args()

    # Load events
    print(f"Loading events from {args.input}...")
    events = load_events(args.input)
    print(f"Loaded {len(events)} total events")
    print_statistics(events)

    # Filter events
    print(f"Filtering for types: {', '.join(INCLUDED_TYPES)}")
    filtered_events = filter_events(events)
    print(f"Filtered to {len(filtered_events)} events")
    print_statistics(filtered_events)

    # Save to CSV
    save_to_csv(filtered_events, args.output)


if __name__ == "__main__":
    main()
