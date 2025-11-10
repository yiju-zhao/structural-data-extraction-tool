#!/usr/bin/env python3
"""
Utility to merge matched team data into session detail CSV.

Reads matched embedding results and session details, then combines them
by adding matched_team, recommendation_reason, and focus_area columns.
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List


def load_matched_data(matched_csv_path: str) -> pd.DataFrame:
    """Load the matched embedding CSV file."""
    df = pd.read_csv(matched_csv_path)
    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df


def load_detail_data(detail_csv_path: str) -> pd.DataFrame:
    """Load the session detail CSV file."""
    df = pd.read_csv(detail_csv_path)
    # Strip whitespace from column names and string values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df


def aggregate_matches_by_session(matched_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Group matches by session_title and aggregate multiple BU matches.

    Returns a dictionary mapping session_title to aggregated match info:
    {
        'session_title': {
            'matched_team': 'bu1; bu2',
            'recommendation_reason': 'bu1: reason1; bu2: reason2',
            'focus_area': 'bu1: area1; bu2: area2'
        }
    }
    """
    aggregated = {}

    # Group by session_title
    grouped = matched_df.groupby('session_title')

    for session_title, group in grouped:
        teams = []
        reasons = []
        focus_areas = []

        for _, row in group.iterrows():
            bu_name = row['bu_name']
            teams.append(bu_name)

            # Add BU prefix to reason and focus_area
            reason = row.get('recommendation_reason', '')
            if pd.notna(reason) and reason:
                reasons.append(f"{bu_name}: {reason}")

            focus_area = row.get('focus_areas', '')
            if pd.notna(focus_area) and focus_area:
                focus_areas.append(f"{bu_name}: {focus_area}")

        aggregated[session_title] = {
            'matched_team': '; '.join(teams),
            'recommendation_reason': '; '.join(reasons),
            'focus_area': '; '.join(focus_areas)
        }

    return aggregated


def merge_data(detail_df: pd.DataFrame,
               matched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge matched data into detail dataframe.

    Adds three new columns: matched_team, recommendation_reason, focus_area
    """
    # Aggregate matches by session title
    aggregated_matches = aggregate_matches_by_session(matched_df)

    # Create new columns
    detail_df['matched_team'] = ''
    detail_df['recommendation_reason'] = ''
    detail_df['focus_area'] = ''

    # Match by title
    for idx, row in detail_df.iterrows():
        session_title = row['title']

        # Look for match in aggregated data
        if session_title in aggregated_matches:
            match_info = aggregated_matches[session_title]
            detail_df.at[idx, 'matched_team'] = match_info['matched_team']
            detail_df.at[idx, 'recommendation_reason'] = match_info['recommendation_reason']
            detail_df.at[idx, 'focus_area'] = match_info['focus_area']

    return detail_df


def merge_session_matches(matched_csv_path: str,
                          detail_csv_path: str,
                          output_csv_path: str) -> None:
    """
    Main function to merge matched sessions with detail CSV.

    Args:
        matched_csv_path: Path to matched embedding CSV
        detail_csv_path: Path to session detail CSV
        output_csv_path: Path for output merged CSV
    """
    print(f"Loading matched data from: {matched_csv_path}")
    matched_df = load_matched_data(matched_csv_path)
    print(f"  Found {len(matched_df)} matched records")

    print(f"Loading detail data from: {detail_csv_path}")
    detail_df = load_detail_data(detail_csv_path)
    print(f"  Found {len(detail_df)} session records")

    print("Merging data...")
    merged_df = merge_data(detail_df, matched_df)

    # Count how many sessions have matches
    matched_count = (merged_df['matched_team'] != '').sum()
    print(f"  Matched {matched_count} sessions")

    print(f"Saving merged data to: {output_csv_path}")
    merged_df.to_csv(output_csv_path, index=False)
    print("Done!")


def main():
    """CLI entry point."""
    if len(sys.argv) != 4:
        print("Usage: python merge_session_matches.py <matched_csv> <detail_csv> <output_csv>")
        print("\nExample:")
        print("  python merge_session_matches.py \\")
        print("    neurips_2025_sessions_SanDiego_matched_embedding_v1.csv \\")
        print("    neurips_2025_sessions_SanDiego_detail.csv \\")
        print("    neurips_2025_sessions_SanDiego_merged.csv")
        sys.exit(1)

    matched_csv = sys.argv[1]
    detail_csv = sys.argv[2]
    output_csv = sys.argv[3]

    # Validate input files exist
    if not Path(matched_csv).exists():
        print(f"Error: Matched CSV file not found: {matched_csv}")
        sys.exit(1)

    if not Path(detail_csv).exists():
        print(f"Error: Detail CSV file not found: {detail_csv}")
        sys.exit(1)

    merge_session_matches(matched_csv, detail_csv, output_csv)


if __name__ == "__main__":
    main()
