#!/usr/bin/env python3
"""
GTC 2026 Lightning Talks Session Extractor

Two-step extraction:
1. Extract session URLs from the catalog page
2. Extract detailed session information from each detail page

Output: JSON file with full session details
"""

import json
import re
import time
import subprocess
from pathlib import Path
from typing import Optional


def run_browser_command(cmd: str, timeout: int = 30) -> str:
    """Execute an agent-browser command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {cmd[:50]}...")
        return ""
    except Exception as e:
        print(f"Error running command: {e}")
        return ""


def open_browser(url: str) -> bool:
    """Open browser and navigate to URL."""
    result = run_browser_command(f'agent-browser open "{url}"', timeout=60)
    time.sleep(3)  # Wait for page to load
    return "✓" in result or "Session Catalog" in result


def close_browser():
    """Close the browser session."""
    run_browser_command("agent-browser close")


def eval_js(script: str) -> str:
    """Evaluate JavaScript in the browser and return result."""
    # Escape the script for shell
    escaped_script = script.replace('"', '\\"').replace('`', '\\`')
    return run_browser_command(f'agent-browser eval "{escaped_script}"', timeout=30)


def scroll_and_collect_sessions() -> list[dict]:
    """
    Step 1: Scroll through the catalog page and collect all session URLs.
    """
    print("Step 1: Extracting session URLs from catalog...")

    sessions = []
    seen_ids = set()

    # JavaScript to extract session data from cards
    extract_script = """
    Array.from(document.querySelectorAll('a')).filter(a => /\\\\[S8\\\\d{4}\\\\]/.test(a.textContent)).map(a => {
        const match = a.textContent.match(/\\\\[S(\\\\d{4,5})\\\\]/);
        return {
            session_id: match ? 'S' + match[1] : null,
            title: a.textContent.replace(/\\\\[S\\\\d+\\\\]/, '').trim(),
            url: a.href
        };
    }).filter(s => s.session_id)
    """

    # Scroll multiple times to load all sessions
    for i in range(5):
        print(f"  Scrolling... ({i+1}/5)")

        # Extract current sessions
        result = eval_js(extract_script)
        try:
            current_sessions = json.loads(result) if result.startswith('[') else []
            for session in current_sessions:
                if session['session_id'] not in seen_ids:
                    seen_ids.add(session['session_id'])
                    sessions.append(session)
                    print(f"    Found: {session['session_id']} - {session['title'][:50]}...")
        except json.JSONDecodeError:
            pass

        # Scroll down
        run_browser_command("agent-browser scroll down 2000")
        time.sleep(2)

    print(f"  Total sessions found: {len(sessions)}")
    return sessions


def extract_session_details(session_url: str) -> Optional[dict]:
    """
    Step 2: Extract detailed information from a session detail page.
    """
    # Navigate to session detail page
    result = run_browser_command(f'agent-browser open "{session_url}"', timeout=60)
    time.sleep(2)

    # Extract all text content and parse it
    detail_script = """
    (() => {
        const body = document.body.innerText;

        // Extract title with session ID
        const titleMatch = body.match(/([^\\n]+\\[S\\d{4,5}\\])/);
        const title = titleMatch ? titleMatch[1].trim() : '';

        // Extract session ID
        const idMatch = title.match(/\\[S(\\d{4,5})\\]/);
        const sessionId = idMatch ? 'S' + idMatch[1] : '';

        // Extract session type and format
        const sessionType = body.includes('Lightning Talk') ? 'Lightning Talk' : 'Unknown';
        const format = body.includes('In-Person') ? 'In-Person' : (body.includes('Virtual') ? 'Virtual' : 'Unknown');

        // Extract speakers - look for lines with | separators (Name | Title | Company)
        const speakerPattern = /^([A-Z][a-z]+ [A-Z][a-z]+(?:\\s[A-Z][a-z]+)?)\\s*\\|\\s*([^|]+)\\|\\s*(.+?)$/gm;
        const speakers = [];
        let speakerMatch;
        const lines = body.split('\\n');
        for (const line of lines) {
            const parts = line.split('|').map(p => p.trim());
            if (parts.length >= 2 && parts[0].match(/^[A-Z][a-z]+ [A-Z]/)) {
                speakers.push({
                    name: parts[0],
                    title: parts[1] || '',
                    company: parts[2] || ''
                });
            }
        }

        // Extract description - text after speakers and before metadata
        const descStart = body.indexOf(sessionId) + sessionId.length + 1;
        let description = '';
        const bodyAfterTitle = body.substring(descStart);
        const industryIdx = bodyAfterTitle.indexOf('Industry:');
        if (industryIdx > 0) {
            // Get text between title and Industry:
            const textBetween = bodyAfterTitle.substring(0, industryIdx);
            // Skip speaker lines
            const descLines = textBetween.split('\\n').filter(line => {
                return line.trim().length > 50 && !line.includes('|');
            });
            description = descLines.join(' ').trim();
        }

        // Extract metadata fields
        const industryMatch = body.match(/Industry:\\s*([^\\n]+)/);
        const topicMatch = body.match(/Topic:\\s*([^\\n]+)/);
        const levelMatch = body.match(/Technical Level:\\s*([^\\n]+)/);
        const audienceMatch = body.match(/Intended Audience:\\s*([^\\n]+)/);
        const techMatch = body.match(/NVIDIA Technology:\\s*([^\\n]+)/);

        // Extract key takeaways
        const takeawaysStart = body.indexOf('Key Takeaways:');
        let takeaways = [];
        if (takeawaysStart > 0) {
            const afterTakeaways = body.substring(takeawaysStart + 15);
            const exploreIdx = afterTakeaways.indexOf('Explore');
            const takeawaysSection = exploreIdx > 0 ? afterTakeaways.substring(0, exploreIdx) : afterTakeaways.substring(0, 1000);
            takeaways = takeawaysSection.split('\\n')
                .map(line => line.trim())
                .filter(line => line.length > 20);
        }

        // Filter out invalid speakers (like "Register Now | Log In")
        const validSpeakers = speakers.filter(s => {
            return s.name && !s.name.includes('Register') && !s.name.includes('Log In') && s.name.length > 2;
        });

        return {
            session_id: sessionId,
            title: title.replace(/\\[S\\d+\\]/, '').trim(),
            session_type: sessionType,
            format: format,
            speakers: validSpeakers.slice(0, 10),  // Limit to 10 speakers
            description: description.substring(0, 2000),
            industry: industryMatch ? industryMatch[1].trim() : '',
            topic: topicMatch ? topicMatch[1].trim() : '',
            technical_level: levelMatch ? levelMatch[1].trim() : '',
            intended_audience: audienceMatch ? audienceMatch[1].trim() : '',
            nvidia_technology: techMatch ? techMatch[1].trim() : '',
            key_takeaways: takeaways,
            url: window.location.href
        };
    })()
    """

    result = eval_js(detail_script)
    try:
        if result.startswith('{'):
            return json.loads(result)
    except json.JSONDecodeError:
        pass

    return None


def main():
    output_dir = Path("/Users/eason/Documents/HW Project/Agent/structural-data-extraction-tool/output")
    output_file = output_dir / "gtc2026_lightning_talks.json"

    print("=" * 60)
    print("GTC 2026 Lightning Talks Session Extractor")
    print("=" * 60)

    # Step 1: Open catalog and extract session URLs
    catalog_url = "https://www.nvidia.com/gtc/session-catalog/?sessionTypes=Lightning%20Talk"
    print(f"\nOpening catalog: {catalog_url}")

    if not open_browser(catalog_url):
        print("Failed to open catalog page")
        return

    # Accept cookies if present
    run_browser_command("agent-browser eval \"document.querySelector('button[class*=Accept]')?.click()\"")
    time.sleep(1)

    # Collect all session URLs
    sessions = scroll_and_collect_sessions()

    if not sessions:
        print("No sessions found!")
        close_browser()
        return

    # Step 2: Extract details for each session
    print("\n" + "=" * 60)
    print("Step 2: Extracting session details...")
    print("=" * 60)

    detailed_sessions = []
    for i, session in enumerate(sessions):
        print(f"\n[{i+1}/{len(sessions)}] Extracting: {session['session_id']}")

        details = extract_session_details(session['url'])
        if details:
            detailed_sessions.append(details)
            print(f"  ✓ Title: {details['title'][:60]}...")
            print(f"  ✓ Speakers: {len(details['speakers'])}")
        else:
            # If detail extraction failed, keep basic info
            detailed_sessions.append({
                "session_id": session['session_id'],
                "title": session['title'],
                "url": session['url'],
                "extraction_error": True
            })
            print(f"  ✗ Failed to extract details")

    # Close browser
    close_browser()

    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    output_data = {
        "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_url": catalog_url,
        "session_type": "Lightning Talk",
        "total_sessions": len(detailed_sessions),
        "sessions": detailed_sessions
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(detailed_sessions)} sessions to:")
    print(f"  {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for session in detailed_sessions:
        print(f"  • [{session['session_id']}] {session['title'][:55]}...")


if __name__ == "__main__":
    main()
