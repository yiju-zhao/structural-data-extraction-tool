#!/bin/bash
# Extract oral session links for a given day
# Usage: ./extract_oral_day.sh <day> (e.g., 4/24)
DAY=$1
DAY_UNDERSCORE=$(echo "$DAY" | tr '/' '_')

echo "=== Processing day $DAY ==="

# Navigate to the day page
npx agent-browser open "https://iclr.cc/virtual/2026/day/$DAY"

# Wait for page to load
sleep 2

# Take snapshot to find "Show more" buttons
npx agent-browser snapshot -i > /tmp/iclr_snapshot_${DAY_UNDERSCORE}.txt 2>&1

# Click all "Show more" buttons that are inside oral session sections
# We use JS to click all show-more buttons except those in poster sessions
npx agent-browser eval "(() => {
  const sections = document.querySelectorAll('.session-box');
  let clicked = 0;
  sections.forEach(section => {
    const heading = section.querySelector('h3 a, h3');
    if (heading && heading.textContent.includes('Oral Session')) {
      const btn = section.querySelector('button[class*=\"show\"], button');
      if (btn && btn.textContent.includes('Show')) {
        btn.click();
        clicked++;
      }
    }
  });
  return 'Clicked ' + clicked + ' show-more buttons';
})()"

sleep 2

# Extract all oral links
npx agent-browser eval "(() => { const data = Array.from(document.querySelectorAll('a[href*=\"/virtual/2026/oral/\"]')).map(a => ({ title: a.textContent.trim(), url: 'https://iclr.cc' + a.getAttribute('href') })); return btoa(unescape(encodeURIComponent(JSON.stringify(data)))); })()" | python3 /Users/eason/Documents/HW-Project/Agent/structural-data-extraction-tool/projects/iclr-2025/configs/save_day.py "${DAY_UNDERSCORE}"
