# Extraction Plan: s2025conference-scheduleorg Thursday

## Pre-execution Setup
- [ ] File: Create `s2025conference-scheduleorg_THURSDAY.md` with header
- [ ] Browser: Go to https://s2025.conference-schedule.org
- [ ] Wait for main page to fully load
- [ ] Locate and click [THURSDAY] tab/button
- [ ] Wait for Thursday content to load
- [ ] Expand all collapsible/accordion sections for Thursday

## Session Extraction Loop
- [ ] Scan, find, and process all session containers (title, time, type, contributors)
- [ ] For each: Capture info, append to markdown, mark as processed
- [ ] Log progress (session counter/scroll positions)
- [ ] Scroll and repeat until "Chapters Party" found and saved

## Validation & Completion
- [ ] Record total sessions count
- [ ] Confirm "Chapters Party" session captured
- [ ] Verify file content and existence
- [ ] Check for duplicates

## Error & Recovery
- [ ] Retry or report errors if page/tab/sessions not found
- [ ] Ensure no infinite loops or premature completion