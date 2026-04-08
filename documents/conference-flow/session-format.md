# Session Upload JSON Format Guide

## Required Fields
| Field | Format | Example |
|-------|--------|---------|
| title | string | "Keynote: Future of AI" |
| date | YYYY-MM-DD | "2026-03-18" |
| start | HH:MM (24h) | "09:00" |
| end | HH:MM (24h) | "10:30" |

## Optional Fields
| Field | Description | Alias |
|-------|-------------|-------|
| session_id | Unique code (used as doc ID) | code |
| room | Room or venue name | location |
| speakers | Array of {name, title, company} | — |
| format | "In-Person", "Virtual", "Both" | — |
| recording | "Yes" or "No" | — |
| session_type | "Talk", "Panel", "Keynote", etc. | sessionType |
| topic | Primary topic/category | mainTopic |
| url | Link to official session page | — |
| key_themes | Array of topic tags | keyThemes |

## Speaker Object
```json
{ "name": "Dr. Jane Smith", "title": "Chief Scientist", "company": "NVIDIA" }
```

## Complete Example
```json
[
  {
    "session_id": "S62911",
    "title": "NVIDIA AI Factory Architecture Deep Dive",
    "date": "2026-03-18",
    "start": "09:00",
    "end": "10:30",
    "room": "Hall A",
    "speakers": [{ "name": "Jensen Huang", "title": "CEO", "company": "NVIDIA" }],
    "format": "In-Person",
    "recording": "Yes",
    "session_type": "Keynote",
    "topic": "AI Infrastructure",
    "url": "https://example.com/session/S62911",
    "key_themes": ["AI", "Infrastructure", "Data Center"]
  }
]
```

## Minimal Example
```json
[
  { "title": "Morning Keynote", "date": "2026-03-18", "start": "09:00", "end": "10:00" },
  { "title": "Lunch Workshop", "date": "2026-03-18", "start": "12:00", "end": "13:00" }
]
```

## Notes
- Maximum 1000 sessions per upload
- Sessions missing required fields are skipped
- If session_id matches an existing session, it will be overwritten
