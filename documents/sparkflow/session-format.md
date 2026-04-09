# Sessions Import Format

## Schema Notes

**Fields:** `venue`, `year`, `sessions[].title`, `sessions[].date`, `sessions[].speaker`, `sessions[].sessionFormat`, `sessions[].hasRecording`, `sessions[].intendedAudience`, `sessions[].publicationTitles`

- Minimal format is `{ venue, year, sessions }`.
- `sessionFormat` accepts `IN_PERSON`, `VIRTUAL`, or `BOTH`.
- `hasRecording` defaults to `false` when omitted, and `publicationTitles` should match publication titles exactly.

## Example JSON

```json
{
  "venue": "CHI",
  "year": 2024,
  "sessions": [
    {
      "title": "Virtual Reality and Immersive Experiences",
      "type": "Paper Session",
      "date": "2024-05-12",
      "startTime": "09:00",
      "endTime": "10:30",
      "location": "Room 301A",
      "abstract": "This session features cutting-edge research on virtual reality interfaces and immersive user experiences.",
      "sessionUrl": "https://programs.sigchi.org/chi/2024/program/session/vr-immersive",
      "topic": [
        "Virtual Reality",
        "Immersive Experiences"
      ],
      "affiliation": [
        "Stanford University",
        "MIT Media Lab"
      ],
      "technology": [
        "Unity",
        "OpenXR"
      ],
      "sessionFormat": "IN_PERSON",
      "hasRecording": true,
      "intendedAudience": "Researchers and practitioners building immersive systems.",
      "publicationTitles": [
        "Understanding User Behavior in Virtual Reality Environments"
      ]
    },
    {
      "title": "Accessibility and Inclusive Design",
      "type": "Paper Session",
      "date": "2024-05-12",
      "startTime": "11:00",
      "endTime": "12:30",
      "location": "Room 302B",
      "abstract": "Papers exploring accessibility challenges and solutions for diverse user populations.",
      "sessionUrl": "https://programs.sigchi.org/chi/2024/program/session/accessibility",
      "topic": [
        "Accessibility",
        "Inclusive Design"
      ],
      "affiliation": [
        "University of Washington",
        "Google Research"
      ],
      "technology": [
        "Screen Readers",
        "ARIA"
      ],
      "sessionFormat": "BOTH",
      "hasRecording": false,
      "intendedAudience": "Accessibility specialists, frontend engineers, and UX researchers.",
      "publicationTitles": [
        "AI-Powered Accessibility Tools for Web Navigation"
      ]
    },
    {
      "title": "Future of Remote Collaboration",
      "type": "Panel",
      "date": "2024-05-13",
      "startTime": "14:00",
      "endTime": "15:30",
      "location": "Main Hall",
      "speaker": [
        "Dr. Jane Doe"
      ],
      "abstract": "A panel discussion on emerging trends in remote collaboration tools and practices.",
      "overview": "Industry experts discuss the evolution of collaboration tools post-pandemic.",
      "transcript": "Dr. Jane Doe: Welcome everyone to this panel on the future of remote collaboration...",
      "sessionUrl": "https://programs.sigchi.org/chi/2024/program/session/remote-collab-panel",
      "topic": [
        "Remote Collaboration",
        "CSCW"
      ],
      "affiliation": [
        "Microsoft Research",
        "Carnegie Mellon University"
      ],
      "technology": [
        "WebRTC",
        "Spatial Audio"
      ],
      "sessionFormat": "VIRTUAL",
      "hasRecording": true,
      "intendedAudience": "Collaboration tool builders and conference organizers.",
      "publicationTitles": [
        "Collaborative Design in Distributed Teams",
        "Paper That Does Not Exist"
      ]
    }
  ]
}
```