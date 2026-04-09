# Publications Import Format

## Schema Notes

**Fields:** `venue`, `year`, `publications[].title`, `publications[].authors`, `publications[].summary`, `publications[].researchTopic`, `publications[].status`, `publications[].pdfUrl`

- Minimal format is `{ venue, year, publications }`.
- `publications[].summary` maps to the Publication model.
- Blank optional URLs are allowed, but valid URLs are preferred.

## Example JSON

```json
{
  "venue": "CHI",
  "year": 2024,
  "publications": [
    {
      "title": "Understanding User Behavior in Virtual Reality Environments",
      "authors": [
        "Alice Smith",
        "Bob Jones",
        "Carol White"
      ],
      "abstract": "This paper presents a comprehensive study of user behavior patterns in immersive virtual reality environments. We conducted a longitudinal study with 120 participants over 6 months.",
      "summary": "A longitudinal study of how participants adapt their behavior inside immersive VR over time.",
      "affiliations": [
        "Stanford University",
        "MIT"
      ],
      "countries": [
        "USA"
      ],
      "keywords": [
        "HCI",
        "VR",
        "user behavior",
        "immersive environments"
      ],
      "researchTopic": "Virtual Reality",
      "status": "Accepted",
      "rating": 4.5,
      "doi": "10.1145/3613904.3642001",
      "pdfUrl": "https://example.com/papers/vr-user-behavior.pdf",
      "githubUrl": "https://github.com/example/vr-study",
      "websiteUrl": "https://example.com/projects/vr-study"
    },
    {
      "title": "AI-Powered Accessibility Tools for Web Navigation",
      "authors": [
        "David Lee",
        "Emma Garcia"
      ],
      "abstract": "We introduce a novel AI-powered browser extension that enhances web accessibility for users with visual impairments through intelligent content summarization and navigation assistance.",
      "summary": "An AI browser extension that improves navigation and summarization for visually impaired users.",
      "affiliations": [
        "University of Washington",
        "Google Research"
      ],
      "countries": [
        "USA"
      ],
      "keywords": [
        "accessibility",
        "AI",
        "web navigation",
        "assistive technology"
      ],
      "researchTopic": "Accessibility",
      "status": "Best Paper",
      "rating": 4.8
    },
    {
      "title": "Collaborative Design in Distributed Teams",
      "authors": [
        "Frank Miller",
        "Grace Kim",
        "Henry Chen"
      ],
      "abstract": "An exploration of collaboration patterns and tool usage in geographically distributed design teams, with implications for remote work tool design.",
      "summary": "Observed collaboration behaviors across distributed product design teams.",
      "affiliations": [
        "Carnegie Mellon University"
      ],
      "countries": [
        "USA",
        "South Korea"
      ],
      "keywords": [
        "CSCW",
        "collaboration",
        "remote work",
        "design teams"
      ],
      "researchTopic": "CSCW",
      "status": "Poster"
    }
  ]
}
```