import requests
import pandas as pd

base = "https://s2025.conference-schedule.org"
candidates = ["/schedule.json", "/sessions.json", "/events.json", "/api/sessions"]

for path in candidates:
    resp = requests.get(base + path)
    if resp.ok and resp.headers.get("Content-Type", "").startswith("application/json"):
        data = resp.json()
        # often data["sessions"] or data["schedule"] holds a list of items
        items = data.get("sessions") or data.get("schedule") or data
        df = pd.json_normalize(items)
        df.to_csv("schedule.csv", index=False)
        print(f"✔️  Found JSON at {path} → schedule.csv")
        break
else:
    print("No JSON endpoint found; falling back to HTML parsing…")
