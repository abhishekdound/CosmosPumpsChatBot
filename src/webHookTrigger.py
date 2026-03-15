import os

import requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = f"{os.getenv('NGROQ_FORWARDING')}/webhook/firecrawl"

response=requests.post(
    "https://api.firecrawl.dev/v2/crawl",
    headers={"Authorization": f"Bearer {os.getenv('FIRECRAWL_API_KEY')}" , "Content-Type": "application/json"},
    json={
        "url": "https://en.wikipedia.org/wiki/Virat_Kohli",
        "webhook": { "url": WEBHOOK_URL , "events": ["page", "completed"] }
    }
)

if response.status_code == 200:
    print("Crawl started successfully!")
    print("Job ID:", response.json().get("id"))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
