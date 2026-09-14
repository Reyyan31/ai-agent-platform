import os
import json
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
response = client.search("what is the latest version of Python", max_results=5)

with open("debug_tavily_output.json", "w", encoding="utf-8") as f:
    json.dump(response, f, indent=2, ensure_ascii=False)
print("Output written to debug_tavily_output.json")