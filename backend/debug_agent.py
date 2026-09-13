from dotenv import load_dotenv
load_dotenv()

import os
import json
from openai import OpenAI
from app.tools import get_tool_descriptions

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))

tools = get_tool_descriptions()
print("TOOLS:", tools)

tools_summary = "\n".join(
    [f"- {name}: {description}" for name, description in tools.items()]
)

system_prompt = (
    "You are an AI routing agent. Your job is to select the best-fitting tool "
    "to fulfill the user's request.\n\n"
    f"Available tools:\n{tools_summary}\n\n"
    "Instructions:\n"
    '- Return a JSON object in the exact format: {"tool": "<tool_name or null>", "args": {}}\n'
    '- If \'echo\' tool is selected, args should be: {"text": "<text to echo>"}\n'
    '- If \'calculator\' tool is selected, args should be: {"expression": "<arithmetic expression>"}\n'
    '- If no available tool matches the request, set "tool" to null and "args" to null or {}\n'
)

print("\nSYSTEM PROMPT:\n", system_prompt)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "please repeat back: hello world"},
        ],
    )
    raw = response.choices[0].message.content
    print("\nRAW MODEL RESPONSE:", raw)

    parsed = json.loads(raw)
    print("\nPARSED:", parsed)
    print("Tool selected:", parsed.get("tool"))
    print("Is it in TOOL_REGISTRY keys?", parsed.get("tool") in tools)

except Exception as e:
    print("\n!!! EXCEPTION RAISED !!!")
    print(type(e).__name__, ":", e)