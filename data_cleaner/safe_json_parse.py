import json

def safe_json_parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1]
        text = "\n".join(lines)
    return json.loads(text.strip())