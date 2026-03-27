import json

def safe_json_parse(text: str) -> dict:
    text = text.strip()
    
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:-1]
        text = "\n".join(lines)
    
    return json.loads(text.strip())


case1 = '{"quality_score": 90, "issues": ["缺失率偏高"]}'

case2 = '''```json
{"quality_score": 85, "issues": ["重复率超标"], "is_production_ready": false}
````'''

case3 = '''```
{"quality_score": 70, "issues": ["主键有空值"]}
```'''

for i, case in enumerate([case1, case2, case3], 1):
    result = safe_json_parse(case)
    print(f"情况{i}: score={result['quality_score']}, issues={result['issues']}")
