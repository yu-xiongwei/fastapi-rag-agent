# import json

# # 第一步：字符串，不能取字段
# report = '{"quality_score": 100, "issues": []}'
# print(type(report))
# print(report["quality_score"])  # 这行会报错，没关系，就是要看报错

import json

report = '{"quality_score": 100, "issues": []}'

# 第二步：json.loads() 变成字典，才能取字段
result = json.loads(report)
print(type(result))
print(result["quality_score"])