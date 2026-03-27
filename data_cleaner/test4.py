report = '{"quality_score": 100, "issues": []}'

# 字符串只能这样用
print(report)  # 打印出来看看

# 不能这样用
print(report["quality_score"])  # 报错！字符串不能取字段