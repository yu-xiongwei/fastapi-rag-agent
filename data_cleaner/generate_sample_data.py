import os
import csv

os.makedirs("data", exist_ok=True)

# 文件1：正常文件
with open("data/users_normal.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "name", "email", "age"])
    writer.writerow([1, "Alice", "alice@example.com", 25])
    writer.writerow([2, "Bob", "bob@example.com", 30])
    writer.writerow([2, "Bob", "bob@example.com", 30])  # 重复行

# 文件2：有问题的文件（格式错误，列数不一致）
with open("data/users_broken.csv", "w", newline="", encoding="utf-8") as f:
    f.write("id,name,email\n")
    f.write("1,Alice\n")          # 缺少列，会触发解析错误
    f.write("NOT_VALID_CSV,,\n")

# 文件3：空文件（只有0字节）
open("data/users_empty.csv", "w").close()

print("✅ 三个测试文件已生成！")