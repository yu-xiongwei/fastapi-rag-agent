import pandas as pd
import logging

# ── 配置日志 ──────────────────────────────────────
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.FileHandler ("telco_cleaner.log", encoding = "utf-8"),
        logging.StreamHandler()
    ]

)

# ── 读取数据 ──────────────────────────────────────
df =df = pd.read_csv("archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
total_rows = len(df)
logging.info(f"读取完成，共 {total_rows} 行，{len(df.columns)} 列")

# ── 记录清洗前基本信息 ────────────────────────────
logging.info(f"列名 : {list(df.columns)}")
logging.info(f"名列缺失值 : \n{df.isnull().sum()}")

# ── Step1：去重 ───────────────────────────────────
before = len(df)
df = df.drop_duplicates()
logging.info(f"Step1 去重 : 删除 {before - len(df)} 行重复数据")

# ── Step2：处理 TotalCharges 的隐藏问题 ───────────
# 这个字段应该是数字，但 CSV 里是字符串，且有空格伪装成空值
logging.info(f"Step2 前 TotalCharges  类型 : {df['TotalCharges'].dtype}")
logging.info(f"TotalCharges 含空格的行数 : {(df['TotalCharges'].str.strip() == '').sum()}")

df['TotalCharges'] = df['TotalCharges'].str.strip()
df['TotalCharges'] = df['TotalCharges'].replace('', None)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
logging.info(f"Step2 TotalCharges 转换后类型 : {df['TotalCharges'].dtype}")
logging.info(f"Step2 转换后空值数量 : {df['TotalCharges'].isnull().sum()}")

# ── Step3：缺失值处理 ─────────────────────────────
missing_before = df.isnull().sum().sum()
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
missing_after = df.isnull().sum().sum()
logging.info(f"Step3 缺失值处理 : {missing_before} -> {missing_after}")

# ── Step4：字符串标准化 ───────────────────────────
str_cols = df.select_dtypes(include=['object', 'str']).columns
for col in str_cols:
    df[col] = df[col].str.strip().str.lower()
logging.info(f"Step4 字符串标准化完成, 处理列: {list(str_cols)}")

# ── Step5：业务逻辑校验 ───────────────────────────
invalid_tenure = (df['tenure'] < 0).sum()
invalid_charges = (df['MonthlyCharges'] <= 0).sum()
logging.info(f"Step5 业务校验 : tenure<0 的行= {invalid_tenure},MonthlyCharges<= 0 的行 = {invalid_charges}")

# ── 生成清洗报告 ──────────────────────────────────
clean_rows = len(df)
usable_rate = round(clean_rows / total_rows * 100, 1)
complete_rate= round((1 - df.isnull().sum().sum() / (clean_rows *len(df.columns))) * 100, 1)

print("\n" + "="*50)
print("📊 数据清洗报告")
print("="*50)
print(f"原始行数 :    {total_rows}")
print(f"清洗后行数 :  {clean_rows}")
print(f"数据可用率：  {usable_rate}%")
print(f"字段完整率:   {complete_rate}%")
print(f"发现问题:     TotalCharges 字段类型错误(字符串存数字)")
print(f"             TotalCharges 含空格伪装的空值")
print(f"             多列字符串首尾有空格")
print("="*50)
             
             