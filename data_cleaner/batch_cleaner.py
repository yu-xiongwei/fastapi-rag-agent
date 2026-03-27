"""
batch_cleaner.py — Day 2 实操
功能：批量处理 data/ 目录下所有 CSV 文件，出错跳过不崩溃，汇总成功/失败统计
知识点：with open / try/except 四种写法 / logging 五个级别 / 主动防御
"""

import os
import logging
import pandas as pd

# ============================================================
# 1. 配置 logging（同时输出到控制台 + cleaner.log 文件）
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("cleaner.log", encoding="utf-8"),
        logging.StreamHandler(stream=open(1, 'w', encoding='utf-8', closefd=False))
    ]
)


# ============================================================
# 2. DataCleaner 类（复用昨天的逻辑，独立可运行版本）
#    如果你有昨天的 data_cleaner.py，把这段删掉换成：
#    from data_cleaner import DataCleaner
# ============================================================
class DataCleaner:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def clean(self):
        # Step1：读取（on_bad_lines='error' 确保列数不一致时抛出异常）
        self.df = pd.read_csv(self.filepath, on_bad_lines='error')
        original_count = len(self.df)

        # Step2：去重
        self.df = self.df.drop_duplicates()

        # Step3：缺失值处理（数值列用中位数填充，其他列删除含空主键的行）
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # Step4：字符串清洗（strip + lower）
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].str.strip().str.lower()

        cleaned_count = len(self.df)
        logging.debug(f"  原始行数：{original_count}，清洗后：{cleaned_count}")

    def export_sql(self, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        table_name = os.path.basename(self.filepath).replace(".csv", "")

        with open(output_path, "w", encoding="utf-8") as f:
            # 建表语句
            cols = ", ".join([f"`{col}` TEXT" for col in self.df.columns])
            f.write(f"CREATE TABLE IF NOT EXISTS `{table_name}` ({cols});\n\n")

            # INSERT 语句
            for _, row in self.df.iterrows():
                values = ", ".join([f"'{str(v)}'" for v in row])
                f.write(f"INSERT INTO `{table_name}` VALUES ({values});\n")


# ============================================================
# 3. 处理单个文件（主动防御 + try/except）
# ============================================================
def process_file(filepath):
    """
    处理单个 CSV 文件。
    返回 True 表示成功，False 表示失败或跳过。
    """
    filename = os.path.basename(filepath)

    # ✅ 主动防御：提前判断空文件，意图清晰，不依赖 except 兜底
    if os.path.getsize(filepath) == 0:
        logging.warning(f"[跳过] {filename} 是空文件（主动检测）")
        return False

    try:
        cleaner = DataCleaner(filepath)
        cleaner.clean()

        output_path = f"output/{filename.replace('.csv', '_cleaned.sql')}"
        cleaner.export_sql(output_path)

        logging.info(f"[成功] {filename} → {output_path}")
        return True

    except pd.errors.ParserError as e:
        # 捕获特定异常：CSV 格式错误
        logging.error(f"[失败] {filename} CSV格式错误：{e}")
        return False

    except pd.errors.EmptyDataError as e:
        # 捕获特定异常：文件内容为空（有表头但无数据也会触发）
        logging.error(f"[失败] {filename} 数据为空：{e}")
        return False

    except Exception as e:
        # 兜底：未知错误，程序继续跑不崩溃
        logging.error(f"[失败] {filename} 未知错误：{e}")
        return False

    finally:
        # finally：无论成功失败都会执行
        logging.debug(f"[完成] {filename} 处理流程结束")


# ============================================================
# 4. 批量处理主函数
# ============================================================
def batch_clean(data_folder="data"):
    logging.info("=" * 50)
    logging.info(f"开始批量清洗，目录：{data_folder}")
    logging.info("=" * 50)

    os.makedirs("output", exist_ok=True)

    # 找出所有 CSV 文件
    csv_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith(".csv")
    ]

    if not csv_files:
        logging.warning("没有找到任何 CSV 文件！")
        return

    logging.info(f"共发现 {len(csv_files)} 个 CSV 文件")

    success_list = []
    failed_list = []

    for filepath in sorted(csv_files):
        result = process_file(filepath)
        if result:
            success_list.append(os.path.basename(filepath))
        else:
            failed_list.append(os.path.basename(filepath))

    # ---- 汇总统计 ----
    logging.info("=" * 50)
    logging.info("批量清洗完成，汇总如下：")
    logging.info(f"✅ 成功 {len(success_list)} 个：{success_list}")
    logging.info(f"❌ 失败/跳过 {len(failed_list)} 个：{failed_list}")
    logging.info(f"📊 总计：{len(csv_files)} 个文件")
    logging.info("=" * 50)


# ============================================================
# 5. 入口
# ============================================================
if __name__ == "__main__":
    batch_clean()