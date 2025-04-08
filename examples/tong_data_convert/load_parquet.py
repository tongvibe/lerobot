import pandas as pd

# 指定你的 Parquet 文件路径
# file_path = '/root/.cache/huggingface/lerobot/IPEC-COMMUNITY/libero_goal_no_noops_lerobot/data/chunk-000/episode_000156.parquet'
file_path = '/home/tong/Downloads/episode_000000.parquet'

try:
    # 读取 Parquet 文件到 Pandas DataFrame
    # Pandas 会自动尝试使用已安装的引擎 (pyarrow 或 fastparquet)
    df = pd.read_parquet(file_path)

    # 查看 DataFrame 的基本信息 (列名, 数据类型, 非空值数量)
    print("DataFrame Info:")
    df.info()

    print("\n" + "="*30 + "\n")

    # 查看前 5 行数据
    print("First 5 rows:")
    print(df.head())

    # 查看数据的维度 (行数, 列数)
    print("\nDataFrame Shape:")
    print(df.shape)

    # 查看列名
    print("\nColumn Names:")
    print(df.columns)

    # 如果只想看某几列的前几行
    # print(df[['column_name1', 'column_name2']].head())

except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。")
except Exception as e:
    print(f"读取 Parquet 文件时出错: {e}")
    print("请确保安装了 'pyarrow' 或 'fastparquet' 库。")