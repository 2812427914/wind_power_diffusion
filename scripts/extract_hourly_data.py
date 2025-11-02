import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = 'data/wtbdata_245days.csv'
HOURLY_DATA_PATH = 'data/wtbdata_hourly.csv'

def extract_hourly_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"原始数据文件不存在: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    # 去除全空行和Tmstamp为空的行
    df = df.dropna(how='all')
    df = df[df['Tmstamp'].notnull() & df['Day'].notnull()]
    # Day字段转为int，Tmstamp补零
    df['Day'] = df['Day'].astype(int)
    df['Tmstamp'] = df['Tmstamp'].astype(str).str.zfill(5)  # 兼容 0:00/00:00
    # 合成完整时间戳（假设Day为1表示第1天，起始日期自定义）
    start_date = pd.to_datetime('2020-01-01')  # 可根据实际情况调整
    df['Date'] = start_date + pd.to_timedelta(df['Day'] - 1, unit='D')
    # 统一时间格式
    df['Datetime'] = pd.to_datetime(df['Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Tmstamp'], errors='coerce')
    # 去除无法解析的时间
    df = df[df['Datetime'].notnull()]

    # 按小时聚合：同一风机在同一小时内的所有记录做数值特征均值
    df['Hour'] = df['Datetime'].dt.floor('H')

    # 仅将关心的列转为数值类型（保留 Wspd 作为特征，Patv 作为目标）
    non_numeric_cols = {'TurbID', 'Day', 'Tmstamp', 'Date', 'Datetime', 'Hour'}
    keep_numeric = [c for c in ['Wspd', 'Patv'] if c in df.columns]
    for col in keep_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 简单清洗异常值：风速和功率
    if 'Wspd' in df.columns:
        # 负风速或异常大风速视为无效
        df.loc[(df['Wspd'] < 0) | (df['Wspd'] > 50), 'Wspd'] = np.nan
    if 'Patv' in df.columns:
        # 明显异常的负功率置为NaN；小负值（如 -0.3 哨兵）置为0；异常大值置为NaN
        df.loc[df['Patv'] < -1, 'Patv'] = np.nan
        df.loc[(df['Patv'] >= -1) & (df['Patv'] < 0), 'Patv'] = 0.0
        df.loc[df['Patv'] > 3000, 'Patv'] = np.nan

    # 简单清洗异常值：风速和功率
    if 'Wspd' in df.columns:
        # 负风速或异常大风速视为无效
        df.loc[(df['Wspd'] < 0) | (df['Wspd'] > 50), 'Wspd'] = np.nan
    if 'Patv' in df.columns:
        # 明显异常的负功率置为NaN；小负值（如 -0.3 哨兵）置为0；异常大值置为NaN
        df.loc[df['Patv'] < -1, 'Patv'] = np.nan
        df.loc[(df['Patv'] >= -1) & (df['Patv'] < 0), 'Patv'] = 0.0
        df.loc[df['Patv'] > 3000, 'Patv'] = np.nan

    # 仅对选定列做均值
    numeric_cols = [c for c in ['Wspd', 'Patv'] if c in df.columns]
    agg = df.groupby(['TurbID', 'Hour'])[numeric_cols].mean().reset_index()

    # 重建时间辅助列
    agg['Datetime'] = agg['Hour']
    agg['Date'] = agg['Datetime'].dt.normalize()
    agg['Tmstamp'] = agg['Datetime'].dt.strftime('%H:00')
    agg['Day'] = (agg['Date'] - start_date).dt.days + 1

    # 排序与列顺序调整（仅保留 Wspd 特征与 Patv 目标）
    cols = ['TurbID', 'Day', 'Tmstamp'] + [c for c in ['Wspd', 'Patv'] if c in agg.columns] + ['Date', 'Datetime']
    df_hourly = agg.sort_values(['TurbID', 'Datetime'])[cols]

    # 保存新数据
    df_hourly.to_csv(HOURLY_DATA_PATH, index=False)
    print(f"原始数据量: {len(df)}, 按小时平均后的数据量: {len(df_hourly)}")
    print(f"已保存每小时平均数据到: {HOURLY_DATA_PATH}")

if __name__ == "__main__":
    extract_hourly_data()