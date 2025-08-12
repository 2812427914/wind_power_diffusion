import pandas as pd
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
    # 只保留整点（分钟==0）
    df_hourly = df[df['Datetime'].dt.minute == 0]
    # 按TurbID和Datetime排序
    df_hourly = df_hourly.sort_values(['TurbID', 'Datetime'])
    # 保存新数据
    df_hourly.to_csv(HOURLY_DATA_PATH, index=False)
    print(f"原始数据量: {len(df)}, 整点数据量: {len(df_hourly)}")
    print(f"已保存每小时数据到: {HOURLY_DATA_PATH}")

if __name__ == "__main__":
    extract_hourly_data()