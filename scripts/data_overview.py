import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # 数据路径
    data_path = 'data/wtbdata_hourly.csv'
    os.makedirs('results', exist_ok=True)
    overview_path = 'results/data_overview.txt'
    with open(overview_path, 'w', encoding='utf-8') as f:
        if not os.path.exists(data_path):
            f.write(f"数据文件未找到: {data_path}\n")
            print(f"数据文件未找到: {data_path}")
            return

        # 读取数据
        df = pd.read_csv(data_path)

        # 1. 基本信息 
        import io
        f.write("==== 基本信息 ====\n")
        buf = io.StringIO()
        df.info(buf=buf)
        f.write(buf.getvalue() + '\n')
        f.write("==== 前5行样例 ====\n")
        f.write(df.head().to_string() + '\n\n')

        # 2. 缺失值统计
        f.write("==== 缺失值统计 ====\n")
        f.write(df.isnull().sum().to_string() + '\n\n')

        # 3. 主要数值字段的描述性统计
        f.write("==== 数值字段描述性统计 ====\n")
        f.write(df.describe().to_string() + '\n\n')

        # 4. 各风机样本数
        f.write("==== 各风机样本数 ====\n")
        if 'TurbID' in df.columns:
            f.write(df['TurbID'].value_counts().head().to_string() + '\n\n')
        else:
            f.write("未找到 TurbID 字段。\n\n")

    # 5. 目标变量（Patv）分布可视化
    plt.figure(figsize=(8,4))
    if 'Patv' in df.columns:
        df['Patv'].hist(bins=50)
        plt.title('Patv（有功功率）分布')
        plt.xlabel('Patv (kW)')
        plt.ylabel('样本数')
        plt.tight_layout()
        plt.savefig('results/patv_hist.png')
        plt.close()
    else:
        with open(overview_path, 'a', encoding='utf-8') as f:
            f.write("未找到 Patv 字段，无法绘制功率分布直方图。\n")

    # 6. 风速与功率关系散点图（采样1000点）
    if 'Wspd' in df.columns and 'Patv' in df.columns:
        sample_df = df.sample(n=min(1000, len(df)), random_state=42)
        plt.figure(figsize=(6,4))
        plt.scatter(sample_df['Wspd'], sample_df['Patv'], alpha=0.3)
        plt.xlabel('Wspd (风速, m/s)')
        plt.ylabel('Patv (有功功率, kW)')
        plt.title('风速-功率关系')
        plt.tight_layout()
        plt.savefig('results/wspd_patv_scatter.png')
        plt.close()
    else:
        with open(overview_path, 'a', encoding='utf-8') as f:
            f.write("未找到 Wspd 或 Patv 字段，无法绘制风速-功率散点图。\n")

    print(f"\n已保存数据总览到 {overview_path}，Patv 分布直方图和风速-功率散点图到 results/ 文件夹。")

if __name__ == "__main__":
    main()