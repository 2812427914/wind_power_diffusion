import pandas as pd

def analyze_corr(csv_path):
    df = pd.read_csv(csv_path)
    # 选取数值型参数
    num_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
    df_num = df[num_cols].dropna()
    # 计算相关性
    corr = df_num.corr()['Patv'].drop('Patv').sort_values(ascending=False)
    print("参数与风电功率(Patv)的相关性：")
    for k, v in corr.items():
        print(f"{k:6s} : {v:.3f}")
    print("\n结论：")
    print("相关性绝对值越大，说明该参数与功率关系越强。通常风速（Wspd）与功率相关性最大，其次可能是温度、风向等。")

if __name__ == "__main__":
    analyze_corr("/Users/bytedance/Downloads/gits/wind_power_diffusion/data/wtbdata_245days.csv")