import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 读取数据集
    file_path = '../processed_data/t018/ml_t018.csv'
    data = pd.read_csv(file_path)

    # 统计每个乘客的交互数量
    interaction_counts = data['u'].value_counts()

    # 计算描述性统计量
    median = interaction_counts.median()
    q1 = interaction_counts.quantile(0.25)
    q3 = interaction_counts.quantile(0.75)

    # 打印统计结果
    print(f"中位数 (Median): {median}")
    print(f"第一四分位数 (Q1): {q1}")
    print(f"第三四分位数 (Q3): {q3}")

    # 确定大部分乘客交互数量的范围
    print(f"大部分乘客的交互数量范围: {q1} - {q3}")

    # 定义直方图的分箱
    bins = range(0, 450, 25)  # 可根据需要调整步长

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  # 设置全局字体大小
    plt.rcParams['axes.titlesize'] = 20  # 设置标题字体大小
    plt.rcParams['axes.labelsize'] = 18  # 设置坐标轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 14  # 设置x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度字体大小

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(interaction_counts, bins=bins, edgecolor='black')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Passengers')
    plt.title('BJSubway-1M')
    plt.grid(False)
    plt.show()
