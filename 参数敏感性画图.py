import matplotlib.pyplot as plt

if __name__ == '__main__':

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 16  # 设置全局字体大小
    plt.rcParams['axes.titlesize'] = 20  # 设置标题字体大小
    plt.rcParams['axes.labelsize'] = 18  # 设置坐标轴标签字体大小
    plt.rcParams['xtick.labelsize'] = 14  # 设置x轴刻度字体大小
    plt.rcParams['ytick.labelsize'] = 14  # 设置y轴刻度字体大小

    # 数据
    x = [10, 20, 32, 40, 80]
    roc_values = [0.9462, 0.9539, 0.9591, 0.9594, 0.9630]
    ap_values = [0.9603, 0.9664, 0.9699, 0.9712, 0.9713]

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制ROC值的折线图
    plt.plot(x, roc_values, marker='o', linestyle='-', color='b', label='ROC')

    # 绘制AP值的折线图
    plt.plot(x, ap_values, marker='s', linestyle='--', color='r', label='AP')

    # 添加标题和标签
    plt.title('BJSubway-40K')
    plt.xlabel('Length of historical interaction sequence')
    plt.ylabel('Values')
    plt.grid(False)
    plt.ylim(0.935, 0.98)

    # 添加图例
    plt.legend()
    plt.show()
