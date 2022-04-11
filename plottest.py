import xlrd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    pd_data = pd.read_excel("C:/Users/86130/Desktop/第三次测试/附件1：超级大卖场.xls", sheet_name=0)
    print(pd_data.head())
    # 根据列明划分
    p_new = pd_data.groupby(['细分']).size()
    print(p_new)
    plt.pie(p_new.tolist(), labels=p_new.index, autopct='%3.1f%%')
    plt.title('股票每年成交笔数饼图')  # 加标题
    plt.show()
