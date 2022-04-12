import xlrd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def get_data(p_new):
    if len(p_new.tolist()) <= 20:
        return p_new.tolist(), p_new.index
    dict = {}
    val = []
    ind = []
    for i in range(len(p_new.tolist())):
        dict[p_new.index[i]] = p_new.tolist()[i]
    t = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    i = 0
    n_qi = 0
    for x in t:
        if i < 20:
            ind.append(x[0])
            val.append(x[1])
        else:
            n_qi += x[1]
        i += 1
    # ind.append("其他")
    # val.append(n_qi)
    return val, ind


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    pd_data = pd.read_excel("C:/Users/86130/Desktop/第三次测试/附件1：超级大卖场.xls", sheet_name=0)
    # 根据列明划分
    p_new = pd_data.groupby(['细分']).size()

    val, ind = get_data(p_new)

    # plt.pie(val, labels=["1","2","3"], autopct='%3.1f%%')
    plt.pie(p_new.tolist(), labels=p_new.index, autopct='%3.1f%%')
    plt.title('股票每年成交笔数饼图')  # 加标题
    plt.show()
