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

    #plt.pie(p_new.tolist(), labels=p_new.index, autopct='%3.1f%%')

    # plt.bar(x=range(len(ind)),  # 指定条形图x轴的刻度值(有的是用left，有的要用x)
    #         height=val,  # 指定条形图y轴的数值（python3.7不能用y，而应该用height）
    #         tick_label=ind,  # 指定条形图x轴的刻度标签
    #         color='steelblue',  # 指定条形图的填充色
    #         )
    # for i in range(len(val)):
    #     plt.text(i, val[i] + 0.1, "%s" % round(val[i], 1), ha='center')  # round(y,1)是将y值四舍五入到一个小数位

    # plt.scatter(ind, val, marker='o')

    # plt.plot(ind, val)
    # plt.title('股票每年成交笔数饼图')  # 加标题
    # plt.show()
    pd_data.drop(["行 ID","订单 ID","订单日期","发货日期","邮寄方式",	"客户 ID","客户名称","细分",
                  "城市","省/自治区","国家/地区","地区",	"产品 ID","类别","子类别",	"产品名称"],axis =1,inplace = True)
    corr = pd_data.corr()
    #ax = plt.subplots(figsize=(15,16))
    ax = sns.heatmap(corr,vmax=.8,square=True,annot=True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()