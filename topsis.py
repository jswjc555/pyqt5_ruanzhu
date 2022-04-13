import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def weights(x):
    newX = MinMaxScaler().fit_transform(x) + 1e-10#至于为啥有这个1e-10，你细品
    P = newX / newX.sum(axis=0)
    k = 1 / np.log(newX.shape[0])
    ej = -k * (P * np.log(P)).sum(axis=0)
    gj = 1 - ej
    wj = gj / gj.sum()
    return wj

def Standard(datas):
    K = np.power(np.sum(pow(datas,2),axis = 0),0.5)
    for i in range(len(K)):
        datas[: , i] = datas[: , i] / K[i]
    return datas

def Score(sta_data,w):
    z_max = np.amax(sta_data , axis=0)
    z_min = np.amin(sta_data , axis=0)
    # 计算每一个样本点与最大值的距离
    tmpmaxdist = np.power(np.sum(np.power((z_max - sta_data) , 2)*w , axis = 1) , 0.5)  # 每个样本距离Z+的距离
    tmpmindist = np.power(np.sum(np.power((z_min - sta_data) , 2)*w , axis = 1) , 0.5)  # 每个样本距离Z+的距离
    score = tmpmindist / (tmpmindist + tmpmaxdist)
    score = score / np.sum(score)  # 归一化处理
    return score

def get_valind(dict,top):
    val = []
    ind = []
    for i in range(top):
        val.append(t[i][1])
        ind.append(t[i][0])
    return val,ind


if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    pd_data = pd.read_excel("C:/Users/86130/Desktop/第三次测试/附件1：超级大卖场.xls", sheet_name=0)


    grouped = pd_data.groupby(pd_data["城市"])
    grouped_sum = grouped.sum()
    print(grouped_sum.columns)
    grouped_sum.drop(["行 ID"],axis =1,inplace = True)
    # print(grouped_sum.index)
    # print(np.array(grouped_sum.values))
    # print(grouped_sum.columns)
    # 熵权法权重
    w = weights(np.array(grouped_sum.values)).round(4)
    sta_data = Standard(np.array(grouped_sum.values))
    sco = Score(sta_data,w)
    print(len(sco))
    dict = {}
    for i in range(len(sco)):
        dict[grouped_sum.index[i]] = sco[i]
    t = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    val,ind = get_valind(t,12)
    plt.figure(figsize=(10, 4))
    plt.grid(alpha=0.4)
    plt.barh(ind, val, height=0.4, label="第二天", color="#FFC125")
    plt.yticks(ind, ind, fontsize=10)
    plt.xlabel("topsis评分",  fontsize=12)
    plt.ylabel("评价对象",  fontsize=12)
    plt.title("熵权法topsis评分分析",  fontsize=15)
    plt.show()

