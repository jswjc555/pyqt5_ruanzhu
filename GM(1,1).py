import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_year(date):
    return date[0:4]


def check_data(x0):
    interval = [np.exp(-2 / (len(x0) + 1)), np.exp(2 / (len(x0) + 2))]
    getSeries = lambda k: x0[k - 1] / x0[k]
    global lambda_k
    lambda_k = [getSeries(i) for i in range(2, len(x0))]
    if min(lambda_k) > interval[0] and max(lambda_k) < interval[1]:
        return 0
    # 计算出偏移量
    c = min(lambda_k) - interval[0] if min(lambda_k) - interval[0] < 0 else max(lambda_k) - interval[1]
    return c


def offset(x0, c):
    y0 = x0 - c
    return y0


def GM1_1(x0):
    # 验证数据是否可以用
    if check_data(x0) == 0:
        print("数据未通过级比检验,结果正确性有风险")
        return 0,0,0,0,0,False
    else:
        x0 = offset(x0,check_data(x0))
    # 计算AOG
    x0 = np.array(x0)
    x1 = np.cumsum(x0)
    # 计算x1的均值生成序列
    x1 = pd.DataFrame(x1)
    z1 = (x1 + x1.shift()) / 2.0  # 该项与该项的前一项相加除以2，用shift比循环代码更简单
    z1 = z1[1:].values.reshape((len(z1) - 1, 1))  # 再将刚刚算出的数据转成ndarray,删去nan
    B = np.append(-z1, np.ones_like(z1), axis=1)  # 合并数据形成B矩阵
    Y = x0[1:].reshape((len(x0) - 1, 1))  # 建立Y矩阵
    # 计算参数a,b np.dot为点乘,np.linalg.inv为矩阵求逆
    [[a], [b]] = np.dot(np.dot((np.linalg.inv(np.dot(B.T, B))), B.T), Y)
    # 方程求解 f(k+1)表示x1(k+1)
    f = lambda k: (x0[0] - b / a) * np.exp(-a * (k - 1)) - (x0[0] - b / a) * np.exp(-a * (k - 2))
    # 求出估计的所有x1的预测值,大小和x0一样
    x1_pre = [f(k) for k in range(1, len(x0) + 1)]
    # 求x0的预测值
    # x1_pre = pd.DataFrame(x1_pre)

    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    # 检验预测值，利用残差检验
    residual_error = np.abs(x0 - np.array(x1_pre))
    residual_error_max = residual_error.max
    # 级比偏差值检验

    return a, b, residual_error_max, f, x1_pre,True


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    pd_data = pd.read_excel("C:/Users/86130/Desktop/第三次测试/附件1：超级大卖场.xls", sheet_name=0)
    # 添加年份列
    pd_data["年份"] = pd_data.apply(lambda x: get_year(x["订单日期"]), axis=1)
    # #判断是否有该列
    # print("城市" in pd_data.columns)
    # #判断某列是否有某值
    # print("安达" in pd_data["城市"].tolist())

    # 获取unique年份
    data_many = pd_data[(pd_data['城市'] == "杭州")]
    grouped = data_many.groupby(data_many['年份'])
    groued_year = grouped["年份"].unique()
    year = []
    for i in groued_year:
        year.append(int(i[0]))

    #grouped = data_many.groupby(data_many["年份"])

    # 对分组数据进行统计求和
    grouped_sum = grouped.sum()
    print(grouped_sum)

    #print(grouped_sum["销售额"].tolist())
    # GM(1,1)预测
    x0 = grouped_sum["销售额"].tolist()

    a, b, residual_error_max, f, x1_pre,pplltt = GM1_1(x0)
    if pplltt:
        # 拟合值
        print(x1_pre)
        # 往后预测年数
        y_n = 5
        # 预测值
        x2_pre = []
        y_year = []
        for i in range(len(x1_pre)+1,len(x1_pre)+y_n+1):
            print(i)
            print(f(i))
            x2_pre.append(f(i))
            y_year.append(year[0]+i-1)
        print(y_year)

        # 画图
        plt.plot(year, x0, color='r', linestyle="dashdot", label='真实值')
        plt.plot(year, x1_pre, color='b', linestyle=":", label="拟合值")
        plt.plot(y_year, x2_pre, color='g', linestyle="dashed", label="预测值")
        plt.legend(loc='upper right')
        plt.show()