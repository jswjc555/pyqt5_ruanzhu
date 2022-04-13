import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import *

def get_kpic(n_clusters):
    n_clusters = ans
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                          , ith_cluster_silhouette_values
                          , facecolor=color
                          , alpha=0.7
                          )
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , str(i))
        y_lower = y_upper + 10
    ax1.set_title("轮廓系数可视化")
    ax1.set_xlabel("轮廓系数")
    ax1.set_ylabel("聚类标签")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1]
                , marker='o'
                , s=10
                , c=colors
                )
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='x',
                c="red", alpha=1, s=200)

    ax2.set_title("聚类数据可视化")
    ax2.set_xlabel("经度")
    ax2.set_ylabel("纬度")
    plt.suptitle(("K-means聚类结果可视化与轮廓系数 "
                  "当聚类中心 = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    pd_data = pd.read_excel("C:/Users/86130/Desktop/第三次测试/数据/超级answerr.xls", sheet_name=12)
    dict = {}
    print(pd_data["城市"][0])
    # for i in range(len(pd_data)):
    X = np.array([pd_data["经度"],pd_data["纬度"]])
    X = X.T

    ans = 0
    last = 0
    for n_clusters in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        n_clusters = n_clusters
        clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
        cluster_labels = clusterer.labels_
        silhouette_avg = silhouette_score(X, cluster_labels)
        if silhouette_avg > last:
            ans = n_clusters
            last = silhouette_avg
        print("当聚类中心数量 =", n_clusters,
              "时，K-means聚类轮廓系数为 :", silhouette_avg)
    print("当聚类中心数量为", ans, "时，轮廓系数最大，聚类效果最好")
    print("聚类中心数量为", ans, "聚类分析...")
    # get_kpic(ans)
    cluster = KMeans(n_clusters=ans, random_state=0).fit(X)  # 实例化并训练模型
    y_pred = cluster.labels_  # 重要属性labels_，查看聚好的类别
    centroid = cluster.cluster_centers_
    print(y_pred)
    distance = []
    for i in range(len(pd_data["城市"])):
        # print(centroid[y_pred[i]])
        print(pd_data["经度"][i])
        distance.append(sqrt(pow(centroid[y_pred[i]][0]-pd_data["经度"][i],2)+pow(centroid[y_pred[i]][1]-pd_data["纬度"][i],2)))
    print(distance)

