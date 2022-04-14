from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
from PyQt5.QtCore import *
import xlrd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from math import sqrt
import base64
import cv2
import os

# 使用 matplotlib中的FigureCanvas (在使用 Qt5 Backends中 FigureCanvas继承自QtWidgets.QWidget)
# from PyQt5.QtGui import *
# import hashlib
# import re
# import datetime

# UI--Logic分离
ui, _ = loadUiType('main.ui')
login, _ = loadUiType('login.ui')
db_file = "user_m.db"


class LoginApp(QWidget, login):
    def __init__(self):
        QWidget.__init__(self)
        self.setupUi(self)
        self.handle_ui_change()
        self.handle_buttons()
        qssStyle = '''
               QPalette{background:#EAF7FF;}*{outline:0px;color:#386487;}

QWidget[form="true"],QLabel[frameShape="1"]{
border:1px solid #C0DCF2;
border-radius:0px;
}

QWidget[form="bottom"]{
background:#DEF0FE;
}

QWidget[form="bottom"] .QFrame{
border:1px solid #386487;
}

QWidget[form="bottom"] QLabel,QWidget[form="title"] QLabel{
border-radius:0px;
color:#386487;
background:none;
border-style:none;
}

QWidget[form="title"],QWidget[nav="left"],QWidget[nav="top"] QAbstractButton{
border-style:none;
border-radius:0px;
padding:5px;
color:#386487;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}

QWidget[nav="top"] QAbstractButton:hover,QWidget[nav="top"] QAbstractButton:pressed,QWidget[nav="top"] QAbstractButton:checked{
border-style:solid;
border-width:0px 0px 2px 0px;
padding:4px 4px 2px 4px;
border-color:#00BB9E;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}

QWidget[nav="left"] QAbstractButton{
border-radius:0px;
color:#386487;
background:none;
border-style:none;
}

QWidget[nav="left"] QAbstractButton:hover{
color:#FFFFFF;
background-color:#00BB9E;
}

QWidget[nav="left"] QAbstractButton:checked,QWidget[nav="left"] QAbstractButton:pressed{
color:#386487;
border-style:solid;
border-width:0px 0px 0px 2px;
padding:4px 4px 4px 2px;
border-color:#00BB9E;
background-color:#EAF7FF;
}

QWidget[video="true"] QLabel{
color:#386487;
border:1px solid #C0DCF2;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}

QWidget[video="true"] QLabel:focus{
border:1px solid #00BB9E;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}

QLineEdit,QTextEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox,QDateEdit,QTimeEdit,QDateTimeEdit{
border:1px solid #C0DCF2;
border-radius:3px;
padding:2px;
background:none;
selection-background-color:#00BB9E;
selection-color:#FFFFFF;
}

QLineEdit:focus,QTextEdit:focus,QPlainTextEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus,QDateEdit:focus,QTimeEdit:focus,QDateTimeEdit:focus,QLineEdit:hover,QTextEdit:hover,QPlainTextEdit:hover,QSpinBox:hover,QDoubleSpinBox:hover,QComboBox:hover,QDateEdit:hover,QTimeEdit:hover,QDateTimeEdit:hover{
border:1px solid #C0DCF2;
}

QLineEdit[echoMode="2"]{
lineedit-password-character:9679;
}

            .QPushButton,.QToolButton{
    border-style:none;
    border:1px solid #C0DCF2;
    color:#386487;
    padding:5px;
    min-height:15px;
    border-radius:5px;
    background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
    }
    
    .QPushButton:hover,.QToolButton:hover{
    background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
    }
    
    .QPushButton:pressed,.QToolButton:pressed{
    background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
    }
        QGroupBox
        {
            border-radius: 5px;
            border-width: 1px;
            border-style: solid;
            border-color: darkCyan;
            margin-top: 5px;
        }
         
        .QGroupBox{
border:1px solid #C0DCF2;
border-radius:5px;
margin-top:3ex;
}

.QGroupBox::title{
subcontrol-origin:margin;
position:relative;
left:10px;
}
QStatusBar::item{
border:0px solid #DEF0FE;
border-radius:3px;
}
QLineEdit,QTextEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox,QDateEdit,QTimeEdit,QDateTimeEdit{
background:#EAF7FF;
}

QTabWidget::pane:top{top:-1px;}
QTabWidget::pane:bottom{bottom:-1px;}
QTabWidget::pane:left{right:-1px;}
QTabWidget::pane:right{left:-1px;}

*:disabled{
background:#EAF7FF;
border-color:#DEF0FE;
color:#C0DCF2;
}
QStatusBar::item{
border:0px solid #DEF0FE;
border-radius:3px;
}
              '''
        # 加载设置好的样式
        self.setStyleSheet(qssStyle)

    def show_regist(self):
        self.label_6.setText("")
        self.label_7.setText("")
        self.registBox.show()

    def hide_regist(self):
        self.registBox.hide()

    def handle_ui_change(self):
        self.hide_regist()

    def handle_login(self):
        if not len(self.login_username.text()):
            self.label_6.setText("用户名不能为空！")
            self.label_7.setText("")
        elif not len(self.login_psw.text()):
            self.label_7.setText("请输入密码！")
            self.label_6.setText("")
        elif len(self.login_username.text()) \
                and len(self.login_psw.text()):
            db_conn = sqlite3.connect(db_file)
            cur = db_conn.cursor()

            sql_select = "SELECT * FROM account WHERE user_name=\'" + self.login_username.text() + "\'"
            result = cur.execute(sql_select).fetchall()
            if len(result):
                if result[0][2] != self.login_psw.text():
                    self.label_7.setText("密码错误！")
                    self.label_6.setText("")
                    self.login_psw.setText("")
                else:
                    self.login_logButton.setText("")
                    self.login_psw.setText("")
                    self.main_app = MainApp(result[0][0])
                    self.close()
                    self.main_app.show()
            else:
                self.label_6.setText("用户未注册！")
                self.login_psw.setText("")

            cur.close()
            db_conn.close()

    def user_regist(self):
        # 请输入用户名
        if not len(self.regist_userlineEdit.text()):
            self.error_user.setText("用户名不能为空！")
            self.error_psw.setText("")
            self.error_confpsw.setText("")
        # 请输入一致的密码
        elif len(self.regist_pswlineEdit.text()) \
                and len(self.regist_confpswlineEdit.text()) \
                and self.regist_pswlineEdit.text() != self.regist_confpswlineEdit.text():
            self.error_psw.setText("两次密码不一致！")
            self.error_user.setText("")
            self.error_confpsw.setText("")
        # 请输入密码
        elif not len(self.regist_pswlineEdit.text()):
            self.error_psw.setText("密码不能为空！")
            self.error_user.setText("")
            self.error_confpsw.setText("")
        # 确认密码不能为空
        elif not len(self.regist_confpswlineEdit.text()):
            self.error_confpsw.setText("确认密码不能为空！")
            self.error_user.setText("")
            self.error_psw.setText("")
        # 注册，数据库插入数据

        else:
            db_conn = sqlite3.connect(db_file)
            cur = db_conn.cursor()

            sql_select = "SELECT * FROM account WHERE user_name=\'" + self.regist_userlineEdit.text() + "\'"
            result = cur.execute(sql_select)
            # 用户名与已有用户重复
            if len(result.fetchall()):
                self.error_user.setText("用户名已存在！")
                self.error_psw.setText("")
                self.error_confpsw.setText("")
                self.regist_pswlineEdit.setText("")
                self.regist_confpswlineEdit.setText("")
            # 注册成功
            else:
                sql = "insert into account (user_name,password) " \
                      "values(?,?)"
                data = (self.regist_userlineEdit.text(),
                        self.regist_pswlineEdit.text())
                cur.execute(sql, data)
                db_conn.commit()
                self.regist_userlineEdit.setText("")
                self.regist_pswlineEdit.setText("")
                self.regist_confpswlineEdit.setText("")
                self.selectInfo()
                self.hide_regist()
            cur.close()
            db_conn.close()

    def selectInfo(self):
        userBox = QMessageBox()
        userBox.setWindowTitle('注册')
        userBox.setText('注册成功')
        userBox.setStandardButtons(QMessageBox.Yes)
        buttonY = userBox.button(QMessageBox.Yes)
        buttonY.setText('确认')
        userBox.exec_()

    def handle_buttons(self):
        self.login_registButton.clicked.connect(self.show_regist)
        self.regist_backButton.clicked.connect(self.hide_regist)
        self.login_logButton.clicked.connect(self.handle_login)
        self.login_exitButton.clicked.connect(self.close)
        self.regist_confButton.clicked.connect(self.user_regist)


def get_kpic(ans, X):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
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


def Standard(datas):
    try:
        K = np.power(np.sum(pow(datas, 2), axis=0), 0.5)
    except Exception as e:
        print(e)
    for i in range(len(K)):
        datas[:, i] = datas[:, i] / K[i]
    return datas


def weights(x):
    newX = MinMaxScaler().fit_transform(x) + 1e-10  # 至于为啥有这个1e-10，你细品
    P = newX / newX.sum(axis=0)
    k = 1 / np.log(newX.shape[0])
    ej = -k * (P * np.log(P)).sum(axis=0)
    gj = 1 - ej
    wj = gj / gj.sum()
    return wj


def Score(sta_data, w):
    z_max = np.amax(sta_data, axis=0)
    z_min = np.amin(sta_data, axis=0)
    # 计算每一个样本点与最大值的距离
    tmpmaxdist = np.power(np.sum(np.power((z_max - sta_data), 2) * w, axis=1), 0.5)  # 每个样本距离Z+的距离
    tmpmindist = np.power(np.sum(np.power((z_min - sta_data), 2) * w, axis=1), 0.5)  # 每个样本距离Z+的距离
    score = tmpmindist / (tmpmindist + tmpmaxdist)
    score = score / np.sum(score)  # 归一化处理
    return score


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
        return 0, 0, 0, 0, 0, False
    else:
        x0 = offset(x0, check_data(x0))
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

    return a, b, residual_error_max, f, x1_pre, True


def get_valind(t, top):
    val = []
    ind = []
    for i in range(top):
        val.append(t[i][1])
        ind.append(t[i][0])
    return val, ind


class MainApp(QMainWindow, ui):
    def __init__(self, user_id):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.handle_ui_change()
        self.handle_buttons()
        self.user_id = user_id
        # 确认一和确认二分开
        self.datafile = ""
        self.sheet_name = 0
        self.queren = False
        self.datafile2 = ""
        self.sheet_name2 = 0
        self.queren2 = False
        self.long_la_topsisScore = {'唐寨': [116.566, 34.4418, 0.00151814643284275],
                                    '德惠': [125.703327, 44.533909, 0.000408782927458089],
                                    '洛阳': [113.391, 31.5138, 0.00235433135535845],
                                    '丰县': [116.592888, 34.696946, 0.00290464141646045],
                                    '韶关': [113.591544, 24.801322, 0.00314836610965642],
                                    '遂宁': [105.571331, 30.513311, 0.00135634257080459],
                                    '马坝': [113.535, 24.635, 0.00302093904725879],
                                    '徐州': [117.184811, 34.261792, 0.00313769809848123],
                                    '淮北': [116.794664, 33.971707, 0.00153481615365616],
                                    '夏镇': [117.099, 34.8656, 0.00210455983080517],
                                    '潼川': [105.038, 31.077, 0.00137067772432855],
                                    '濉溪': [116.767435, 33.916407, 0.00151138826452428],
                                    '扶余': [126.042758, 44.986199, 0.000421664560530269],
                                    '金乡': [116.310364, 35.06977, 0.00229464841685268],
                                    '滕州': [117.162098, 35.088498, 0.00221435418324383],
                                    '九台': [125.844682, 44.157155, 0.000491114362972526],
                                    '南充': [106.082974, 30.795281, 0.00137827191192205],
                                    '宿州': [116.984084, 33.633891, 0.00194799893290951],
                                    '澄江': [114.245, 24.889, 0.00289835776703347],
                                    '济宁': [116.587245, 35.415393, 0.00231798057056476],
                                    '商丘': [115.650497, 34.437054, 0.00258386292950662],
                                    '郴州': [113.032067, 25.793589, 0.00197554437262021],
                                    '枣庄': [117.557964, 34.856424, 0.00217566720035787],
                                    '清远': [113.051227, 23.685022, 0.00304366582027171],
                                    '山亭': [117.458968, 35.096077, 0.0020987371889581],
                                    '长春': [125.3245, 43.886841, 0.0014193951550065],
                                    '邳州': [117.963923, 34.314708, 0.00290742966080632],
                                    '兖州': [116.828996, 35.556445, 0.00215581782197475],
                                    '巨野': [116.089341, 35.390999, 0.00209734874901339],
                                    '南隆': [105.989, 31.3604, 0.00135029963026202],
                                    '彭城': [105.968, 31.466, 0.00145960260133959],
                                    '睢城': [117.93, 33.9079, 0.00290113422592978],
                                    '怀城': [112.099, 23.9715, 0.00301865417745852],
                                    '曲阜': [116.991885, 35.592788, 0.00211333065077372],
                                    '绵阳': [104.741722, 31.46402, 0.00140120299869828],
                                    '安埠': [126.691, 45.7257, 0.00301392589962244],
                                    '亳州': [115.782939, 33.869338, 0.00153889450392181],
                                    '广州': [113.280637, 23.125178, 0.00408882833165926],
                                    '定陶': [115.569601, 35.072701, 0.00210551419635694],
                                    '汶上': [116.487146, 35.721746, 0.00209821056980918],
                                    '德阳': [104.398651, 31.127991, 0.00136206330324362],
                                    '双城': [126.308784, 45.377942, 0.00106334170466631],
                                    '阆中': [105.975266, 31.580466, 0.00135805687811996],
                                    '宁阳': [116.799297, 35.76754, 0.00210071072015217],
                                    '归仁': [118.161, 33.7353, 0.00288435015322864],
                                    '乐城': [112.335, 23.4286, 0.0030113754224232],
                                    '内江': [105.066138, 29.58708, 0.00142104965442923],
                                    '平邑': [117.631884, 35.511519, 0.00213281317785199],
                                    '菏泽': [115.469381, 35.246531, 0.00214469087628749],
                                    '成都': [104.065735, 30.659462, 0.00167910175590099],
                                    '金沙': [113.187, 23.1479, 0.00289260195674256],
                                    '禄步': [112.387, 23.2649, 0.00300392563299978],
                                    '哈尔滨': [126.642464, 45.756967, 0.00190237153094812],
                                    '江油': [104.744431, 31.776386, 0.00138054186594105],
                                    '郸城': [115.189, 33.643852, 0.00253659565456146],
                                    '佛山': [113.122717, 23.028762, 0.00302294461158581],
                                    '石龙': [113.833, 23.1123, 0.00303568828372558],
                                    '河源': [114.697802, 23.746266, 0.00305847092402267],
                                    '耒阳': [112.847215, 26.414162, 0.00203501329343425],
                                    '肇庆': [112.472529, 23.051546, 0.0031845720375029],
                                    '东莞': [113.746262, 23.046237, 0.00302966332967106],
                                    '肇源': [125.081974, 45.518832, 0.00111706223996816],
                                    '重庆': [106.504962, 29.533155, 0.00209143883896869],
                                    '虎门': [113.797, 22.8587, 0.00301755163900957],
                                    '蒙阴': [117.943271, 35.712435, 0.00213515137912969],
                                    '深圳': [114.085947, 22.547, 0.00417077673622748],
                                    '榆树': [126.550107, 44.827642, 0.000395441826742128],
                                    '惠城': [114.413978, 23.079883, 0.0030955650101451],
                                    '新泰': [117.766092, 35.910387, 0.00215913447329073],
                                    '惠州': [114.412599, 23.079404, 0.00306280699500547],
                                    '自贡': [104.773447, 29.352765, 0.00137734896315384],
                                    '平阴': [116.455054, 36.286923, 0.00209753162946715],
                                    '江门': [113.094942, 22.590431, 0.00312500664882494],
                                    '阳谷': [115.784287, 36.113708, 0.00211371105922634],
                                    '云浮': [112.044439, 22.929801, 0.00301975940656711],
                                    '蚌埠': [117.363228, 32.939667, 0.00159667270115913],
                                    '临沂': [115.954, 32.5584, 0.002362099775511],
                                    '濮阳': [115.041299, 35.768234, 0.00252230472751652],
                                    '永丰': [112.095, 23.359, 0.00197865565187471],
                                    '合川': [106.265554, 29.990993, 0.000738511850888938],
                                    '五常': [127.15759, 44.919418, 0.00106635157125411],
                                    '周口': [114.649653, 33.620357, 0.00260502156986527],
                                    '山河屯': [127.225, 44.7114, 0.00106820519367783],
                                    '淡水': [114.492, 22.8591, 0.00300626563712312],
                                    '莱芜': [117.677736, 36.214397, 0.00213616586199024],
                                    '吉林市': [126.55302, 43.843577, 0.0005569277095457],
                                    '莘县': [115.667291, 36.237597, 0.00210370994505327],
                                    '西华': [114.530067, 33.784378, 0.00251229381262599],
                                    '皇岗': [114.085947, 22.547, 0.00310135411018101],
                                    '阿城': [126.972726, 45.538372, 0.0011100549675181],
                                    '坪山': [114.338441, 22.69423, 0.0030100260983848],
                                    '济南': [117.000923, 36.675807, 0.00249584368519284],
                                    '界首': [115.362117, 33.26153, 0.00151669156320651],
                                    '阜阳': [115.819729, 32.896969, 0.00151793562526091],
                                    '开封': [114.341447, 34.797049, 0.00269890707368874],
                                    '聊城': [115.980367, 36.456013, 0.00218010949830665],
                                    '浯溪': [111.829, 26.5377, 0.0022070116940889],
                                    '龙口': [112.902, 22.8024, 0.00211653800401276],
                                    '九江': [113.048, 22.8316, 0.00202583843077553],
                                    '江口': [111.48, 23.5469, 0.00225109410918667],
                                    '肇东': [125.991402, 46.069471, 0.00106526942108601],
                                    '淮南': [117.018329, 32.647574, 0.00160009061085483],
                                    '淮阴': [119.020817, 33.622452, 0.00292210383399694],
                                    '长清': [116.74588, 36.561049, 0.00211114870904108],
                                    '衡阳': [112.607693, 26.900358, 0.00207286228431584],
                                    '广元': [105.829757, 32.433668, 0.00140580943300392],
                                    '颍上城关镇': [116.259122, 32.637065, 0.00152442448710996],
                                    '珠海': [113.553986, 22.224979, 0.00315714812751317],
                                    '三岔子': [104.496, 30.2751, 0.000396663544701697],
                                    '呼兰': [126.603302, 45.98423, 0.00108977588045123],
                                    '台城': [112.708, 22.255, 0.0030122538081225],
                                    '南麻': [118.107, 36.1988, 0.00210822249829863],
                                    '冷水滩': [111.607156, 26.434364, 0.00197283957244664],
                                    '龙江': [114.249, 23.6382, 0.00106377063781918],
                                    '黄石': [115.207, 24.3521, 0.00170828722240937],
                                    '黎城': [118.994, 33.0034, 0.00290005546199549],
                                    '恩城': [112.245, 22.2371, 0.00304924301630314],
                                    '沂水': [118.634543, 35.787029, 0.00210668500889206],
                                    '临水': [115.954, 32.5584, 0.00149609830798173],
                                    '鹿城': [115.662, 32.6523, 0.00154039461218286],
                                    '十字路': [114.627, 33.1737, 0.0021089289127297],
                                    '兰西': [126.289315, 46.259037, 0.00109344870476699],
                                    '北碚': [106.437868, 29.82543, 0.00067400374299815],
                                    '鹤壁': [114.295444, 35.748236, 0.0025542207251543],
                                    '禹城': [116.642554, 36.934485, 0.00212541845309194],
                                    '梅州': [116.117582, 24.299112, 0.00306647872680512],
                                    '宝应': [119.321284, 33.23694, 0.00288991079401431],
                                    '明光': [117.998048, 32.781206, 0.00151405656443132],
                                    '永川': [105.894714, 29.348748, 0.000713060302566814],
                                    '汕尾': [115.364238, 22.774485, 0.0030453442561715],
                                    '临清': [115.713462, 36.842598, 0.00217976069292856],
                                    '阳春': [111.7905, 22.169598, 0.0030150919792843],
                                    '漯河': [114.026405, 33.575855, 0.00251449634251995],
                                    '前郭': [124.826808, 45.116288, 0.000397278008894132],
                                    '宜宾': [104.630825, 28.760189, 0.00138272009069865],
                                    '安阳': [114.352482, 36.103442, 0.00258864716313964],
                                    '郑州': [113.665412, 34.757975, 0.00290855232963196],
                                    '明水': [125.907544, 47.183527, 0.00108553740257613],
                                    '安达': [125.329926, 46.410614, 0.00111358860143404],
                                    '许昌': [113.826063, 34.022956, 0.00256963304035936],
                                    '东海': [115.66, 22.9598, 0.00301423670956416],
                                    '舒兰': [126.947813, 44.410906, 0.000403104626078722],
                                    '周村': [117.851036, 36.803699, 0.00211679718130026],
                                    '新乡': [113.883991, 35.302616, 0.00259101444141437],
                                    '塘坪': [111.863, 21.9775, 0.00302292631104928],
                                    '阳江': [111.975107, 21.859222, 0.00309142001772927],
                                    '石马': [115.874, 24.2873, 0.00220009219514337],
                                    '南京': [118.767413, 32.041544, 0.00289795127813994],
                                    '高邮': [119.443842, 32.785164, 0.00291063120380355],
                                    '驻马店': [114.024736, 32.980169, 0.00251706289524904],
                                    '吉舒': [126.929, 44.1396, 0.000392346152390065],
                                    '临朐': [118.539876, 36.516371, 0.00209839619789669],
                                    '大庆': [125.11272, 46.590734, 0.0011948384766953],
                                    '大同': [113.295259, 40.09031, 0.00152460588367028],
                                    '信宜': [110.941656, 22.352681, 0.00301534221591366],
                                    '仪征': [119.182443, 32.271965, 0.00289911183576159],
                                    '揭阳': [116.355733, 23.543778, 0.00310025672472327],
                                    '汉中': [107.028621, 33.077668, 0.00218930340298133],
                                    '山城': [114.184202, 35.896058, 0.00216263662070775],
                                    '青州': [118.484693, 36.697855, 0.00215164030157291],
                                    '万县': [106.504962, 29.533155, 0.000692079135047648],
                                    '碣石': [115.835, 22.7229, 0.00300454205946506],
                                    '醴陵': [113.507157, 27.657873, 0.00200672319046909],
                                    '萍乡': [113.852186, 27.622946, 0.00202313415285719],
                                    '扬州': [119.421003, 32.393159, 0.00298400391764373],
                                    '烟筒山': [126.016, 43.3515, 0.00039333021588143],
                                    '黄陂': [115.714, 24.4575, 0.00165443543992512],
                                    '合肥': [117.283042, 31.86119, 0.00172017614079191],
                                    '龙凤': [125.145794, 46.573948, 0.00106547240098126],
                                    '禹州': [113.471316, 34.154403, 0.00251759318290571],
                                    '日照': [119.461208, 35.428588, 0.00211348084870256],
                                    '合德': [120.083, 33.7752, 0.00291517703300145],
                                    '襄城': [113.493166, 33.855943, 0.00253042064982395],
                                    '仙居': [114.64, 32.1102, 0.00229998902877697],
                                    '青冈': [126.112268, 46.686596, 0.00106935470373346],
                                    '江都': [119.567481, 32.426564, 0.00293945647978975],
                                    '兴化': [119.840162, 32.938065, 0.00289694439279849],
                                    '株洲': [113.151737, 27.835806, 0.00205255477436773],
                                    '德州': [116.307428, 37.453968, 0.00213266829130013],
                                    '湘潭': [112.944052, 27.82973, 0.00203027457129827],
                                    '潮州': [116.632301, 23.661701, 0.00309219279198761],
                                    '汕头': [116.708463, 23.37102, 0.00338757600505194],
                                    '湘乡': [112.525217, 27.734918, 0.0019740290634954],
                                    '西乡': [107.765858, 32.987961, 0.0024994003550137],
                                    '镇江': [119.452753, 32.204402, 0.00290846295485078],
                                    '宜春': [114.391136, 27.8043, 0.00142418290716384],
                                    '盐城': [120.139998, 33.377631, 0.00296075579227833],
                                    '平顶山': [113.307718, 33.735241, 0.00261012441343297],
                                    '尚志': [127.968539, 45.214953, 0.00109644858247871],
                                    '寿光': [118.736451, 36.874411, 0.00215760673923789],
                                    '信阳': [114.075031, 32.123274, 0.00259040951721602],
                                    '高州': [110.853251, 21.915153, 0.00300853091225123],
                                    '滁州': [118.316264, 32.303627, 0.00153531639625851],
                                    '公主岭': [124.817588, 43.509474, 0.000447527971511824],
                                    '娄底': [112.008497, 27.728136, 0.00196927009396709],
                                    '让胡路': [124.868341, 46.653254, 0.00106381001874852],
                                    '泰州': [119.915176, 32.484882, 0.00294493917358128],
                                    '潍坊': [119.107078, 36.70925, 0.00226077271317139],
                                    '焦作': [113.238266, 35.23904, 0.00255168297793007],
                                    '安丘': [119.206886, 36.427417, 0.00210124086995952],
                                    '宾州': [118.016974, 37.383542, 0.00211594868602848],
                                    '涟源': [111.670847, 27.692301, 0.00204854227733706],
                                    '长沙': [112.982279, 28.19409, 0.00219184075313145],
                                    '绥化': [126.99293, 46.637393, 0.00110138541547313],
                                    '昆阳': [113.362, 33.628, 0.00229020940838998],
                                    '胶南': [116.244, 23.4601, 0.00222686900556712],
                                    '罗城': [111.596, 22.7793, 0.00134973938354191],
                                    '望奎': [126.484191, 46.83352, 0.00106433124164808],
                                    '西安': [129.61311, 44.581032, 0.00279002767139696],
                                    '登封': [113.037768, 34.459939, 0.00253992056044102],
                                    '姜堰': [120.148208, 32.508483, 0.00290001188455545],
                                    '冷水江': [111.434674, 27.685759, 0.00200555652527142],
                                    '东台': [120.314101, 32.853174, 0.00288517101389983],
                                    '浏阳': [113.633301, 28.141112, 0.00198116629694806],
                                    '鱼洞': [105.939, 32.5419, 0.00066940347058373],
                                    '边庄': [105.28501, 27.301693, 0.00210305014550942],
                                    '寒亭': [119.207866, 36.772103, 0.00212030448602812],
                                    '化州': [110.63839, 21.654953, 0.0030271581420062],
                                    '泰兴': [120.020228, 32.168784, 0.00291916114500635],
                                    '天长': [119.011212, 32.6815, 0.00149896333150734],
                                    '青岛': [120.355173, 36.082982, 0.00280559671399332],
                                    '常州': [119.946973, 31.772752, 0.00308148293365849],
                                    '新余': [114.930835, 27.810834, 0.00200342055579792],
                                    '吴川': [110.780508, 21.428453, 0.0030268177108425],
                                    '邯郸': [114.490686, 36.612273, 0.00158662248551666],
                                    '巢湖': [117.874155, 31.600518, 0.0015930647428155],
                                    '石桥': [112.598, 33.2116, 0.00302528179158874],
                                    '东营': [118.66471, 37.434564, 0.00213059649153925],
                                    '杨村': [103.108, 29.2078, 0.00119663098783246],
                                    '廉洲': [110.284961, 21.611281, 0.00305805641737348],
                                    '高密': [119.757033, 36.37754, 0.00210891908314113],
                                    '廉江': [110.284961, 21.611281, 0.00301273461745418],
                                    '磐石': [126.059929, 42.942476, 0.000429249276404885],
                                    '梧州': [111.297604, 23.474803, 0.00085761455162569],
                                    '云阳': [112.688, 33.5861, 0.00255419939216444],
                                    '胶州': [120.006202, 36.285878, 0.00220745704393196],
                                    '湛江': [110.364977, 21.274898, 0.00314401695578718],
                                    '天津': [117.190182, 39.125596, 0.00330033453668758],
                                    '济水': [112.595, 35.0944, 0.00251083698243382],
                                    '泗水': [110.92, 21.8512, 0.00209923727188677],
                                    '南宫': [115.398102, 37.359668, 0.00148214442548049],
                                    '龙泉': [113.474, 33.4962, 0.00162203864003858],
                                    '益阳': [112.355042, 28.570066, 0.00205897331741673],
                                    '雄州': [116.084, 39.0089, 0.00304512185015809],
                                    '无锡': [120.301663, 31.574729, 0.00303478478913589],
                                    '无城': [117.941, 31.4229, 0.0015178349108049],
                                    '沈阳': [123.429096, 41.796767, 0.00188857688558594],
                                    '唐河': [112.838492, 32.687892, 0.00250257930223147],
                                    '辽源': [125.145349, 42.902692, 0.000455977231924903],
                                    '宣化': [113.317, 34.3881, 0.00157129489874944],
                                    '蛟河': [127.342739, 43.720579, 0.000429728300622188],
                                    '南阳': [112.540918, 32.999082, 0.002653022380004],
                                    '邢台': [114.508851, 37.0682, 0.00158218046269789],
                                    '张家港': [120.543441, 31.865553, 0.00290278118478407],
                                    '平度': [119.959012, 36.788828, 0.00213503652261023],
                                    '涪陵': [107.394905, 29.703652, 0.000673768986146833],
                                    '芜湖': [118.376451, 31.326319, 0.00163273586987433],
                                    '临江': [114.674, 23.5894, 0.000497707542619156],
                                    '利川': [108.943491, 30.294247, 0.00165185744947525],
                                    '昌图': [124.11017, 42.784441, 0.000772148054508296],
                                    '八步': [102.93, 29.9651, 0.000814813114654389],
                                    '南通': [120.864608, 32.016212, 0.00294612502696208],
                                    '衡水': [115.665993, 37.735097, 0.00147012001940182],
                                    '金华': [112.66, 32.8239, 0.00233207262758878],
                                    '乐山': [106.596, 27.6336, 0.00136231173345341],
                                    '麻城': [115.02541, 31.177906, 0.00165360409202972],
                                    '绥棱': [127.111121, 47.247195, 0.00106803380790338],
                                    '武汉': [114.298572, 30.584355, 0.0024958831924884],
                                    '桦甸': [126.745445, 42.972093, 0.000420211988665352],
                                    '海州': [121.657639, 42.011162, 0.00287900454624215],
                                    '上海': [121.472644, 31.231706, 0.00401772813141214],
                                    '虢镇': [107.371, 34.3606, 0.00208821067988489],
                                    '沅江': [112.361088, 28.839713, 0.00196919834337737],
                                    '安康': [109.029273, 32.6903, 0.00208546042822494],
                                    '兰溪': [115.174, 30.3245, 0.00231293072172188],
                                    '苏州': [120.619585, 31.299379, 0.00304726113519849],
                                    '洪江': [109.831765, 27.201876, 0.00197484343997566],
                                    '常熟市': [120.74852, 31.658156, 0.00289066916693944],
                                    '泰来': [123.47, 46.4807, 0.00107138258262322],
                                    '即墨': [120.447352, 36.390847, 0.0021254659317418],
                                    '莱州': [119.942135, 37.182725, 0.00212299141397823],
                                    '拜泉': [126.091911, 47.607363, 0.00106576767044422],
                                    '铜陵': [117.816576, 30.929935, 0.00157455843640684],
                                    '怀化': [109.97824, 27.550082, 0.00199389987793435],
                                    '岳阳': [113.132855, 29.37029, 0.00229998778219839],
                                    '海门': [121.176609, 31.893528, 0.00300338222450673],
                                    '新野': [112.365624, 32.524006, 0.00249910129946646],
                                    '齐齐哈尔': [123.95792, 47.342081, 0.00122126061185103],
                                    '剑光': [115.796, 28.1973, 0.00197469304434048],
                                    '昭通': [103.717216, 27.336999, 0.00155476142764307],
                                    '漳州': [117.661801, 24.510897, 0.00221541246404181],
                                    '松陵': [120.641601, 31.160404, 0.00289044032381948],
                                    '辉南': [126.042821, 42.683459, 0.000397322519887974],
                                    '义马': [111.869417, 34.746868, 0.0025012527870797],
                                    '厦门': [118.11022, 24.490474, 0.00262475967333443],
                                    '黑山': [127.637, 46.5429, 0.000769477963585977],
                                    '开原': [124.045551, 42.542141, 0.000773905508782879],
                                    '常德': [111.691347, 29.040225, 0.00207365612911868],
                                    '东丰': [125.529623, 42.675228, 0.000394415719896462],
                                    '邓州': [112.092716, 32.681642, 0.00255677057993518],
                                    '石家庄': [114.502461, 38.045474, 0.0018498725552546],
                                    '广水': [113.826601, 31.617731, 0.00166692568381029],
                                    '辛集': [112.959, 33.7495, 0.0014626230240958],
                                    '铁力': [128.030561, 46.985772, 0.00109184736013698],
                                    '池州': [117.489157, 30.656037, 0.00153003050147873],
                                    '三明': [117.635001, 26.265444, 0.00220258897706961],
                                    '沧州': [116.857461, 38.310582, 0.00150001855267931],
                                    '桂林': [113.793, 35.9527, 0.00103992579011792],
                                    '安庆': [117.090906, 30.526611, 0.00152600060129478],
                                    '招远': [120.403142, 37.364919, 0.00218117592334005],
                                    '抚顺': [123.921109, 41.875956, 0.0010547909438202],
                                    '恩施': [109.486761, 30.282406, 0.00165317817275399],
                                    '北京': [116.405285, 39.904989, 0.00286554379546414],
                                    '余下': [108.595, 34.0747, 0.00208547138523622],
                                    '梅河口': [125.687336, 42.530002, 0.000403080003976234],
                                    '南昌': [115.892151, 28.676493, 0.00206315070177382],
                                    '莱阳': [120.711151, 36.977037, 0.00214760957086679],
                                    '新石': [114.469, 38.0048, 0.00165746430449718],
                                    '南洲': [112.387, 29.3767, 0.00196667215364898],
                                    '湖州': [120.102398, 30.867198, 0.0023131344965246],
                                    '富拉尔基区': [123.638873, 47.20697, 0.00108083904254232],
                                    '启东': [121.659724, 31.810158, 0.00293380549335591],
                                    '栾城': [114.654281, 37.886911, 0.00145828066193031],
                                    '随州': [113.37377, 31.717497, 0.00168033691155404],
                                    '邵武': [117.491544, 27.337952, 0.00224700072061933],
                                    '宣州': [118.758412, 30.946003, 0.00151217511527563],
                                    '咸阳': [108.705117, 34.333439, 0.00218214644339508],
                                    '保定': [115.482331, 38.867657, 0.00184840687306875],
                                    '铁岭': [123.844279, 42.290585, 0.000803998532148284],
                                    '荔城': [119.020047, 25.430047, 0.00301432492237039],
                                    '鄂州': [114.890593, 30.396536, 0.00168829472798342],
                                    '黄州': [114.878934, 30.447435, 0.00165528050768641],
                                    '杭州': [120.153576, 30.287459, 0.00251767719351971],
                                    '黄梅': [115.942548, 30.075113, 0.00166465194488465],
                                    '毕节': [105.28501, 27.301693, 0.000985215016580028],
                                    '泉州': [118.589421, 24.908853, 0.00246979738976303],
                                    '海林': [129.387902, 44.574149, 0.00107990987840509],
                                    '孝感': [113.926655, 30.926423, 0.00167946193366182],
                                    '溪美': [118.321, 24.9432, 0.0022128149125737],
                                    '栖霞': [120.834097, 37.305854, 0.00213413526844536],
                                    '吉首': [109.738273, 28.314827, 0.00198091367278361],
                                    '东村': [121.154, 36.8232, 0.00212443218236759],
                                    '津市': [111.879609, 29.630867, 0.00196968178140694],
                                    '枣阳': [112.765268, 32.123083, 0.00167186708166947],
                                    '牡丹江': [129.618602, 44.582962, 0.00116704070743956],
                                    '任丘': [116.106764, 38.706513, 0.00147729462245251],
                                    '晋江': [118.577338, 24.807322, 0.000690199972833809],
                                    '南平': [118.178459, 26.635627, 0.00222207360035973],
                                    '景德镇': [117.214664, 29.29256, 0.00211503717627992],
                                    '嘉兴': [120.750865, 30.762653, 0.00232973543076468],
                                    '西昌': [102.258758, 27.885786, 0.00135608811183329],
                                    '富阳': [119.949869, 30.049871, 0.00231044391972299],
                                    '应城': [113.573842, 30.939038, 0.00167037264200459],
                                    '定州': [114.991389, 38.517602, 0.00147302208106492],
                                    '大冶': [114.974842, 30.098804, 0.00165231322194743],
                                    '蓬莱': [120.762689, 37.811168, 0.00212008635979927],
                                    '蔡甸': [114.029341, 30.582186, 0.00165482511601122],
                                    '武穴': [115.56242, 29.849342, 0.00165495274672156],
                                    '嘉善': [120.921871, 30.841352, 0.0022924370354452],
                                    '库尔勒': [86.145948, 41.763122, 0.000855717528229769],
                                    '汉川': [113.835301, 30.652165, 0.00165394178751417],
                                    '桓仁': [125.393, 41.301, 0.000788022570884836],
                                    '晋城': [112.851274, 35.497553, 0.00112481922019461],
                                    '贵溪': [117.212103, 28.283693, 0.00198405440886802],
                                    '桂平': [110.074668, 23.382473, 0.000847097391921851],
                                    '长治': [113.113556, 36.191112, 0.00111482843408392],
                                    '枝城': [111.51, 30.2963, 0.00168278217425632],
                                    '蒲圻': [113.812, 29.6809, 0.00165800241383895],
                                    '河坡': [98.8763, 31.3646, 0.00300795270681674],
                                    '浦阳': [120.217, 29.9677, 0.00229940935435806],
                                    '渭南': [109.502882, 34.499381, 0.00221691467990265],
                                    '烟台': [121.391382, 37.539297, 0.0022390701733036],
                                    '铜川': [108.979608, 34.916582, 0.00212163989294359],
                                    '张家口': [120.593, 31.8121, 0.00153182042784653],
                                    '襄樊': [112.144146, 32.042426, 0.00180980787260427],
                                    '鄱阳': [116.673748, 28.993374, 0.00199545539105207],
                                    '朗乡': [128.889, 47.0858, 0.00107776803901216],
                                    '莆田': [119.007558, 25.431011, 0.00230722667261652],
                                    '讷河': [124.882172, 48.481133, 0.0010716832149663],
                                    '绍兴': [120.582112, 29.997117, 0.00232589841764843],
                                    '衢州': [118.87263, 28.941708, 0.00229575268328921],
                                    '诸暨': [120.244326, 29.713662, 0.00229564078379405],
                                    '甘南': [123.506034, 47.917838, 0.00106871594249233],
                                    '依兰': [129.565594, 46.315105, 0.00110563506815502],
                                    '咸宁': [114.328963, 29.832798, 0.00165680269925642],
                                    '罗容': [109.91, 23.3666, 0.000816204362485144],
                                    '仙桃': [113.453974, 30.364953, 0.0016784080591398],
                                    '上虞': [120.874185, 30.016769, 0.0023060678482498],
                                    '义乌': [120.074911, 29.306863, 0.00234875642549727],
                                    '宜城': [112.261441, 31.709203, 0.0016660893574105],
                                    '虎石台': [123.443, 41.9644, 0.000768819259261081],
                                    '福州': [119.306239, 26.075302, 0.00239204939665264],
                                    '威宁': [104.286523, 26.859099, 0.000981424156285278],
                                    '敦化': [128.22986, 43.366921, 0.000446973976544739],
                                    '钟祥': [112.587267, 31.165573, 0.00166932618033303],
                                    '郑家屯': [123.502, 43.5002, 0.000393549414768326],
                                    '黄山': [118.317325, 29.709239, 0.00152296617507912],
                                    '太原': [112.549248, 37.857014, 0.00172882063449221],
                                    '文登': [122.057139, 37.196211, 0.00216821653281793],
                                    '玉林': [110.154393, 22.63136, 0.000844728078910418],
                                    '南岔': [129.28246, 47.137314, 0.00109977372780394],
                                    '本溪': [123.770519, 41.297909, 0.000883866138495271],
                                    '唐山': [118.175393, 39.635113, 0.00171457168288073],
                                    '廊坊': [116.704441, 39.523927, 0.00151166317302752],
                                    '余姚': [121.156294, 30.045404, 0.00230958768711121],
                                    '浦城': [118.536822, 27.920412, 0.00221301166200969],
                                    '塘沽': [117.190182, 39.125596, 0.00128199327195272],
                                    '明月': [113.449, 27.5026, 0.00039673021290246],
                                    '老河口': [111.675732, 32.385438, 0.00167452109653629],
                                    '埠河': [112.288, 30.235, 0.00166893996391469],
                                    '咸水沽': [117.367, 39.0169, 0.00120758607540459],
                                    '开通': [122.954, 44.7272, 0.00045167113412786],
                                    '南渡': [119.287, 31.4714, 0.000857909302252344],
                                    '丹江口': [111.513793, 32.538839, 0.00165900427471438],
                                    '荆州': [112.23813, 30.326857, 0.001671691048349],
                                    '威海': [122.116394, 37.509691, 0.00212957203325395],
                                    '杨柳青': [116.935, 39.1211, 0.0011965749047376],
                                    '沙市': [112.257433, 30.315895, 0.00165539057370414],
                                    '贵阳': [106.713478, 26.578343, 0.00102514123061312],
                                    '友好': [128.838961, 47.854303, 0.00107331450021487],
                                    '忻州': [112.733538, 38.41769, 0.00108602693968338],
                                    '曲靖': [103.797851, 25.501557, 0.00158450499224587],
                                    '遵义': [106.937265, 27.706626, 0.00061471216953376],
                                    '林口': [130.268402, 45.286645, 0.00108764427936897],
                                    '通辽': [122.263119, 43.617429, 0.000717344535963908],
                                    '荆门': [112.204251, 31.03542, 0.00167182956117269],
                                    '苏家屯': [123.341604, 41.665904, 0.000769040900904032],
                                    '韩城': [110.452391, 35.475238, 0.00208845106970634],
                                    '宁波': [121.549792, 29.868388, 0.0023410640220855],
                                    '阳泉': [113.583285, 37.861188, 0.00110206110174523],
                                    '宝山': [123.76, 48.6949, 0.00108083903243987],
                                    '开化': [118.414435, 29.136503, 0.00159981334603004],
                                    '朝阳': [112.52, 34.8121, 0.000880604116065724],
                                    '枝江': [111.751799, 30.425364, 0.00165723894901942],
                                    '镇赉': [123.255, 45.7841, 0.000397957404982332],
                                    '二道江': [126.045987, 41.777564, 0.000396654390546519],
                                    '嫩江': [125.229904, 49.177461, 0.00106279829998621],
                                    '汉沽': [117.83, 39.3031, 0.00123207957597974],
                                    '丽水': [119.921786, 28.451993, 0.00229574592672885],
                                    '新民': [122.828868, 41.996508, 0.000768800647866614],
                                    '来宾': [109.229772, 23.733766, 0.000833424300640382],
                                    '安顺': [105.932188, 26.245544, 0.000988952066054636],
                                    '宜昌': [111.290843, 30.702636, 0.00171088411181502],
                                    '昆明': [102.712251, 25.040609, 0.00190193721627161],
                                    '宁海': [121.432606, 29.299836, 0.00219849523063687],
                                    '宽甸': [124.779, 40.7078, 0.000777585047548678],
                                    '松江河': [127.525, 42.2587, 0.000395641241565755],
                                    '佳木斯': [130.361634, 46.809606, 0.00120787938022994],
                                    '唐家庄': [118.457, 39.7533, 0.00147331400534182],
                                    '八面通': [130.55, 45.0985, 0.00106557476538857],
                                    '丰润': [118.155779, 39.831363, 0.00145794778615711],
                                    '洮南': [122.783779, 45.339113, 0.000453672147757166],
                                    '阿克苏': [88.6154, 43.3497, 0.000875339819824752],
                                    '南宁': [108.320004, 22.82402, 0.00126657350735392],
                                    '临海': [121.131229, 28.845441, 0.00230038607422484],
                                    '辽阳': [123.18152, 41.269402, 0.00079733206019399],
                                    '州城': [100.582, 25.7627, 0.00210354392399416],
                                    '双阳': [125.643, 47.5854, 0.00039899067536568],
                                    '中枢': [103.66, 25.0695, 0.00151268024938924],
                                    '昌黎': [119.164541, 39.709729, 0.00146660997008591],
                                    '鞍山': [122.995632, 41.110626, 0.00084353894236836],
                                    '桦南': [130.570112, 46.240118, 0.00107066248986605],
                                    '黄岩': [121.262138, 28.64488, 0.0022961946760669],
                                    '铜仁': [109.191555, 27.718346, 0.000978773375810231],
                                    '榆次': [112.740056, 37.6976, 0.00108055131798671],
                                    '鸡西': [130.975966, 45.300046, 0.00112639889532838],
                                    '秦皇岛': [119.586579, 39.942531, 0.00159978025403901],
                                    '温州': [120.672111, 28.000575, 0.00237893735988108],
                                    '椒江': [121.431049, 28.67615, 0.00230458903608113],
                                    '滴道': [130.846823, 45.348812, 0.00106929490578614],
                                    '恒山': [130.910636, 45.213242, 0.00108797385071104],
                                    '白城': [122.841114, 45.619026, 0.000405192452013282],
                                    '七台河': [131.015584, 45.771266, 0.00117104948385197],
                                    '辽中': [122.731269, 41.512725, 0.000769971107256497],
                                    '鹤岗': [130.277487, 47.332085, 0.00111778371098058],
                                    '都匀': [107.517021, 26.258205, 0.000983854984870112],
                                    '小围寨': [107.523, 26.2462, 0.000975636232872696],
                                    '凤城': [124.071067, 40.457567, 0.000791840753126132],
                                    '温岭': [121.373611, 28.368781, 0.00231098362848693],
                                    '良乡': [116.152, 39.7125, 0.00107751881154648],
                                    '丹东': [124.383044, 40.124296, 0.000841982225311789],
                                    '鸡东': [131.148907, 45.250892, 0.00110694436560203],
                                    '平遥': [112.174059, 37.195474, 0.00106308829740392],
                                    '平南': [109.165, 22.5313, 0.000819309845014913],
                                    '临汾': [111.517973, 36.08415, 0.00110215860214564],
                                    '龙井': [114.185, 32.3873, 0.000392659692097282],
                                    '东宁': [131.125296, 44.063578, 0.00106587883347635],
                                    '绥芬河': [131.164856, 44.396864, 0.00106502450153514],
                                    '辛置': [111.738, 36.5474, 0.00106743662452801],
                                    '连然': [102.474, 24.9443, 0.00151591290879073],
                                    '通州': [116.658603, 39.902486, 0.00107650775780834],
                                    '弥阳': [103.536, 24.4624, 0.00152991281661525],
                                    '门头沟': [116.105381, 39.937183, 0.0010925484626792],
                                    '平凉': [106.684691, 35.54279, 0.000894769381777107],
                                    '南台': [122.826, 40.9537, 0.000769355226508626],
                                    '海口': [110.33119, 20.031971, 0.00109489359868373],
                                    '榆林': [109.741193, 38.290162, 0.00216648440193028],
                                    '上梅': [107.105, 38.1345, 0.00198604379740869],
                                    '运城': [111.003957, 35.022778, 0.00111065273461943],
                                    '顺义': [116.653525, 40.128936, 0.00107794950705214],
                                    '承德': [117.939152, 40.976204, 0.00146774580115984],
                                    '海城': [122.752199, 40.852533, 0.000771923022405517],
                                    '昌吉': [87.304112, 44.013183, 0.000852786869145004],
                                    '双鸭山': [131.157304, 46.643442, 0.00107984115537193],
                                    '昌平': [107.809, 22.7578, 0.00109200680911534],
                                    '廉州镇': [109.279, 21.7247, 0.000822578176224208],
                                    '蒲庙': [108.591, 22.7884, 0.000831063081174483],
                                    '岭东': [131.163675, 46.591076, 0.00106591299350059],
                                    '玉溪': [102.543907, 24.350461, 0.00151171207589303],
                                    '原平': [112.713132, 38.729186, 0.00106008957390877],
                                    '钦州': [108.624175, 21.967127, 0.000911404455379834],
                                    '岫岩': [123.28833, 40.281509, 0.000773992831710169],
                                    '兰州': [103.823557, 36.058039, 0.000942235126627454],
                                    '白山': [123.289, 47.288, 0.000411897151964738],
                                    '营口': [122.235151, 40.667432, 0.000823467759224721],
                                    '密山': [131.874137, 45.54725, 0.00110071898949556],
                                    '开远': [103.258679, 23.713832, 0.00155220634120907],
                                    '北海': [109.119254, 21.473343, 0.000815041038971238],
                                    '大理': [100.241369, 25.593067, 0.00151583275994914],
                                    '延吉': [129.51579, 42.906964, 0.000472198133890314],
                                    '和龙': [129.008748, 42.547004, 0.000393632980217229],
                                    '石河子': [86.041075, 44.305886, 0.000859500348055783],
                                    '白银': [104.173606, 36.54568, 0.000877436248986916],
                                    '肃州': [94.6591, 40.1522, 0.00106999853890081],
                                    '庄河': [122.970612, 39.69829, 0.000769513493805021],
                                    '大连': [121.618622, 38.91459, 0.00105331503855745],
                                    '富锦': [132.037951, 47.250747, 0.00107946901892749],
                                    '北票': [120.766951, 41.803286, 0.00077181185686373],
                                    '汪清': [129.766161, 43.315426, 0.000399852465544159],
                                    '图们': [129.846701, 42.966621, 0.000406058068334921],
                                    '六合': [123.878, 48.5022, 0.000400891344281141],
                                    '南票': [120.752314, 41.098813, 0.000768924572564683],
                                    '瓦房店': [122.002656, 39.63065, 0.000770887261347223],
                                    '加格达奇': [124.126716, 50.424654, 0.000706384472767167],
                                    '百色': [106.616285, 23.897742, 0.000824736195418373],
                                    '兴城': [120.729365, 40.619413, 0.000771630731822257],
                                    '普兰店': [121.9705, 39.401555, 0.000769467198274491],
                                    '阿里河': [124.024, 50.5364, 0.000702751687450928],
                                    '塔河': [124.710516, 52.335229, 0.00106323789197997],
                                    '梨树': [130.697781, 45.092195, 0.000413552745430181],
                                    '琼山': [110.354722, 20.001051, 0.000544747592162986],
                                    '西宁': [101.778916, 36.623178, 0.00081070065300391],
                                    '大连湾': [121.65, 39.0825, 0.000770109258825781],
                                    '珲春': [130.365787, 42.871057, 0.000394480302991732],
                                    '集宁': [113.116453, 41.034134, 0.000896239266343545],
                                    '叶柏寿': [119.638, 41.4098, 0.000769253342482851],
                                    '景洪': [100.797947, 22.002087, 0.00152467994387025],
                                    '牙克石': [120.729005, 49.287024, 0.000748358179138992],
                                    '凌源': [119.404789, 41.243086, 0.000773573132592787],
                                    '丰镇': [113.163462, 40.437534, 0.000708035314644392],
                                    '平庄': [119.242, 42.0153, 0.000706393667879878],
                                    '赤峰': [118.895357, 42.258931, 0.000713558942493077],
                                    '四平': [122.115, 39.7925, 0.000475376477620683],
                                    '西丰': [133.239, 46.9748, 0.000777282699373933],
                                    '金昌': [102.187888, 38.514238, 0.000884091081021978],
                                    '海拉尔': [119.764923, 49.213889, 0.000717820024165826],
                                    '和田': [79.927542, 37.108944, 0.000852389441296042],
                                    '呼和浩特': [111.670801, 40.818311, 0.000746211053993616],
                                    '三亚': [109.508268, 18.247872, 0.000582977533004751],
                                    '东胜': [109.98945, 39.81788, 0.000714448401872473],
                                    '张掖': [100.455472, 38.932897, 0.000882240601161731],
                                    '包头': [109.840405, 40.658168, 0.000782555483578635],
                                    '乌达': [106.722711, 39.502288, 0.000708983438782051],
                                    '乌海': [106.825563, 39.673734, 0.000720839402530383],
                                    '嘉峪关': [98.277304, 39.786529, 0.000890570073835979],
                                    '锡林浩特': [116.091903, 43.944301, 0.000709435578851009],
                                    '满洲里': [117.455561, 49.590788, 0.000712951412547668],
                                    '莎车': [77.248884, 38.414499, 0.000865584313530761],
                                    '那曲': [92.060214, 31.476004, 0.000346527939447057],
                                    '银川': [133.972, 48.0559, 0.000320841928531706],
                                    '石嘴山': [106.376173, 39.01333, 0.00027889383214054],
                                    '石炭井': [106.32, 39.06, 0.000256989860167995],
                                    '拉萨': [91.132212, 29.660361, 0.000346809408917844]}
        qssStyle = '''
 QPalette{background:#EAF7FF;}*{outline:0px;color:#386487;}
QWidget[form="true"],QLabel[frameShape="1"]{
border:1px solid #C0DCF2;
border-radius:0px;
}
QWidget[form="bottom"]{
background:#DEF0FE;
}
QWidget[form="bottom"] .QFrame{
border:1px solid #386487;
}
QWidget[form="bottom"] QLabel,QWidget[form="title"] QLabel{
border-radius:0px;
color:#386487;
background:none;
border-style:none;
}
QWidget[form="title"],QWidget[nav="left"],QWidget[nav="top"] QAbstractButton{
border-style:none;
border-radius:0px;
padding:5px;
color:#386487;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}
QWidget[nav="top"] QAbstractButton:hover,QWidget[nav="top"] QAbstractButton:pressed,QWidget[nav="top"] QAbstractButton:checked{
border-style:solid;
border-width:0px 0px 2px 0px;
padding:4px 4px 2px 4px;
border-color:#00BB9E;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}
QWidget[nav="left"] QAbstractButton{
border-radius:0px;
color:#386487;
background:none;
border-style:none;
}
QWidget[nav="left"] QAbstractButton:hover{
color:#FFFFFF;
background-color:#00BB9E;
}
QWidget[nav="left"] QAbstractButton:checked,QWidget[nav="left"] QAbstractButton:pressed{
color:#386487;
border-style:solid;
border-width:0px 0px 0px 2px;
padding:4px 4px 4px 2px;
border-color:#00BB9E;
background-color:#EAF7FF;
}
QWidget[video="true"] QLabel{
color:#386487;
border:1px solid #C0DCF2;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}
QWidget[video="true"] QLabel:focus{
border:1px solid #00BB9E;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}
QLineEdit,QTextEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox,QDateEdit,QTimeEdit,QDateTimeEdit{
border:1px solid #C0DCF2;
border-radius:3px;
padding:2px;
background:none;
selection-background-color:#00BB9E;
selection-color:#FFFFFF;
}
QLineEdit:focus,QTextEdit:focus,QPlainTextEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus,QDateEdit:focus,QTimeEdit:focus,QDateTimeEdit:focus,QLineEdit:hover,QTextEdit:hover,QPlainTextEdit:hover,QSpinBox:hover,QDoubleSpinBox:hover,QComboBox:hover,QDateEdit:hover,QTimeEdit:hover,QDateTimeEdit:hover{
border:1px solid #C0DCF2;
}
QLineEdit[echoMode="2"]{
lineedit-password-character:9679;
}
.QFrame{
border:1px solid #C0DCF2;
border-radius:3px;
}
.QGroupBox{
border:1px solid #C0DCF2;
border-radius:5px;
margin-top:3ex;
}
.QGroupBox::title{
subcontrol-origin:margin;
position:relative;
left:10px;
}
.QPushButton,.QToolButton{
border-style:none;
border:1px solid #C0DCF2;
color:#386487;
padding:5px;
min-height:15px;
border-radius:5px;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}
.QPushButton:hover,.QToolButton:hover{
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}
.QPushButton:pressed,.QToolButton:pressed{
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}
.QToolButton::menu-indicator{
image:None;
}
QToolButton#btnMenu,QPushButton#btnMenu_Min,QPushButton#btnMenu_Max,QPushButton#btnMenu_Close{
border-radius:3px;
color:#386487;
padding:3px;
margin:0px;
background:none;
border-style:none;
}
QToolButton#btnMenu:hover,QPushButton#btnMenu_Min:hover,QPushButton#btnMenu_Max:hover{
color:#FFFFFF;
margin:1px 1px 2px 1px;
background-color:rgba(51,127,209,230);
}
QPushButton#btnMenu_Close:hover{
color:#FFFFFF;
margin:1px 1px 2px 1px;
background-color:rgba(238,0,0,128);
}
QRadioButton::indicator{
width:15px;
height:15px;
}
QRadioButton::indicator::unchecked{
image:url(:/qss/lightblue/radiobutton_unchecked.png);
}
QRadioButton::indicator::unchecked:disabled{
image:url(:/qss/lightblue/radiobutton_unchecked_disable.png);
}
QRadioButton::indicator::checked{
image:url(:/qss/lightblue/radiobutton_checked.png);
}
QRadioButton::indicator::checked:disabled{
image:url(:/qss/lightblue/radiobutton_checked_disable.png);
}
QGroupBox::indicator,QTreeWidget::indicator,QListWidget::indicator{
padding:0px -3px 0px 0px;
}
QCheckBox::indicator,QGroupBox::indicator,QTreeWidget::indicator,QListWidget::indicator{
width:13px;
height:13px;
}
QCheckBox::indicator:unchecked,QGroupBox::indicator:unchecked,QTreeWidget::indicator:unchecked,QListWidget::indicator:unchecked{
image:url(:/qss/lightblue/checkbox_unchecked.png);
}
QCheckBox::indicator:unchecked:disabled,QGroupBox::indicator:unchecked:disabled,QTreeWidget::indicator:unchecked:disabled,QListWidget::indicator:disabled{
image:url(:/qss/lightblue/checkbox_unchecked_disable.png);
}
QCheckBox::indicator:checked,QGroupBox::indicator:checked,QTreeWidget::indicator:checked,QListWidget::indicator:checked{
image:url(:/qss/lightblue/checkbox_checked.png);
}
QCheckBox::indicator:checked:disabled,QGroupBox::indicator:checked:disabled,QTreeWidget::indicator:checked:disabled,QListWidget::indicator:checked:disabled{
image:url(:/qss/lightblue/checkbox_checked_disable.png);
}
QCheckBox::indicator:indeterminate,QGroupBox::indicator:indeterminate,QTreeWidget::indicator:indeterminate,QListWidget::indicator:indeterminate{
image:url(:/qss/lightblue/checkbox_parcial.png);
}
QCheckBox::indicator:indeterminate:disabled,QGroupBox::indicator:indeterminate:disabled,QTreeWidget::indicator:indeterminate:disabled,QListWidget::indicator:indeterminate:disabled{
image:url(:/qss/lightblue/checkbox_parcial_disable.png);
}
QTimeEdit::up-button,QDateEdit::up-button,QDateTimeEdit::up-button,QDoubleSpinBox::up-button,QSpinBox::up-button{
image:url(:/qss/lightblue/add_top.png);
width:10px;
height:10px;
padding:2px 5px 0px 0px;
}
QTimeEdit::down-button,QDateEdit::down-button,QDateTimeEdit::down-button,QDoubleSpinBox::down-button,QSpinBox::down-button{
image:url(:/qss/lightblue/add_bottom.png);
width:10px;
height:10px;
padding:0px 5px 2px 0px;
}
QTimeEdit::up-button:pressed,QDateEdit::up-button:pressed,QDateTimeEdit::up-button:pressed,QDoubleSpinBox::up-button:pressed,QSpinBox::up-button:pressed{
top:-2px;
}
QTimeEdit::down-button:pressed,QDateEdit::down-button:pressed,QDateTimeEdit::down-button:pressed,QDoubleSpinBox::down-button:pressed,QSpinBox::down-button:pressed,QSpinBox::down-button:pressed{
bottom:-2px;
}
QComboBox::down-arrow,QDateEdit[calendarPopup="true"]::down-arrow,QTimeEdit[calendarPopup="true"]::down-arrow,QDateTimeEdit[calendarPopup="true"]::down-arrow{
image:url(:/qss/lightblue/add_bottom.png);
width:10px;
height:10px;
right:2px;
}
QComboBox::drop-down,QDateEdit::drop-down,QTimeEdit::drop-down,QDateTimeEdit::drop-down{
subcontrol-origin:padding;
subcontrol-position:top right;
width:15px;
border-left-width:0px;
border-left-style:solid;
border-top-right-radius:3px;
border-bottom-right-radius:3px;
border-left-color:#C0DCF2;
}
QComboBox::drop-down:on{
top:1px;
}
QMenuBar::item{
color:#386487;
background-color:#DEF0FE;
margin:0px;
padding:3px 10px;
}
QMenu,QMenuBar,QMenu:disabled,QMenuBar:disabled{
color:#386487;
background-color:#DEF0FE;
border:1px solid #C0DCF2;
margin:0px;
}
QMenu::item{
padding:3px 20px;
}
QMenu::indicator{
width:13px;
height:13px;
}
QMenu::item:selected,QMenuBar::item:selected{
color:#386487;
border:0px solid #C0DCF2;
background:#F2F9FF;
}
QMenu::separator{
height:1px;
background:#C0DCF2;
}
QProgressBar{
min-height:10px;
background:#DEF0FE;
border-radius:5px;
text-align:center;
border:1px solid #DEF0FE;
}
QProgressBar:chunk{
border-radius:5px;
background-color:#C0DCF2;
}
QSlider::groove:horizontal{
background:#DEF0FE;
height:8px;
border-radius:4px;
}
QSlider::add-page:horizontal{
background:#DEF0FE;
height:8px;
border-radius:4px;
}
QSlider::sub-page:horizontal{
background:#C0DCF2;
height:8px;
border-radius:4px;
}
QSlider::handle:horizontal{
width:13px;
margin-top:-3px;
margin-bottom:-3px;
border-radius:6px;
background:qradialgradient(spread:pad,cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,stop:0.6 #EAF7FF,stop:0.8 #C0DCF2);
}
QSlider::groove:vertical{
width:8px;
border-radius:4px;
background:#DEF0FE;
}
QSlider::add-page:vertical{
width:8px;
border-radius:4px;
background:#DEF0FE;
}
QSlider::sub-page:vertical{
width:8px;
border-radius:4px;
background:#C0DCF2;
}
QSlider::handle:vertical{
height:14px;
margin-left:-3px;
margin-right:-3px;
border-radius:6px;
background:qradialgradient(spread:pad,cx:0.5,cy:0.5,radius:0.5,fx:0.5,fy:0.5,stop:0.6 #EAF7FF,stop:0.8 #C0DCF2);
}
QScrollBar:horizontal{
background:#DEF0FE;
padding:0px;
border-radius:6px;
max-height:12px;
}
QScrollBar::handle:horizontal{
background:#C0DCF2;
min-width:50px;
border-radius:6px;
}
QScrollBar::handle:horizontal:hover{
background:#00BB9E;
}
QScrollBar::handle:horizontal:pressed{
background:#00BB9E;
}
QScrollBar::add-page:horizontal{
background:none;
}
QScrollBar::sub-page:horizontal{
background:none;
}
QScrollBar::add-line:horizontal{
background:none;
}
QScrollBar::sub-line:horizontal{
background:none;
}
QScrollBar:vertical{
background:#DEF0FE;
padding:0px;
border-radius:6px;
max-width:12px;
}
QScrollBar::handle:vertical{
background:#C0DCF2;
min-height:50px;
border-radius:6px;
}
QScrollBar::handle:vertical:hover{
background:#00BB9E;
}
QScrollBar::handle:vertical:pressed{
background:#00BB9E;
}
QScrollBar::add-page:vertical{
background:none;
}
QScrollBar::sub-page:vertical{
background:none;
}
QScrollBar::add-line:vertical{
background:none;
}
QScrollBar::sub-line:vertical{
background:none;
}
QScrollArea{
border:0px;
}
QTreeView,QListView,QTableView,QTabWidget::pane{
border:1px solid #C0DCF2;
selection-background-color:#F2F9FF;
selection-color:#386487;
alternate-background-color:#DAEFFF;
gridline-color:#C0DCF2;
}
QTreeView::branch:closed:has-children{
margin:4px;
border-image:url(:/qss/lightblue/branch_open.png);
}
QTreeView::branch:open:has-children{
margin:4px;
border-image:url(:/qss/lightblue/branch_close.png);
}
QTreeView,QListView,QTableView,QSplitter::handle,QTreeView::branch{
background:#EAF7FF;
}
QTableView::item:selected,QListView::item:selected,QTreeView::item:selected{
color:#386487;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}
QTableView::item:hover,QListView::item:hover,QTreeView::item:hover,QHeaderView{
color:#386487;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}
QTableView::item,QListView::item,QTreeView::item{
padding:1px;
margin:0px;
}
QHeaderView::section,QTableCornerButton:section{
padding:3px;
margin:0px;
color:#386487;
border:1px solid #C0DCF2;
border-left-width:0px;
border-right-width:1px;
border-top-width:0px;
border-bottom-width:1px;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}
QTabBar::tab{
border:1px solid #C0DCF2;
color:#386487;
margin:0px;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}
QTabBar::tab:selected,QTabBar::tab:hover{
border-style:solid;
border-color:#00BB9E;
background:#EAF7FF;
}
QTabBar::tab:top,QTabBar::tab:bottom{
padding:3px 8px 3px 8px;
}
QTabBar::tab:left,QTabBar::tab:right{
padding:8px 3px 8px 3px;
}
QTabBar::tab:top:selected,QTabBar::tab:top:hover{
border-width:2px 0px 0px 0px;
}
QTabBar::tab:right:selected,QTabBar::tab:right:hover{
border-width:0px 0px 0px 2px;
}
QTabBar::tab:bottom:selected,QTabBar::tab:bottom:hover{
border-width:0px 0px 2px 0px;
}
QTabBar::tab:left:selected,QTabBar::tab:left:hover{
border-width:0px 2px 0px 0px;
}
QTabBar::tab:first:top:selected,QTabBar::tab:first:top:hover,QTabBar::tab:first:bottom:selected,QTabBar::tab:first:bottom:hover{
border-left-width:1px;
border-left-color:#C0DCF2;
}
QTabBar::tab:first:left:selected,QTabBar::tab:first:left:hover,QTabBar::tab:first:right:selected,QTabBar::tab:first:right:hover{
border-top-width:1px;
border-top-color:#C0DCF2;
}
QTabBar::tab:last:top:selected,QTabBar::tab:last:top:hover,QTabBar::tab:last:bottom:selected,QTabBar::tab:last:bottom:hover{
border-right-width:1px;
border-right-color:#C0DCF2;
}
QTabBar::tab:last:left:selected,QTabBar::tab:last:left:hover,QTabBar::tab:last:right:selected,QTabBar::tab:last:right:hover{
border-bottom-width:1px;
border-bottom-color:#C0DCF2;
}
QStatusBar::item{
border:0px solid #DEF0FE;
border-radius:3px;
}
QToolBox::tab,QGroupBox#gboxDevicePanel,QGroupBox#gboxDeviceTitle,QFrame#gboxDevicePanel,QFrame#gboxDeviceTitle{
padding:3px;
border-radius:5px;
color:#386487;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}
QToolTip{
border:0px solid #386487;
padding:1px;
color:#386487;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}
QToolBox::tab:selected{
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #F2F9FF,stop:1 #DAEFFF);
}
QPrintPreviewDialog QToolButton{
border:0px solid #386487;
border-radius:0px;
margin:0px;
padding:3px;
background:none;
}
QColorDialog QPushButton,QFileDialog QPushButton{
min-width:80px;
}
QToolButton#qt_calendar_prevmonth{
icon-size:0px;
min-width:20px;
image:url(:/qss/lightblue/calendar_prevmonth.png);
}
QToolButton#qt_calendar_nextmonth{
icon-size:0px;
min-width:20px;
image:url(:/qss/lightblue/calendar_nextmonth.png);
}
QToolButton#qt_calendar_prevmonth,QToolButton#qt_calendar_nextmonth,QToolButton#qt_calendar_monthbutton,QToolButton#qt_calendar_yearbutton{
border:0px solid #386487;
border-radius:3px;
margin:3px 3px 3px 3px;
padding:3px;
background:none;
}
QoolButton#qt_calendar_prevmonth:hover,QToolButton#qt_calendar_nextmonth:hover,QToolButton#qt_calendar_monthbutton:hover,QToolButton#qt_calendar_yearbutton:hover,QToolButton#qt_calendar_prevmonth:pressed,QToolButton#qt_calendar_nextmonth:pressed,QToolButton#qt_calendar_monthbutton:pressed,QToolButton#qt_calendar_yearbutton:pressed{
border:1px solid #C0DCF2;
}
QCalendarWidget QSpinBox#qt_calendar_yearedit{
margin:2px;
}
QCalendarWidget QToolButton::menu-indicator{
image:None;
}
QCalendarWidget QTableView{
border-width:0px;
}
QCalendarWidget QWidget#qt_calendar_navigationbar{
border:1px solid #C0DCF2;
border-width:1px 1px 0px 1px;
background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:1,stop:0 #DEF0FE,stop:1 #C0DEF6);
}
QComboBox QAbstractItemView::item{
min-height:20px;
min-width:10px;
}
QTableView[model="true"]::item{
padding:0px;
margin:0px;
}
QTableView QLineEdit,QTableView QComboBox,QTableView QSpinBox,QTableView QDoubleSpinBox,QTableView QDateEdit,QTableView QTimeEdit,QTableView QDateTimeEdit{
border-width:0px;
border-radius:0px;
}
QTableView QLineEdit:focus,QTableView QComboBox:focus,QTableView QSpinBox:focus,QTableView QDoubleSpinBox:focus,QTableView QDateEdit:focus,QTableView QTimeEdit:focus,QTableView QDateTimeEdit:focus{
border-width:0px;
border-radius:0px;
}
QLineEdit,QTextEdit,QPlainTextEdit,QSpinBox,QDoubleSpinBox,QComboBox,QDateEdit,QTimeEdit,QDateTimeEdit{
background:#EAF7FF;
}
QTabWidget::pane:top{top:-1px;}
QTabWidget::pane:bottom{bottom:-1px;}
QTabWidget::pane:left{right:-1px;}
QTabWidget::pane:right{left:-1px;}
*:disabled{
background:#EAF7FF;
border-color:#DEF0FE;
color:#C0DCF2;
}
/*TextColor:#386487*/
/*PanelColor:#EAF7FF*/
/*BorderColor:#C0DCF2*/
/*NormalColorStart:#DEF0FE*/
/*NormalColorEnd:#C0DEF6*/
/*DarkColorStart:#F2F9FF*/
/*DarkColorEnd:#DAEFFF*/
/*HighColor:#00BB9E*/
                      '''
        # 加载设置好的样式
        self.setStyleSheet(qssStyle)

        # 动态显示时间在label上
        # 初始化一个定时器
        self.timer = QTimer()
        # 定时器结束，触发showTime方法
        self.timer.timeout.connect(self.showTime)
        self.timer.start()

    def showTime(self):
        # 获取系统当前时间
        time = QDateTime.currentDateTime()
        # 设置系统时间的显示格式
        timeDisplay = time.toString('yyyy-MM-dd hh:mm:ss dddd')
        # 在标签上显示时间
        self.time_label.setText("     " + timeDisplay)

    # UI变化处理
    def handle_ui_change(self):
        self.salechart_comboBox.addItems(["饼状图", "条形图", "散点图", "折线图", "热力图"])
        self.forc_year.addItems(["2", "3", "4", "5"])
        self.km_num.addItems(["2", "3", "4", "5", "6", "7"])
        self.tabWidget.tabBar().setVisible(False)
        self.userWidget.tabBar().setVisible(False)
        self.locate_tab.tabBar().setVisible(False)
        self.sale_tabWidget.tabBar().setVisible(False)
        self.hide_help()
        self.hide_help2()
        self.hide_help3()
        self.ana_view.setHorizontalHeaderLabels(["序号", "操作时间", "细分变量", "预测变量", "预测年数"])
        self.ana_view2.setHorizontalHeaderLabels(["序号", "操作时间", "赋权指标", "分析变量"])
        self.ana_view3.setHorizontalHeaderLabels(["序号", "操作时间", "拟建个数"])
        self.sta_view.setHorizontalHeaderLabels(["序号", "操作时间", "选择变量", "图表类型"])

    # 所有Button的消息与槽的通信
    def handle_buttons(self):
        self.saleButton.clicked.connect(self.open_sale_tab)
        self.locateButton.clicked.connect(self.open_locate_tab)
        self.userButton.clicked.connect(self.open_user_tab)
        self.locate_backButton.clicked.connect(self.back_locate_tab)
        self.forc_backButton.clicked.connect(self.back_locate_tab)
        self.km_backButton.clicked.connect(self.back_locate_tab)
        self.ana_back.clicked.connect(self.locateinput_tab)
        self.sale_back.clicked.connect(self.back_sale_tab)
        self.exitButton.clicked.connect(self.exit_sys)
        self.user_infButton.clicked.connect(self.show_userchoice)
        self.changeback_Button.clicked.connect(self.show_userfirst)
        self.sta_back.clicked.connect(self.show_userfirst)
        self.ana_back.clicked.connect(self.show_userfirst)
        self.ana_back2.clicked.connect(self.show_userfirst)
        self.user_logButton.clicked.connect(self.handle_login)
        self.user_editButton.clicked.connect(self.handle_rewrite)
        # 导入数据
        self.saleimportButton.clicked.connect(self.handle_file_dialog)
        self.locateimportButton.clicked.connect(self.handle_file_dialog2)
        self.sale_conf.clicked.connect(self.qr1)
        self.locate_conf.clicked.connect(self.qr2)
        # 数据可视化
        self.saleanl_Button.clicked.connect(self.handle_data_visual)
        # 卖场分析可视化之topsis
        self.top_conf.clicked.connect(self.topsis_queren)
        # 卖场分析可视化之GM(1,1)
        self.forc_conf.clicked.connect(self.GM_queren)
        # K-means可视化
        self.km_conf.clicked.connect(self.K_means_queren)
        # 帮助
        self.top_help.clicked.connect(self.show_help)
        self.top_help2.clicked.connect(self.show_help2)
        self.top_help3.clicked.connect(self.show_help3)
        self.help_back.clicked.connect(self.hide_help)
        self.help_back2.clicked.connect(self.hide_help2)
        self.help_back3.clicked.connect(self.hide_help3)
        self.locate_topsisButton.clicked.connect(self.choose_top)
        self.locate_fcsButton.clicked.connect(self.choose_forc)
        self.locate_kmeansButton.clicked.connect(self.choose_km)
        # 数据可视化记录
        self.user_saleButton.clicked.connect(self.open_sta_tab)
        self.user_locateButton.clicked.connect(self.open_ana_tab)
        # 查看图片
        self.sta_browse.clicked.connect(self.open_sta_pic)
        self.ana_browse.clicked.connect(self.open_GM_pic)
        self.ana_browse2.clicked.connect(self.open_topsis_pic)
        self.ana_browse3.clicked.connect(self.open_kmeans_pic)
        # 删除记录
        self.sta_delete.clicked.connect(self.del_sta)
        self.ana_delete.clicked.connect(self.del_GM)
        self.ana_delete2.clicked.connect(self.del_topsis)
        self.ana_delete3.clicked.connect(self.del_kmeans)


    def K_means_queren(self):
        if not self.queren2:
            self.statusBar().showMessage("请先导入数据", 5000)
            return
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        citys = []
        topsis_score = []
        long_la = []
        for i in self.long_la_topsisScore:
            citys.append(i)
            topsis_score.append(self.long_la_topsisScore[i][2])
            long_la.append([self.long_la_topsisScore[i][0], self.long_la_topsisScore[i][1]])
        X = np.array(long_la)

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
            self.km_narr.append("当聚类中心数量 =" + str(n_clusters) + "时，K-means聚类轮廓系数为 :" + str(silhouette_avg) + "\n")
        self.km_na2.append("当聚类中心数量为" + str(ans) + "时，轮廓系数最大，聚类效果最好")
        QtWidgets.QApplication.processEvents()
        get_kpic(ans, X)
        cluster = KMeans(n_clusters=ans, random_state=0).fit(X)  # 实例化并训练模型
        y_pred = cluster.labels_  # 重要属性labels_，查看聚好的类别
        centroid = cluster.cluster_centers_
        distance = []
        for i in range(len(long_la)):
            distance.append(
                sqrt(pow(centroid[y_pred[i]][0] - long_la[i][0], 2) + pow(centroid[y_pred[i]][1] - long_la[i][1], 2)))
        dictt = {}
        for i in range(len(citys)):
            dictt[citys[i]] = topsis_score[i] / distance[i]
        t = sorted(dictt.items(), key=lambda x: x[1], reverse=True)

        x_city = []
        y_score = []
        n = 0
        for i in t:
            if n == int(self.km_num.currentText()):
                break
            x_city.append(i[0])
            y_score.append(i[1] * 1000)
            n += 1
        plt.figure(figsize=(10, 4))
        plt.grid(alpha=0.4)
        plt.barh(x_city, y_score, height=0.4, color="#FFC125")
        plt.yticks(x_city, x_city, fontsize=10)
        plt.xlabel("topsis评分", fontsize=12)
        plt.ylabel("评价对象", fontsize=12)
        plt.title("拟建议建仓储的" + self.km_num.currentText() + "个城市与相应评分", fontsize=15)
        plt.show()
        # 数据库搞一搞
        time = QDateTime.currentDateTime()
        ymd = time.toString("yyyy-MM-dd hh:mm:ss")
        conn = sqlite3.connect('user_m.db')
        cur = conn.cursor()
        pname = "ttt.png"
        plt.savefig(pname)
        if os.path.exists(pname):
            with open(pname, 'rb') as f:
                Pic_byte = f.read()
                # 字节码进行编码
                content = base64.b64encode(Pic_byte)
                sql = f"INSERT INTO KmeansRecord " \
                      f"(user_id,num_cangku,date_time,width, height, image_bytes) " \
                      f"VALUES (?,?,?,?,?,?);"
                cur.execute(sql, (self.user_id,
                                  self.km_num.currentText(),
                                  ymd, 418, 412, content))
                conn.commit()
        else:
            print("无法找到图片")
        os.remove(pname)

    def GM_queren(self):
        if not self.queren2:
            self.statusBar().showMessage("请先导入数据", 5000)
            return
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'
        pd_data = pd.read_excel(self.datafile2, sheet_name=self.sheet_name2)
        pd_data["年份"] = pd_data.apply(lambda x: get_year(x["订单日期"]), axis=1)
        if not len(self.forc_cate.text()) == 0:
            if self.forc_cate.text() not in pd_data[self.ana_var.currentText()].tolist():
                self.statusBar().showMessage("请输入" + self.ana_var.currentText() + "中存在的值！", 5000)
                return
            else:
                pd_data = pd_data[(pd_data[self.ana_var.currentText()] == self.forc_cate.text())]
        else:
            self.statusBar().showMessage("请输入分析变量的细分类别",5000)
            return
        grouped = pd_data.groupby(pd_data['年份'])
        groued_year = grouped["年份"].unique()
        year = []
        for i in groued_year:
            year.append(int(i[0]))
        grouped_sum = grouped.sum()
        # GM(1,1)预测
        print(3)
        x0 = grouped_sum[self.forc_var.currentText()].tolist()
        a, b, residual_error_max, f, x1_pre, pplltt = GM1_1(x0)
        print(4)
        if pplltt == False:
            self.statusBar().showMessage("数据未通过级比检验,结果正确性有风险", 5000)
        if pplltt:
            # 往后预测年数
            y_n = int(self.forc_year.currentText())
            # 预测值
            x2_pre = []
            y_year = []
            try:
                for i in range(len(x1_pre) + 1, len(x1_pre) + y_n + 1):
                    x2_pre.append(f(i))
                    y_year.append(year[0] + i - 1)
            except Exception as e:
                print(e)
            s_str = "GM(1,1)以" + str(self.ana_var.currentText()) + str(self.forc_cate.text()) + \
                    "为基准对" + str(self.forc_var.currentText()) + "的往后" + str(self.forc_year.currentText()) + \
                    "年的预测"
            # 画图
            plt.title(s_str, fontsize=15)
            plt.plot(year, x0, color='r', linestyle="dashdot", label='真实值')
            plt.plot(year, x1_pre, color='b', linestyle=":", label="拟合值")
            plt.plot(y_year, x2_pre, color='g', linestyle="dashed", label="预测值")
            plt.legend(loc='upper right')
            plt.show()
        # 数据库搞一搞
        time = QDateTime.currentDateTime()
        ymd = time.toString("yyyy-MM-dd hh:mm:ss")
        conn = sqlite3.connect('user_m.db')
        cur = conn.cursor()
        pname = "ttt.png"
        plt.savefig(pname)
        if os.path.exists(pname):
            with open(pname, 'rb') as f:
                Pic_byte = f.read()
                # 字节码进行编码
                content = base64.b64encode(Pic_byte)
                sql = f"INSERT INTO GM11Record " \
                      f"(user_id,ana_var,xifen_leibie,yuce_var,yuce_year,date_time,width, height, image_bytes) " \
                      f"VALUES (?,?,?,?,?,?,?,?,?);"
                cur.execute(sql, (self.user_id,
                                  self.ana_var.currentText(),
                                  self.forc_cate.text(),
                                  self.forc_var.currentText(),
                                  self.forc_year.currentText(),
                                  ymd, 418, 412, content))
                conn.commit()
        else:
            print("无法找到图片")
        os.remove(pname)

    def topsis_queren(self):
        if not self.queren2:
            self.statusBar().showMessage("请先导入数据", 5000)
            return
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        pd_data = pd.read_excel(self.datafile2, sheet_name=self.sheet_name2)
        grouped = pd_data.groupby(pd_data[self.top_var.currentText()])
        grouped_sum = grouped.sum()
        if not self.totalsale_top.isChecked() and \
                not self.order_top.isChecked() and \
                not self.discount_top.isChecked() and \
                not self.profit_top.isChecked():
            self.statusBar().showMessage("请至少选择一个指标", 5000)
            return
        s_str = ""
        if not self.totalsale_top.isChecked():
            grouped_sum.drop(["销售额"], axis=1, inplace=True)
        else:
            s_str += "*销售额*"
        if not self.order_top.isChecked():
            grouped_sum.drop(["数量"], axis=1, inplace=True)
        else:
            s_str += "*数量*"
        if not self.discount_top.isChecked():
            grouped_sum.drop(["折扣"], axis=1, inplace=True)
        else:
            s_str += "*折扣*"
        if not self.profit_top.isChecked():
            grouped_sum.drop(["利润"], axis=1, inplace=True)
        else:
            s_str += "*利润*"
        w = weights(np.array(grouped_sum.values)).round(4)
        sta_data = Standard(np.array(grouped_sum.values))
        sco = Score(sta_data, w)
        dictt = {}
        for i in range(len(sco)):
            dictt[grouped_sum.index[i]] = sco[i]
        t = sorted(dictt.items(), key=lambda x: x[1], reverse=True)
        val, ind = get_valind(t, 12)
        for i in range(len(val)):
            val[i] *= 3000
        try:
            plt.figure(figsize=(10, 4))
            plt.grid(alpha=0.4)
            plt.barh(ind, val, height=0.4, label="第二天", color="#FFC125")
            plt.yticks(ind, ind, fontsize=10)
            plt.xlabel("topsis评分", fontsize=12)
            plt.ylabel("评价对象", fontsize=12)
            plt.title("熵权法以" + s_str + "为指标的topsis评分分析", fontsize=15)
            plt.show()
        except Exception as e:
            print(e)
        # 数据库搞一搞
        time = QDateTime.currentDateTime()
        ymd = time.toString("yyyy-MM-dd hh:mm:ss")
        conn = sqlite3.connect('user_m.db')
        cur = conn.cursor()
        pname = "ttt.png"
        plt.savefig(pname)
        if os.path.exists(pname):
            with open(pname, 'rb') as f:
                Pic_byte = f.read()
                # 字节码进行编码
                content = base64.b64encode(Pic_byte)
                sql = f"INSERT INTO topsisRecord " \
                      f"(user_id,ana_var,quanzhong_zhibiao,date_time,width, height, image_bytes) " \
                      f"VALUES (?,?,?,?,?,?,?);"
                cur.execute(sql, (self.user_id, self.top_var.currentText(), s_str, ymd, 418, 412, content))
                conn.commit()
        else:
            print("无法找到图片")
        os.remove(pname)

    def show_help(self):
        self.groupBox_3.show()

    def hide_help(self):
        self.groupBox_3.hide()

    def show_help2(self):
        self.groupBox_4.show()

    def hide_help2(self):
        self.groupBox_4.hide()

    def show_help3(self):
        self.groupBox_5.show()

    def hide_help3(self):
        self.groupBox_5.hide()

    # 删除记录
    def del_sta(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"delete from dataRecord where data_record_id=?"
        cur.execute(sql, [self.sta_combo.currentText()])
        conn.commit()
        self.sta_view.clear()
        self.sta_view.setHorizontalHeaderLabels(["序号", "操作时间", "选择变量", "图表类型"])
        self.sta_combo.clear()
        sql = f"SELECT data_record_id," \
              f"date_time," \
              f"ana_var," \
              f"chart_type FROM dataRecord WHERE user_id=?"
        result = cur.execute(sql, [self.user_id]).fetchall()
        rows = len(result)
        cols = len(result[0])
        self.sta_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sta_view.clear()
        self.sta_view.setHorizontalHeaderLabels(["序号", "操作时间", "选择变量", "图表类型"])
        for i in range(rows):
            self.sta_combo.addItem(str(result[i][0]))
            for j in range(cols):
                item = QtWidgets.QTableWidgetItem(str(result[i][j]))
                self.sta_view.setItem(i, j, item)

    def del_GM(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"delete from GM11Record where data_record_id=?"
        cur.execute(sql, [self.ana_combo.currentText()])
        conn.commit()
        self.ana_view.clear()
        self.ana_view.setHorizontalHeaderLabels(["序号", "操作时间", "细分变量", "预测变量", "预测年数"])
        self.ana_combo.clear()
        sql = f"SELECT data_record_id," \
              f"date_time," \
              f"xifen_leibie," \
              f"yuce_var," \
              f"yuce_year FROM GM11Record WHERE user_id=?"
        result = cur.execute(sql, [self.user_id]).fetchall()
        rows = len(result)
        cols = len(result[0])
        self.ana_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ana_view.clear()
        self.ana_view.setHorizontalHeaderLabels(["序号", "操作时间", "细分变量", "预测变量", "预测年数"])
        for i in range(rows):
            self.ana_combo.addItem(str(result[i][0]))
            for j in range(cols):
                item = QtWidgets.QTableWidgetItem(str(result[i][j]))
                self.ana_view.setItem(i, j, item)

    def del_topsis(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"delete from topsisRecord where data_record_id=?"
        cur.execute(sql, [self.ana_combo2.currentText()])
        conn.commit()
        self.ana_view2.clear()
        self.ana_view2.setHorizontalHeaderLabels(["序号", "操作时间", "赋权指标", "分析变量"])
        self.ana_combo2.clear()
        sql = f"SELECT data_record_id," \
              f"date_time," \
              f"quanzhong_zhibiao," \
              f"ana_var FROM topsisRecord WHERE user_id=?"
        result = cur.execute(sql, [self.user_id]).fetchall()
        rows = len(result)
        cols = len(result[0])
        self.ana_view2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ana_view2.clear()
        self.ana_view2.setHorizontalHeaderLabels(["序号", "操作时间", "赋权指标", "分析变量"])
        for i in range(rows):
            self.ana_combo2.addItem(str(result[i][0]))
            for j in range(cols):
                item = QtWidgets.QTableWidgetItem(str(result[i][j]))
                self.ana_view2.setItem(i, j, item)

    def del_kmeans(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"delete from KmeansRecord where data_record_id=?"
        cur.execute(sql, [self.ana_combo3.currentText()])
        conn.commit()
        self.ana_view3.clear()
        self.ana_view3.setHorizontalHeaderLabels(["序号", "操作时间", "拟建个数"])
        self.ana_combo3.clear()
        sql = f"SELECT data_record_id," \
              f"date_time," \
              f"num_cangku FROM KmeansRecord WHERE user_id=?"
        result = cur.execute(sql, [self.user_id]).fetchall()
        rows = len(result)
        cols = len(result[0])
        self.ana_view3.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ana_view3.clear()
        self.ana_view3.setHorizontalHeaderLabels(["序号", "操作时间", "拟建个数"])
        for i in range(rows):
            self.ana_combo3.addItem(str(result[i][0]))
            for j in range(cols):
                item = QtWidgets.QTableWidgetItem(str(result[i][j]))
                self.ana_view3.setItem(i, j, item)


    # 打开图片
    def open_sta_pic(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"SELECT image_bytes FROM dataRecord WHERE data_record_id=?"
        cur.execute(sql,[self.sta_combo.currentText()])
        value = cur.fetchone()
        if value:
            #base64编码对应的解码（解码完字符串）
            str_encode=base64.b64decode(value[0])
            # 将open方法读取的字节码转为opencv格式的数据
            nparr = np.frombuffer(str_encode, np.uint8)
            img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("img",img_decode)
            cv2.waitKey(0)

    def open_GM_pic(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"SELECT image_bytes FROM GM11Record WHERE data_record_id=?"
        cur.execute(sql, [self.ana_combo.currentText()])
        value = cur.fetchone()
        if value:
            # base64编码对应的解码（解码完字符串）
            str_encode = base64.b64decode(value[0])
            # 将open方法读取的字节码转为opencv格式的数据
            nparr = np.frombuffer(str_encode, np.uint8)
            img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("img", img_decode)
            cv2.waitKey(0)

    def open_topsis_pic(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"SELECT image_bytes FROM topsisRecord WHERE data_record_id=?"
        cur.execute(sql, [self.ana_combo2.currentText()])
        value = cur.fetchone()
        if value:
            # base64编码对应的解码（解码完字符串）
            str_encode = base64.b64decode(value[0])
            # 将open方法读取的字节码转为opencv格式的数据
            nparr = np.frombuffer(str_encode, np.uint8)
            img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("img", img_decode)
            cv2.waitKey(0)

    def open_kmeans_pic(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"SELECT image_bytes FROM KmeansRecord WHERE data_record_id=?"
        cur.execute(sql, [self.ana_combo3.currentText()])
        value = cur.fetchone()
        if value:
            # base64编码对应的解码（解码完字符串）
            str_encode = base64.b64decode(value[0])
            # 将open方法读取的字节码转为opencv格式的数据
            nparr = np.frombuffer(str_encode, np.uint8)
            img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("img", img_decode)
            cv2.waitKey(0)

    # 选项卡联动
    def open_sale_tab(self):
        self.tabWidget.setCurrentIndex(0)
        self.sale_tabWidget.setCurrentIndex(0)
        self.num_label1.setText("")

    def open_locate_tab(self):
        self.tabWidget.setCurrentIndex(1)
        self.locate_tab.setCurrentIndex(0)
        self.num_label2.setText("")

    def open_user_tab(self):
        self.tabWidget.setCurrentIndex(2)
        self.userWidget.setCurrentIndex(0)

    def open_sta_tab(self):
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"SELECT data_record_id," \
              f"date_time," \
              f"ana_var," \
              f"chart_type FROM dataRecord WHERE user_id=?"
        result = cur.execute(sql, [self.user_id]).fetchall()
        rows = len(result)
        cols = len(result[0])
        self.sta_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sta_view.clear()
        self.sta_view.setHorizontalHeaderLabels(["序号", "操作时间", "选择变量", "图表类型"])
        self.sta_combo.clear()
        for i in range(rows):
            self.sta_combo.addItem(str(result[i][0]))
            for j in range(cols):
                item = QtWidgets.QTableWidgetItem(str(result[i][j]))
                self.sta_view.setItem(i,j,item)
        self.userWidget.setCurrentIndex(2)

    def open_ana_tab(self):
        # GM(1,1)
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        sql = f"SELECT data_record_id," \
              f"date_time," \
              f"xifen_leibie," \
              f"yuce_var," \
              f"yuce_year FROM GM11Record WHERE user_id=?"
        result = cur.execute(sql, [self.user_id]).fetchall()
        rows = len(result)
        cols = len(result[0])
        self.ana_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ana_view.clear()
        self.ana_view.setHorizontalHeaderLabels(["序号", "操作时间", "细分变量", "预测变量", "预测年数"])
        self.ana_combo.clear()
        for i in range(rows):
            self.ana_combo.addItem(str(result[i][0]))
            for j in range(cols):
                item = QtWidgets.QTableWidgetItem(str(result[i][j]))
                self.ana_view.setItem(i,j,item)
        # Topsis分析
        sql = f"SELECT data_record_id," \
              f"date_time," \
              f"quanzhong_zhibiao," \
              f"ana_var FROM topsisRecord WHERE user_id=?"
        result = cur.execute(sql, [self.user_id]).fetchall()
        rows = len(result)
        cols = len(result[0])
        self.ana_view2.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ana_view2.clear()
        self.ana_view2.setHorizontalHeaderLabels(["序号", "操作时间", "赋权指标", "分析变量"])
        self.ana_combo2.clear()
        for i in range(rows):
            self.ana_combo2.addItem(str(result[i][0]))
            for j in range(cols):
                item = QtWidgets.QTableWidgetItem(str(result[i][j]))
                self.ana_view2.setItem(i, j, item)
        # Kmeans分析
        sql = f"SELECT data_record_id," \
              f"date_time," \
              f"num_cangku FROM KmeansRecord WHERE user_id=?"
        result = cur.execute(sql, [self.user_id]).fetchall()
        rows = len(result)
        cols = len(result[0])
        self.ana_view3.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ana_view3.clear()
        self.ana_view3.setHorizontalHeaderLabels(["序号", "操作时间", "拟建个数"])
        self.ana_combo3.clear()
        for i in range(rows):
            self.ana_combo3.addItem(str(result[i][0]))
            for j in range(cols):
                item = QtWidgets.QTableWidgetItem(str(result[i][j]))
                self.ana_view3.setItem(i, j, item)
        self.userWidget.setCurrentIndex(3)


    def choose_top(self):
        self.locate_tab.setCurrentIndex(2)

    def choose_forc(self):
        self.locate_tab.setCurrentIndex(3)

    def choose_km(self):
        self.locate_tab.setCurrentIndex(4)

    def back_locate_tab(self):
        self.locate_tab.setCurrentIndex(1)

    def locateinput_tab(self):
        self.locate_tab.setCurrentIndex(0)

    def back_sale_tab(self):
        self.sale_tabWidget.setCurrentIndex(0)

    def show_userchoice(self):
        self.userWidget.setCurrentIndex(1)

    def show_userfirst(self):
        self.userWidget.setCurrentIndex(0)

    def handle_data_visual(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
        if self.queren == False:
            self.statusBar().showMessage("请先导入数据!", 5000)
            return
        pd_data = pd.read_excel(self.datafile, sheet_name=self.sheet_name)
        p_new = pd_data.groupby([self.salevar_comboBox.currentText()]).size()
        if self.salechart_comboBox.currentText() == "饼状图":
            val, ind = self.set_pie(p_new)
            plt.pie(val, labels=ind, autopct='%3.1f%%')
            plt.show()
        elif self.salechart_comboBox.currentText() == "条形图":
            val, ind = self.set_bar(p_new)
            plt.bar(x=range(len(ind)),  # 指定条形图x轴的刻度值(有的是用left，有的要用x)
                    height=val,  # 指定条形图y轴的数值（python3.7不能用y，而应该用height）
                    tick_label=ind,  # 指定条形图x轴的刻度标签
                    color='steelblue',  # 指定条形图的填充色
                    )
            for i in range(len(val)):
                plt.text(i, val[i] + 0.1, "%s" % round(val[i], 1), ha='center')  # round(y,1)是将y值四舍五入到一个小数位
            plt.show()
        elif self.salechart_comboBox.currentText() == "散点图":
            val, ind = self.set_scatter(p_new)
            plt.scatter(ind, val, marker='o')
            plt.show()
        elif self.salechart_comboBox.currentText() == "折线图":
            val, ind = self.set_plot(p_new)
            plt.plot(ind, val)
            plt.show()
        elif self.salechart_comboBox.currentText() == "热力图":
            if not self.sheet_name == 0:
                self.statusBar().showMessage("该图表不支持热力图!", 5000)
                return
            drop_list = []
            for x in pd_data.columns:
                if pd_data[x].dtype == object:
                    drop_list.append(x)
            pd_data.drop(drop_list, axis=1, inplace=True)
            corr = pd_data.corr()
            ax = sns.heatmap(corr, vmax=.8, square=True, annot=True)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()
        # 数据库搞一搞
        time = QDateTime.currentDateTime()
        ymd = time.toString("yyyy-MM-dd hh:mm:ss")
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        pname = "ttt.png"
        plt.savefig(pname)
        if os.path.exists(pname):
            with open(pname, 'rb') as f:
                Pic_byte = f.read()
                # 字节码进行编码
                content = base64.b64encode(Pic_byte)
                sql = f"INSERT INTO dataRecord " \
                      f"(user_id,ana_var,chart_type,date_time,width, height, image_bytes) " \
                      f"VALUES (?,?,?,?,?,?,?);"
                cur.execute(sql, (self.user_id,
                                  self.salevar_comboBox.currentText(),
                                  self.salechart_comboBox.currentText(),
                                  ymd, 418, 412, content))
                conn.commit()
        else:
            print("无法找到图片")
        os.remove(pname)

    def handle_file_dialog(self):
        dig = QFileDialog()
        dig.setFileMode(QFileDialog.ExistingFile)
        dig.setFilter(QDir.Files)

        if dig.exec_():
            # 接受选中文件的路径，默认为列表
            filenames = dig.selectedFiles()
            self.datafile = filenames[0]
            # 列表中的第一个元素即是文件路径，以只读的方式打开文件
            try:
                table = xlrd.open_workbook(filenames[0])
                self.sheet_name = int(self.sale_numEdit.text()) - 1
                table_by_sheet0 = table.sheet_by_index(int(self.sale_numEdit.text()) - 1)
                rows = table_by_sheet0.nrows
                cols = table_by_sheet0.ncols

                content_list = []
                for i in range(rows):
                    content_list.append(table_by_sheet0.row_values(i))
                self.sale_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
                self.sale_view.clear()
                self.sale_view.setHorizontalHeaderLabels(content_list[0])
                self.sale_view.setSelectionBehavior(QAbstractItemView.SelectRows)
                for i in range(max(rows - 1, 21)):
                    for j in range(cols):
                        item = QtWidgets.QTableWidgetItem(str(content_list[i + 1][j]))
                        self.sale_view.setItem(i, j, item)
            except Exception as e:
                print(e)

    def qr1(self):
        if self.datafile == "":
            self.statusBar().showMessage("请先导入数据", 5000)
            return
        self.queren = True
        try:
            pd_data = pd.read_excel(self.datafile, sheet_name=self.sheet_name)
            table = xlrd.open_workbook(self.datafile)
            if len(self.sale_numEdit.text()) == 0:
                self.num_label1.setText("请输入要读入的表格编号")
                return
            elif not self.sale_numEdit.text().isdigit():
                self.num_label1.setText("请确认输入的表格编号为数字")
                return
            elif int(self.sale_numEdit.text()) - 1 > len(table.sheets()) - 1:
                self.num_label1.setText("输入的表格编号超出索引！")
                return
            self.sale_tabWidget.setCurrentIndex(1)
            self.sheet_name = int(self.sale_numEdit.text()) - 1
            table_by_sheet0 = table.sheet_by_index(int(self.sale_numEdit.text()) - 1)
            rows = table_by_sheet0.nrows
            cols = table_by_sheet0.ncols

            for i in range(len(pd_data.columns)):
                self.salevar_comboBox.addItem(pd_data.columns[i])

        except Exception as e:
            print(e)

    def qr2(self):
        if self.datafile2 == "":
            self.statusBar().showMessage("请先导入数据", 5000)
            return
        self.queren2 = True
        try:
            pd_data = pd.read_excel(self.datafile2, sheet_name=self.sheet_name)
            table = xlrd.open_workbook(self.datafile2)
            if len(self.sale_numEdit.text()) == 0:
                self.num_label1.setText("请输入要读入的表格编号")
                return
            elif not self.sale_numEdit.text().isdigit():
                self.num_label1.setText("请确认输入的表格编号为数字")
                return
            elif int(self.sale_numEdit.text()) - 1 > len(table.sheets()) - 1:
                self.num_label1.setText("输入的表格编号超出索引！")
                return
            self.locate_tab.setCurrentIndex(1)
            self.sheet_name2 = int(self.sale_numEdit.text()) - 1
            table_by_sheet0 = table.sheet_by_index(int(self.sale_numEdit.text()) - 1)
            rows = table_by_sheet0.nrows
            cols = table_by_sheet0.ncols

            for i in range(len(pd_data.columns)):
                if pd_data[pd_data.columns[i]].dtype == object:
                    self.top_var.addItem(pd_data.columns[i])
                    self.ana_var.addItem(pd_data.columns[i])
                else:
                    self.forc_var.addItem(pd_data.columns[i])

        except Exception as e:
            print(e)

    def handle_file_dialog2(self):
        dig = QFileDialog()
        dig.setFileMode(QFileDialog.ExistingFile)
        dig.setFilter(QDir.Files)

        if dig.exec_():
            # 接受选中文件的路径，默认为列表
            filenames = dig.selectedFiles()
            # 列表中的第一个元素即是文件路径，以只读的方式打开文件
            self.datafile2 = filenames[0]
            try:
                table = xlrd.open_workbook(filenames[0])
                self.sheet_name2 = int(self.locate_numEdit.text()) - 1
                table_by_sheet0 = table.sheet_by_index(int(self.locate_numEdit.text()) - 1)
                rows = table_by_sheet0.nrows
                cols = table_by_sheet0.ncols

                content_list = []
                for i in range(rows):
                    content_list.append(table_by_sheet0.row_values(i))
                self.locate_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
                self.locate_view.clear()
                self.locate_view.setHorizontalHeaderLabels(content_list[0])
                self.locate_view.setSelectionBehavior(QAbstractItemView.SelectRows)
                for i in range(max(rows - 1, 21)):
                    for j in range(cols):
                        item = QtWidgets.QTableWidgetItem(str(content_list[i + 1][j]))
                        self.locate_view.setItem(i, j, item)

            except Exception as e:
                print(e)

    def handle_login(self):
        if not len(self.user_nameEdit.text()):
            self.error1.setText("用户名不能为空！")
            self.error2.setText("")
        elif not len(self.user_pswEdit.text()):
            self.error2.setText("请输入密码！")
            self.error1.setText("")
        elif len(self.user_nameEdit.text()) \
                and len(self.user_pswEdit.text()):
            db_conn = sqlite3.connect(db_file)
            cur = db_conn.cursor()

            sql_select = "SELECT * FROM account WHERE user_name=\'" + self.user_nameEdit.text() + "\'"
            result = cur.execute(sql_select).fetchall()

            if len(result):
                if result[0][2] != self.user_pswEdit.text():
                    self.error2.setText("密码错误！")
                    self.error1.setText("")
                    self.user_pswEdit.setText("")
                else:
                    self.user_editBox.setEnabled(True)
                    self.error2.setText("")
                    self.error1.setText("")
            else:
                self.error1.setText("用户未注册！")
                self.user_pswEdit.setText("")

            cur.close()
            db_conn.close()

    def handle_rewrite(self):
        if not len(self.user_nnameEdit.text()):
            self.error3.setText("用户名不能为空！")
            self.error4.setText("")
            self.error5.setText("")
        # 确认密码不能为空
        elif not len(self.user_confpswfEdit.text()):
            self.error5.setText("确认密码不能为空！")
            self.error3.setText("")
            self.error4.setText("")
        # 请输入一致的密码
        elif len(self.user_nnameEdit.text()) \
                and len(self.user_npswEdit.text()) \
                and self.user_confpswfEdit.text() != self.user_npswEdit.text():
            self.error5.setText("两次密码不一致！")
            self.error4.setText("")
            self.error3.setText("")
            self.user_confpswfEdit.setText("")
        # 请输入密码
        elif not len(self.user_npswEdit.text()):
            self.error3.setText("密码不能为空！")
            self.error4.setText("")
            self.error5.setText("")


        elif len(self.user_nnameEdit.text()) \
                and len(self.user_npswEdit.text()) \
                and self.user_confpswfEdit.text() == self.user_npswEdit.text():
            db_conn = sqlite3.connect(db_file)
            cur = db_conn.cursor()
            sql_select = "SELECT * FROM account WHERE user_name=\'" + self.user_nnameEdit.text() + "\'"
            result = cur.execute(sql_select).fetchall()
            old_name = self.user_nameEdit.text()
            new_name = self.user_nnameEdit.text()
            new_psw = self.user_npswEdit.text()
            sql = "UPDATE account SET user_name=?,password=? WHERE user_name=\'" + old_name + "\'"
            # 4、执行语句
            cur.execute(sql, (new_name, new_psw))
            self.user_nameEdit.setText("")
            self.user_pswEdit.setText("")
            self.user_nnameEdit.setText("")
            self.user_npswEdit.setText("")
            self.user_confpswfEdit.setText("")
            self.statusBar().showMessage('用户信息修改成功！', 5000)
            # 5、insert、update、delete必须显示提交
            db_conn.commit()
            cur.close()
            db_conn.close()

    def set_pie(self, p_new):
        if len(p_new.tolist()) <= 20:
            plt.figure(num=self.salevar_comboBox.currentText() + "饼状图")  # 可选参数,facecolor为背景颜色
            plt.title(self.salevar_comboBox.currentText() + "可视化" + self.salechart_comboBox.currentText())  # 加标题
            return p_new.tolist(), p_new.index
        plt.figure(num=self.salevar_comboBox.currentText() + "Top20饼状统计图")  # 可选参数,facecolor为背景颜色
        plt.title(self.salevar_comboBox.currentText() + "Top20可视化" + self.salechart_comboBox.currentText())  # 加标题
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
        return val, ind

    def set_bar(self, p_new):
        if len(p_new.tolist()) <= 12:
            plt.figure(num=self.salevar_comboBox.currentText() + "条形统计图")  # 可选参数,facecolor为背景颜色
            plt.title(self.salevar_comboBox.currentText() + "可视化" + self.salechart_comboBox.currentText())  # 加标题
            return p_new.tolist(), p_new.index
        plt.figure(num=self.salevar_comboBox.currentText() + "Top12条形统计图")  # 可选参数,facecolor为背景颜色
        plt.title(self.salevar_comboBox.currentText() + "Top12可视化" + self.salechart_comboBox.currentText())  # 加标题
        dict = {}
        val = []
        ind = []
        for i in range(len(p_new.tolist())):
            dict[p_new.index[i]] = p_new.tolist()[i]
        t = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        i = 0
        n_qi = 0
        for x in t:
            if i < 12:
                ind.append(x[0])
                val.append(x[1])
            else:
                n_qi += x[1]
            i += 1
        return val, ind

    def set_scatter(self, p_new):
        if len(p_new.tolist()) <= 12:
            plt.figure(num=self.salevar_comboBox.currentText() + "散点统计图")  # 可选参数,facecolor为背景颜色
            plt.title(self.salevar_comboBox.currentText() + "可视化" + self.salechart_comboBox.currentText())  # 加标题
            return p_new.tolist(), p_new.index
        plt.figure(num=self.salevar_comboBox.currentText() + "Top12散点统计图")  # 可选参数,facecolor为背景颜色
        plt.title(self.salevar_comboBox.currentText() + "Top12可视化" + self.salechart_comboBox.currentText())  # 加标题
        dict = {}
        val = []
        ind = []
        for i in range(len(p_new.tolist())):
            dict[p_new.index[i]] = p_new.tolist()[i]
        t = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        i = 0
        n_qi = 0
        for x in t:
            if i < 12:
                ind.append(x[0])
                val.append(x[1])
            else:
                n_qi += x[1]
            i += 1
        return val, ind

    def set_plot(self, p_new):
        if len(p_new.tolist()) <= 12:
            plt.figure(num=self.salevar_comboBox.currentText() + "折线统计图")  # 可选参数,facecolor为背景颜色
            plt.title(self.salevar_comboBox.currentText() + "可视化" + self.salechart_comboBox.currentText())  # 加标题
            return p_new.tolist(), p_new.index
        plt.figure(num=self.salevar_comboBox.currentText() + "Top12折线统计图")  # 可选参数,facecolor为背景颜色
        plt.title(self.salevar_comboBox.currentText() + "Top12可视化" + self.salechart_comboBox.currentText())  # 加标题
        dict = {}
        val = []
        ind = []
        for i in range(len(p_new.tolist())):
            dict[p_new.index[i]] = p_new.tolist()[i]
        t = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        i = 0
        n_qi = 0
        for x in t:
            if i < 12:
                ind.append(x[0])
                val.append(x[1])
            else:
                n_qi += x[1]
            i += 1
        return val, ind

    def exit_sys(self):
        messageBox = QMessageBox()
        messageBox.setWindowTitle('退出系统')
        messageBox.setText('确认退出吗？')
        messageBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        buttonY = messageBox.button(QMessageBox.Yes)
        buttonY.setText('退出系统')
        buttonN = messageBox.button(QMessageBox.No)
        buttonN.setText('退出登录')
        buttonC = messageBox.button(QMessageBox.Cancel)
        buttonC.setText('取消')
        messageBox.exec_()
        if messageBox.clickedButton() == buttonY:
            self.close()
        elif messageBox.clickedButton() == buttonN:
            self.main_app = LoginApp()
            self.close()
            self.main_app.show()
        else:
            return


def main():
    app = QApplication(sys.argv)
    window = LoginApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
