import sqlite3
import sys
import base64
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtCore import QDateTime


def huahua(name):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    # 添加图形属性
    plt.xlabel('这个是行属性字符串')
    plt.ylabel('这个是列属性字符串')
    plt.title('这个是总标题')
    y = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 这个是y轴的数据
    first_bar = plt.bar(range(len(y)), y, color='blue')  # 初版柱形图，x轴0-9，y轴是列表y的数据，颜色是蓝色
    # 开始绘制x轴的数据
    index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    name_list = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9']  # x轴标签
    plt.xticks(index, name_list)  # 绘制x轴的标签
    # 柱形图顶端数值显示
    for data in first_bar:
        y = data.get_height()
        x = data.get_x()
        plt.text(x + 0.15, y, str(y), va='bottom')  # 0.15为偏移值，可以自己调整，正好在柱形图顶部正中
    # 图片的显示及存储
    plt.savefig(name)


time = QDateTime.currentDateTime()
ymd = time.toString("yyyy-MM-dd hh:mm:ss")
conn = sqlite3.connect('user_m.db')
cur = conn.cursor()
pname = "ttt.png"
Pic_byte = None
huahua(pname)
if os.path.exists(pname):
    with open(pname, 'rb') as f:
        Pic_byte = f.read()
        # 字节码进行编码
        content = base64.b64encode(Pic_byte)
        sql = f"INSERT INTO dataRecord " \
              f"(user_id,ana_var,chart_type,date_time,width, height, image_bytes) " \
              f"VALUES (?,?,?,?,?,?,?);"
        cur.execute(sql, (55, "城市", "饼状图", ymd, 418, 412, content))
        conn.commit()
else:
    print("无法找到图片")


# 删除图片
os.remove("ttt.png")


# # 读取数据库的图片数据
sql = f"SELECT image_bytes FROM dataRecord WHERE data_record_id=?"
cur.execute(sql,[8])
value = cur.fetchone()
if value:
    #base64编码对应的解码（解码完字符串）
    str_encode=base64.b64decode(value[0])
    # 将open方法读取的字节码转为opencv格式的数据
    nparr = np.frombuffer(str_encode, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imshow("img",img_decode)
    cv2.waitKey(0)



# #1.打开数据库，获得连接对象
# conn=sqlite3.connect("user_m.db")
# #2.获得数据库的操作游标
# c=conn.cursor()
# sql = '''
#         CREATE TABLE "KmeansRecord" (
#             "data_record_id"	INTEGER NOT NULL,
#             "user_id"  INTEGER NOT NULL,
#             "num_cangku"	TEXT NOT NULL,
#             "date_time"          TEXT NOT NULL,
#             "width" INTEGER,
#             "height" INTEGER,
#             "image_bytes" BLOB,
#             PRIMARY KEY("data_record_id" AUTOINCREMENT)
#             FOREIGN KEY (user_id) REFERENCES account(user_id)
# )
# '''
# c.execute(sql)
# conn.commit()

