from PyQt5.QtWidgets import *
from login import Ui_login
from main import Ui_MainWindow
import sqlite3
# from PyQt5.uic import loadUiType
import sys
from os import path
import hashlib
import re
import datetime

# 数据库文件
db_file = "user_m.db"
# 获取与数据库的连接


# UI--Logic分离
# ui, _ = loadUiType('main.ui')
# login, _ = loadUiType('login.ui')

class LoginApp(QWidget, Ui_login):
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
        self.registBox.show()

    def hide_regist(self):
        self.registBox.hide()

    def handle_ui_change(self):
        self.hide_regist()

    def handle_login(self):
        if not len(self.login_username.text()):
            print("请输入用户名")
        elif not len(self.login_psw.text()):
            print("请输入密码")
        elif len(self.login_username.text())\
            and len(self.login_psw.text()):
            db_conn = sqlite3.connect(db_file)
            cur = db_conn.cursor()

            sql_select = "SELECT * FROM account WHERE user_name=\'" + self.login_username.text() + "\'"
            result = cur.execute(sql_select).fetchall()

            if len(result):
                if result[0][2] != self.login_psw.text():
                    print("密码错误")
                else:
                    self.main_app = MainApp()
                    self.close()
                    self.main_app.show()

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
                print("注册成功")
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
        # cyh
        self.regist_confButton.clicked.connect(self.user_regist)


class MainApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
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
QGroupBox::indicator,QTreeWidget::indicator,QListWidget::indicator{
padding:0px -3px 0px 0px;
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



                      '''
        # 加载设置好的样式
        self.setStyleSheet(qssStyle)

    # UI变化处理
    def handle_ui_change(self):
        self.tabWidget.tabBar().setVisible(False)
        self.userWidget.tabBar().setVisible(False)

    # 所有Button的消息与槽的通信
    def handle_buttons(self):
        self.saleButton.clicked.connect(self.open_sale_tab)
        self.locateButton.clicked.connect(self.open_locate_tab)
        self.userButton.clicked.connect(self.open_user_tab)
        self.locate_backButton.clicked.connect(self.back_locate_tab)
        self.exitButton.clicked.connect(self.exit_sys)
        self.user_infButton.clicked.connect(self.show_userchoice)
        self.user_logButton.clicked.connect(self.user_log)
        self.changeback_Button.clicked.connect(self.show_userfirst)

    # 选项卡联动
    def open_sale_tab(self):
        self.tabWidget.setCurrentIndex(0)

    def open_locate_tab(self):
        self.tabWidget.setCurrentIndex(1)

    def open_user_tab(self):
        self.tabWidget.setCurrentIndex(2)

    def back_locate_tab(self):
        self.locate_tab.setCurrentIndex(1)

    def show_userchoice(self):
        self.userWidget.setCurrentIndex(1)

    def show_userfirst(self):
        self.userWidget.setCurrentIndex(0)

    def user_log(self):
        self.user_editBox.setEnabled(True)

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
