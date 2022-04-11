from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import sys
import sqlite3
from PyQt5.QtCore import *
import xlrd
from PyQt5.QtGui import *
import hashlib
import re
import datetime

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
                    self.main_app = MainApp()
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

class MainApp(QMainWindow, ui):
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
        self.changeback_Button.clicked.connect(self.show_userfirst)
        self.user_logButton.clicked.connect(self.handle_login)
        self.user_editButton.clicked.connect(self.handle_rewrite)
        # 导入数据
        self.saleimportButton.clicked.connect(self.handle_file_dialog)
        self.locateimportButton.clicked.connect(self.handle_file_dialog2)


     # 选项卡联动
    def open_sale_tab(self):
        self.tabWidget.setCurrentIndex(0)
        self.sale_tabWidget.setCurrentIndex(0)
    def open_locate_tab(self):
        self.tabWidget.setCurrentIndex(1)
        self.locate_tab.setCurrentIndex(0)
    def open_user_tab(self):
        self.tabWidget.setCurrentIndex(2)
        self.userWidget.setCurrentIndex(0)
    def back_locate_tab(self):
        self.locate_tab.setCurrentIndex(1)
    def show_userchoice(self):
        self.userWidget.setCurrentIndex(1)
    def show_userfirst(self):
        self.userWidget.setCurrentIndex(0)

    def handle_file_dialog(self):
        dig = QFileDialog()
        dig.setFileMode(QFileDialog.ExistingFile)
        dig.setFilter(QDir.Files)

        if dig.exec_():
            # 接受选中文件的路径，默认为列表
            filenames = dig.selectedFiles()
            # 列表中的第一个元素即是文件路径，以只读的方式打开文件
            try:
                table = xlrd.open_workbook(filenames[0])
                if len(self.sale_numEdit.text()) == 0:
                    print("请输入要读入的表格编号(从0开始的数字)")
                    return
                elif not self.sale_numEdit.text().isdigit():
                    print("请确认输入的表格编号为数字")
                    return
                elif int(self.sale_numEdit.text()) - 1 > len(table.sheets()) - 1:
                    print("输入的表格编号超出索引！")
                    return
                table_by_sheet0 = table.sheet_by_index(int(self.sale_numEdit.text()) - 1)
                rows = table_by_sheet0.nrows
                cols = table_by_sheet0.ncols

                content_list = []
                for i in range(rows):
                    content_list.append(table_by_sheet0.row_values(i))
                self.sale_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
                self.sale_view.clear()
                self.sale_view.setHorizontalHeaderLabels(content_list[0])
                # 添加下拉列表
                self.salevar_comboBox.addItems(content_list[0])
                self.sale_view.setSelectionBehavior(QAbstractItemView.SelectRows)
                for i in range(max(rows - 1, 21)):
                    for j in range(cols):
                        item = QtWidgets.QTableWidgetItem(str(content_list[i + 1][j]))
                        self.sale_view.setItem(i, j, item)

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
            try:
                table = xlrd.open_workbook(filenames[0])
                if len(self.locate_numEdit.text()) == 0:
                    print("请输入要读入的表格编号(从0开始的数字)")
                    return
                elif not self.locate_numEdit.text().isdigit():
                    print("请确认输入的表格编号为数字")
                    return
                elif int(self.locate_numEdit.text()) - 1 > len(table.sheets()) - 1:
                    print("输入的表格编号超出索引！")
                    return
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
            sql = "UPDATE account SET user_name=?,password=? WHERE user_name=\'" +old_name +"\'"
            # 4、执行语句
            cur.execute(sql, (new_name, new_psw))
            self.user_nameEdit.setText("")
            self.user_pswEdit.setText("")
            self.user_nnameEdit.setText("")
            self.user_npswEdit.setText("")
            self.user_confpswfEdit.setText("")
            self.statusBar().showMessage('用户信息修改成功！',5000)
            # 5、insert、update、delete必须显示提交
            db_conn.commit()
            cur.close()
            db_conn.close()






    def exit_sys(self):
        messageBox = QMessageBox()
        messageBox.setWindowTitle('退出系统')
        messageBox.setText('确认退出吗？')
        messageBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No| QMessageBox.Cancel)
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
