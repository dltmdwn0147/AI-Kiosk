import os
import sys
import ast
import sqlite3
import datetime
import time
import pandas as pd
import socket
import re
import json
import threading
from random import randint

from PyQt5 import uic
from PyQt5.QtGui import *

# shopping_cart 모듈 임포트 (같은 폴더에 있어야 함)
try:
    import shopping_cart
    from shopping_cart import *
except ImportError:
    pass # shopping_cart가 없어도 UI 테스트는 가능하도록

# manager_page 모듈 임포트
try:
    from manager_page import *
except ImportError:
    pass

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

# --- 경로 설정 (절대 경로 사용) ---
# 현재 실행 중인 파일(mega_kiosk_ver1.py)의 폴더 경로를 기준점으로 잡습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'DATA')
UI_DIR = os.path.join(BASE_DIR, 'UI')
IMG_DIR = os.path.join(BASE_DIR, 'img')
DB_PATH = os.path.join(DATA_DIR, 'data.db')

# --- 소켓 통신 설정 ---
HOST = '127.0.0.1'
PORT = 9999
BUFFER_SIZE = 1024

def resource_path(relative_path):
    """UI 및 리소스 절대 경로 반환"""
    # UI 파일 경로 등을 BASE_DIR 기준으로 찾습니다.
    # 기존 코드의 ./UI/... 형태를 호환하기 위해 ./를 제거하거나 join을 사용
    clean_path = relative_path.replace('./', '').replace('/', os.sep)
    return os.path.join(BASE_DIR, clean_path)

# UI 불러오기 (경로 안전장치 추가)
main_page_class = uic.loadUiType(resource_path('UI/mega_ui_ver3.ui'))[0]
choose_option_class = uic.loadUiType(resource_path('UI/mega_choose_option_page.ui'))[0]
msg_box_class = uic.loadUiType(resource_path('UI/msg_box.ui'))[0]
point_page_class = uic.loadUiType(resource_path('UI/point_page.ui'))[0]
manager_page_class = uic.loadUiType(resource_path('UI/manager_page.ui'))[0]
receipt_page = uic.loadUiType(resource_path('UI/receipt_page_2.ui'))[0]

# --- 통신 스레드 클래스 ---
class CommThread(QThread):
    received_data = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.client_socket = None
        self._socket_lock = threading.Lock()

    def run(self):
        while True:
            client_socket = None
            try:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((HOST, PORT))
                print(f"[Client] Connected to {HOST}:{PORT}")
                with self._socket_lock:
                    self.client_socket = client_socket

                # TCP는 메시지 경계를 보장하지 않으므로 버퍼에 누적 후 개행 단위로 처리한다.
                buffer = ""
                while True:
                    data = client_socket.recv(BUFFER_SIZE)
                    if not data:
                        raise ConnectionError("Server closed the connection")

                    buffer += data.decode('utf-8', errors='ignore')
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line:
                            self.received_data.emit(line)

            except Exception as e:
                print(f"[Client] Connection issue: {e}")
                print("💡 서버(main_openai.py)가 실행 중인지 확인해주세요. 잠시 후 재연결합니다...")
                if client_socket:
                    try:
                        client_socket.close()
                    except Exception:
                        pass
                with self._socket_lock:
                    if self.client_socket is client_socket:
                        self.client_socket = None
                time.sleep(2)

    def send_message(self, payload: dict):
        message = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._socket_lock:
            sock = self.client_socket
        if not sock:
            print("[Client] Send skipped: not connected")
            return
        try:
            sock.sendall(message.encode("utf-8"))
        except Exception as e:
            print(f"[Client] Send error: {e}")

class Rept(QDialog, receipt_page):
    """영수증"""
    def __init__(self, parent, order_num, t_price):
        super().__init__()
        self.setupUi(self)
        self.parent = parent

        self.order_number_label.setText(str(order_num))
        self.total_price_label.setText(str(t_price))
        self.set_datetime()
        self.parent.fill_the_table_widget(self.tableWidget)

    def set_datetime(self):
        now = datetime.datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        self.date_label.setText(str(formatted_now))

class Point_Page(QDialog, point_page_class):
    """포인트 적립 창"""
    data_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        buttons = self.point_buttons.findChildren(QPushButton)
        for button in buttons:
            button.clicked.connect(self.write_point_num)

        self.point_confirm_btn.clicked.connect(self.point_check)
        self.cancel_btn.clicked.connect(self.close)

    def write_point_num(self):
        num_list = [str(num) for num in range(0, 10)]
        num_list.append('010')

        self.user_num = []
        btn_name = self.sender().text()
        now_label_text = self.user_number_label.text()

        if btn_name in num_list:
            now_label_text += str(btn_name)
            self.only_num = now_label_text.replace('-', '')
            self.user_num.append(btn_name)
            
            if len(self.only_num) <= 11:
                self.user_number_label.setText(self.mask_numbers(now_label_text))
        else:
            if len(self.user_num) > 0:
                self.user_num = self.user_num[:-1]
            now_label_text = now_label_text[:-1]
            self.user_number_label.setText(self.mask_numbers(now_label_text))

    def mask_numbers(self, i):
        i = i.replace('-', '')
        if len(i) <= 3:
            return i
        elif 3 < len(i) < 8:
            return f'{i[:3]}-{(len(i) - 3) * "*"}'
        else:
            return i[:3] + '-****-' + i[7:]

    def point_check(self):
        now_label_text = self.user_number_label.text()
        if len(now_label_text) == 11:
            self.data_signal.emit(now_label_text)
            self.close()
        else:
            self.close()
            # DB 경로 수정됨 (DB_PATH 사용)
            con = sqlite3.connect(DB_PATH)
            df = pd.read_sql('select * from order_table', con)
            df.to_sql('order_table', con, if_exists='replace', index=False)
            con.commit()
            con.close()
            print('포인트 적립 확인')

class MSG_Dialog(QDialog, msg_box_class):
    """메세지 박스 다이얼로그"""
    data_signal = pyqtSignal(str)

    def __init__(self, parent, page_data):
        super().__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.sign = page_data
        self.stackedWidget.setCurrentWidget(self.one_btn_page)

        if page_data == 1:
            self.info_label.setText("메뉴가 품절이라 선택하실 수 없습니다.")
        elif page_data == 2:
            self.info_label.setText("메뉴를 1개 이상 선택하셔야 합니다.")
        elif page_data == 3:
            self.info_label.setText(f"결제중입니다.. 5초 후에 창이 닫힙니다.")
            self.remain_time = 5
            self.p_timer = QTimer()
            self.p_timer.timeout.connect(self.update_p_timer)
            self.p_timer.setInterval(1000)
            self.p_timer.start()
        elif page_data == 4:
            self.info_label.setText("포인트 적립을 하시겠습니까?")
            self.stackedWidget.setCurrentWidget(self.two_btn_page)
        elif page_data == 5:
            self.info_label.setText("회원정보가 사라집니다. 계속하시겠습니까?")
            self.stackedWidget.setCurrentWidget(self.two_btn_page)
        elif page_data == 6:
            self.info_label.setText("유효한 카드번호가 아닙니다. 다시 입력하세요.")
        elif page_data == 7:
            self.info_label.setText("이미 KT할인이 적용되었습니다.")
        elif page_data == 8:
            self.info_label.setText("영수증 출력을 하시겠습니까?")
            self.stackedWidget.setCurrentWidget(self.two_btn_page)

        self.ok_btn.clicked.connect(self.check_and_close)
        self.no_btn.clicked.connect(self.check_no_btn_and_close)
        self.yes_btn.clicked.connect(self.show_num_keypad)

    def check_and_close(self):
        if self.sign == 8:
            print('영수증 드릴게')
        else:
            self.close()

    def check_no_btn_and_close(self):
        if self.sign == 4:
            self.close()
            msg_box_page = MSG_Dialog(self.parent, 8)
            msg_box_page.exec_()
        if self.sign == 8:
            self.close()
            self.parent.stackedWidget.setCurrentWidget(self.parent.opening_page)
            self.parent.timer.start()
            self.parent.delete_order_table_values()
        else:
            self.close()

    def update_p_timer(self):
        self.remain_time -= 1
        if self.remain_time == 0:
            self.p_timer.stop()
            self.close()
            self.show_point_msg_box()
        self.info_label.setText(f"결제중입니다.. {self.remain_time}초 후에 창이 닫힙니다.")

    def show_point_msg_box(self):
        msg_box_page = MSG_Dialog(self.parent, 4)
        msg_box_page.exec_()

    def show_num_keypad(self):
        if not self.sign == 8:
            self.close()
            key_page = Point_Page()
            key_page.data_signal.connect(self.get_label_text)
            key_page.show()
            key_page.exec_()
        else:
            self.parent.order_num += 1
            rept = Rept(self.parent, self.parent.order_num, self.parent.get_total_price() )
            rept.show()
            rept.exec_()
            self.close()
            self.parent.stackedWidget.setCurrentWidget(self.parent.opening_page)
            self.parent.timer.start()
            self.parent.delete_order_table_values()

    def get_label_text(self, text):
        print(text)

class Option_Class(QDialog, choose_option_class):
    """선택옵션 창"""
    data_signal = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setupUi(self)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.move(30, 40)

        data = parent.send_info
        self.drink_name = data['menu_name_x'].to_string(index=False)
        self.drink_price = data['price'].to_string(index=False)
        self.drink_info = data['info'].to_string(index=False)

        # 이미지 경로도 절대 경로로 변환 고려 (DB에 저장된 경로가 상대 경로라면 수정 필요)
        # 일단 그대로 둡니다.
        self.menu_photo_label.setPixmap(QPixmap(data['img_path'].to_string(index=False)))
        self.menu_name_label.setText(str(self.drink_name))
        self.menu_info_label.setText(str(self.drink_info))
        self.menu_price_label.setText(str(self.drink_price) + '원')

        option_df = data.loc[:, 'cinnamon':'zero_cider_changed']
        option_df_dict = option_df.to_dict('list')
        option_df_keys = list(option_df_dict.keys())
        option_df_keys.append('decaffein')
        
        option_df_dict_not_null = {key: [int(x) for x in value[0].split(',')] for key, value in option_df_dict.items()
                                   if value != ['0']}

        if '디카페인' in self.drink_name:
            option_df_dict_not_null['decaffein'] = 1
            if 'strong_or_weak' in option_df_dict_not_null:
                del option_df_dict_not_null['strong_or_weak']

        self.option_frame_list = [getattr(self, f'option_frame_{frame}') for frame in range(1, 13)]

        for idx, key in enumerate(option_df_keys):
            if key in list(option_df_dict_not_null.keys()):
                self.option_frame_list[idx].setVisible(True)
            else:
                self.option_frame_list[idx].setVisible(False)

        self.btn_duplicates_check()

        self.option_buttons = self.option_bottom_frame.findChildren(QPushButton)
        for option_btn in self.option_buttons:
            option_btn.clicked.connect(self.set_extra_charge)
            if option_btn.isChecked():
                self.set_extra_charge()

        self.cancel_btn.clicked.connect(lambda x: self.close())
        self.cancel_btn.clicked.connect(self.close)
        self.order_btn.clicked.connect(self.order_confirm)
        self.reset_btn.clicked.connect(self.btn_duplicates_check)

        # DB 연결 수정됨
        self.con = sqlite3.connect(DB_PATH)

    def set_extra_charge(self):
        # CSV 경로 수정됨
        csv_path = os.path.join(DATA_DIR, 'drinks_price.csv')
        option_price = pd.read_csv(csv_path)
        option_price_eng_name = option_price['eng_name'].to_list()

        add_price = 0
        self.customer_order_option_list = []
        self.customer_order_option_list_kor = []
        self.option_buttons = self.option_bottom_frame.findChildren(QPushButton)

        for btn in self.option_buttons:
            if btn.isChecked() and btn.isVisible():
                btn_object_name = btn.objectName()
                idx = option_price_eng_name.index(btn_object_name)
                drinks_price = option_price.loc[idx, 'noraml_drink']
                drinks_option_name = option_price.loc[idx, 'eng_name']
                drinks_option_kor_name = option_price.loc[idx, 'kor_name']
                add_price += drinks_price
                if '안함' not in drinks_option_kor_name:
                    self.customer_order_option_list_kor.append(drinks_option_kor_name)
                self.customer_order_option_list.append(drinks_option_name)

        self.update_drink_price = str(int(self.drink_price) + int(add_price))
        self.menu_price_label.setText(self.update_drink_price + '원')
        if len(self.customer_order_option_list_kor) != 0:
            self.choose_option_label.setText(str(', '.join(self.customer_order_option_list_kor)))
        else:
            self.choose_option_label.setText('없음')

    def btn_duplicates_check(self):
        self.option_button_groups = []
        for i in range(1, 13):
            option_frame = self.option_frame_list[i - 1]
            buttons = option_frame.findChildren(QPushButton)
            button_group = QButtonGroup()
            button_group.setExclusive(True)

            for btn in buttons:
                button_group.addButton(btn)
            button_group.buttonClicked.connect(self.btn_check)
            button_group.buttonClicked.connect(self.btn_clicked_style)
            self.option_button_groups.append(button_group)

        for btn_group in self.option_button_groups:
            btn_group.buttons()[0].click()

    def btn_clicked_style(self, btn):
        for btn_group in self.option_button_groups:
            if btn in btn_group.buttons():
                for button in btn_group.buttons():
                    button.setStyleSheet('')
        btn.setStyleSheet('border: 3px solid rgb(229, 79, 65);')

    def btn_check(self):
        sender = self.sender()
        for button_group in self.option_button_groups:
            if sender not in button_group.buttons():
                button_group.setExclusive(True)

    def close(self):
        self.parent.remove_label()
        self.accept()

    def order_confirm(self):
        self.parent.drink_num += 1
        option_str = str(self.customer_order_option_list)
        cur = self.con.cursor()
        cur.execute("INSERT INTO order_table (id, drink_cnt, order_drink, price, custom_option)"
                    "VALUES(?,?,?,?,?);", 
                    (self.parent.drink_num, 1, self.drink_name, self.update_drink_price, option_str))
        self.con.commit()

        add_shopping_item_to_listwidget(
            self.parent.drinks_cart_list_widget, str(self.parent.drink_num),
            self.drink_name, self.update_drink_price, self.parent.menu_cnt_label, self.parent.payment_admit_btn)

        cur.execute('SELECT SUM(drink_cnt) FROM order_table')
        result = cur.fetchone()[0]
        self.parent.menu_cnt_label.setText(str(result) + '개')

        order_df = pd.read_sql('select * from order_table', self.con)
        order_df['drink_cnt'] = order_df['drink_cnt'].astype(int)
        order_df['price'] = order_df['price'].astype(int)
        total_price = (order_df['drink_cnt'] * order_df['price']).sum()
        self.parent.payment_admit_btn.setText(f'  {str(total_price)}원\n  결제하기')

        cur.close()
        self.con.close()
        self.parent.remove_label()
        self.accept()
        self.close()

class WindowClass(QMainWindow, main_page_class):
    """오픈화면 & 메인화면 창"""
    clicked = pyqtSignal()

    def add_page_mouse_press(self, event):
        self.stackedWidget.setCurrentWidget(self.main_page)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self._reset_cart_on_start()

        self.comm_thread = CommThread()
        self.comm_thread.received_data.connect(self.handle_server_command)
        self.comm_thread.start()

        self.stackedWidget.setCurrentIndex(0)
        self.set_ad_image()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.move(10,30)

        self.ad_label.mousePressEvent = lambda event: (self.stackedWidget.setCurrentWidget(self.main_page))

        # DB 경로 수정됨 (DB_PATH 사용)
        con = sqlite3.connect(DB_PATH)
        self.price_df = pd.read_sql('select * from drinks_price', con)
        self.menu_df = pd.read_sql('select * from drinks_menu', con)
        self.img_path_df = pd.read_sql('select * from drinks_img_path', con)
        self.order_table_df = pd.read_sql('select * from order_table', con)
        self.drink_num = 0
        self.order_num = 100

        self.DURATION_INT = 120
        self.remaining_time = self.DURATION_INT

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.timer.setInterval(1000)

        self.category_stackedWidget.setCurrentWidget(self.category_1)
        self.category_btn_list = [getattr(self, f"category_btn_{i}") for i in range(1, 16)]
        for btn in self.category_btn_list:
            btn.clicked.connect(self.set_categroy_num)
            btn.clicked.connect(self.change_categroy_btn_color)
            btn.clicked.connect(self.show_menu_arrow_btn)
            btn.clicked.connect(lambda: self.menu_stackedWidget.setCurrentWidget(self.page_1))
            btn.clicked.connect(
                lambda x, category=btn: self.start_timer(btn))

        self.category_right_btn.clicked.connect(lambda: self.check_current_page(1))
        self.category_right_btn_2.clicked.connect(lambda: self.check_current_page(2))
        self.category_left_btn.clicked.connect(lambda: self.check_current_page(1))
        self.category_left_btn_2.clicked.connect(lambda: self.check_current_page(2))

        self.category_btn_1.click()
        self.menu_arrow_btn_num = 2
        self.menu_left_btn.clicked.connect(lambda: self.menu_stackedWidget.setCurrentWidget(self.page_1))
        self.menu_right_btn.clicked.connect(lambda: self.menu_stackedWidget.setCurrentWidget(self.page_2))
        self.menu_right_btn.clicked.connect(self.show_menu_arrow_btn)
        self.menu_left_btn.clicked.connect(self.show_menu_arrow_btn)

        self.menu_frame_list = [getattr(self, f"menu_frame_{i}") for i in range(1, 25)]
        for frame in self.menu_frame_list:
            frame.mousePressEvent = lambda event, name=frame.objectName(): self.click_frame(event, name)

        self.all_remove_label.clicked.connect(self.delete_order_table_values)
        self.home_button.clicked.connect(
            lambda: self.stackedWidget.setCurrentWidget(self.opening_page))

        self.logo_label.mousePressEvent = lambda event: self.open_manager_page()
        self.manager_page_num = 0

        self.menu_price_label_list = [getattr(self, f"menu_price_label_{i}") for i in range(1, 25)]
        for label in self.menu_price_label_list:
            label.setStyleSheet('color: rgb(229, 79, 64);font: 63 12pt "Pretendard SemiBold";')

        self.payment_admit_btn.clicked.connect(self.move_to_order_check_page)

        self.cancel_btn_2.clicked.connect(self.timer_restart_and_go_to_main_page)
        self.back_to_main_page_btn.clicked.connect(self.timer_restart_and_go_to_main_page)
        self.eat_here_btn.clicked.connect(lambda: self.move_to_payment_choose('for_here'))
        self.take_out_btn.clicked.connect(lambda: self.move_to_payment_choose('to_go'))

        self.payment_choose_signal()
        self.cancel_btn.clicked.connect(
            lambda: self.stackedWidget.setCurrentWidget(self.order_check_page))
        self.kt_discount = False

        self.cancel_btn_3.clicked.connect(
            lambda: self.stackedWidget.setCurrentWidget(self.payment_choose_page))
        self.cancel_btn_4.clicked.connect(
            lambda: self.stackedWidget.setCurrentWidget(self.payment_choose_page))
        self.card_img_frame.mousePressEvent = lambda event: self.mobile_pay_msgbox()
        self.barcode_type = None
        
        self.use_coupon.clicked.connect(self.askRcpt)
        self.cancel_btn_5.clicked.connect(
            lambda: self.stackedWidget.setCurrentWidget(self.payment_choose_page))
        self.order_btn_2.clicked.connect(self.check_discount_and_move)

        keyboard_buttons = self.keyboard_frame.findChildren(QPushButton)
        for btn in keyboard_buttons:
            btn.clicked.connect(self.change_card_num)

        # 이미지 경로 수정 (절대경로)
        self.qr_check_frame.setCursor(QCursor(QPixmap(os.path.join(IMG_DIR, 'qt자료/bacord')).scaled(80, 80)))
        self.card_label.setCursor(QCursor(QPixmap(os.path.join(IMG_DIR, 'qt자료/payment_phone.png')).scaled(120, 100)))
        self.horizontalSlider.setCursor(QCursor(QPixmap(os.path.join(IMG_DIR, 'qt자료/matercard.png')).scaled(80, 70)))

    def _reset_cart_on_start(self):
        self.drinks_cart_list_widget.clear()
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        try:
            cur.execute("DELETE FROM order_table")
            con.commit()
        except Exception:
            pass
        finally:
            cur.close()
            con.close()
        self.menu_cnt_label.setText("0개")
        self.payment_admit_btn.setText("  0원\n  결제하기")
        self.drink_num = 0

    def handle_server_command(self, data):
        print(f"[Main] Server Command Received: {data}")
        command = data.strip()
        if not command or command.lower() == "none" or command.startswith("참고:"):
            return

        if command.startswith("{") and command.endswith("}"):
            try:
                payload = json.loads(command)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                action_type = str(payload.get("type", "")).strip()
                menu_name = str(payload.get("menu_name", "")).strip()
                temperature = str(payload.get("temperature", "")).strip().upper()
                try:
                    quantity = int(payload.get("quantity", 1) or 1)
                except ValueError:
                    quantity = 1
                if action_type == "checkout_confirm":
                    self.stackedWidget.setCurrentWidget(self.payment_choose_page)
                    return
                if action_type == "checkout_cancel":
                    self.stackedWidget.setCurrentWidget(self.order_check_page)
                    return
                if action_type == "checkout_request":
                    self.move_to_order_check_page()
                    return
                if action_type == "reset":
                    self._reset_cart_on_start()
                    self.stackedWidget.setCurrentWidget(self.opening_page)
                    return
                if action_type == "open_main":
                    self.stackedWidget.setCurrentWidget(self.main_page)
                    return
                if action_type:
                    self._apply_cart_action(action_type, menu_name, quantity, temperature)
                    return
        
        try:
            def normalize(text: str) -> str:
                # 공백/특수문자 제거, 자주 붙는 말 제거
                cleaned = re.sub(r"[^0-9A-Za-z가-힣]", "", str(text))
                for filler in ["해주세요", "해주세요요", "해줘요", "해줘", "주세요", "주세요요", "줘요", "줘", "한잔", "잔", "하나", "개"]:
                    cleaned = cleaned.replace(filler, "")
                return cleaned.replace(" ", "")

            normalized_command = normalize(command)
            target_menu = self.menu_df[
                self.menu_df['menu_name'].apply(
                    lambda name: normalize(name) in normalized_command or normalized_command in normalize(name)
                )
            ]
            
            if not target_menu.empty:
                row = target_menu.iloc[0]
                drink_name = row['menu_name']
                drink_price = str(row['price'])
                
                self.drink_num += 1
                option_str = "[]" 
                
                # DB 경로 수정됨
                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()
                
                cur.execute("INSERT INTO order_table (id, drink_cnt, order_drink, price, custom_option)"
                            "VALUES(?,?,?,?,?);", 
                            (self.drink_num, 1, drink_name, drink_price, option_str))
                con.commit()
                
                add_shopping_item_to_listwidget(
                    self.drinks_cart_list_widget, str(self.drink_num),
                    drink_name, drink_price, self.menu_cnt_label, self.payment_admit_btn)
                
                cur.execute('SELECT SUM(drink_cnt) FROM order_table')
                result = cur.fetchone()[0]
                self.menu_cnt_label.setText(str(result) + '개')
                
                order_df = pd.read_sql('select * from order_table', con)
                order_df['drink_cnt'] = order_df['drink_cnt'].astype(int)
                order_df['price'] = order_df['price'].astype(int)
                total_price = (order_df['drink_cnt'] * order_df['price']).sum()
                self.payment_admit_btn.setText(f'  {str(total_price)}원\n  결제하기')
                
                cur.close()
                con.close()
                print(f"[Main] '{drink_name}' added to cart via voice.")
            else:
                print(f"[Main] Cannot find menu: {command}")
        except Exception as e:
            print(f"[Main] Error processing voice command: {e}")

    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z가-힣]", "", str(text))
        return cleaned.replace(" ", "")

    def _normalize_order_drink(self, text: str) -> str:
        # "(ICE)" 같은 온도 표기를 제거하고 비교용으로 정규화
        base = re.sub(r"\([^)]*\)", "", str(text))
        return self._normalize_text(base)

    def _find_menu_row(self, menu_name: str, temperature: str = ""):
        norm = self._normalize_text(menu_name)
        if not norm:
            return None
        wants_decaf = "디카페인" in menu_name
        if temperature:
            matches = self.menu_df[self.menu_df["degree"].str.upper() == temperature]
        else:
            matches = self.menu_df

        def is_match(name: str) -> bool:
            name_norm = self._normalize_text(name)
            return norm == name_norm or norm in name_norm or name_norm in norm

        matches = matches[matches["menu_name"].apply(is_match)]
        if not wants_decaf:
            matches = matches[~matches["menu_name"].str.contains("디카페인", na=False)]

        if matches.empty:
            return None
        exact = matches[matches["menu_name"].apply(lambda n: self._normalize_text(n) == norm)]
        if not exact.empty:
            return exact.iloc[0]
        # 부분 매칭은 이름이 짧은 메뉴를 우선
        matches = matches.assign(_len=matches["menu_name"].str.len()).sort_values("_len")
        return matches.iloc[0]

    def _refresh_cart_from_db(self):
        self.drinks_cart_list_widget.clear()
        con = sqlite3.connect(DB_PATH)
        order_df = pd.read_sql("select * from order_table", con)
        con.close()

        if order_df.empty:
            self.menu_cnt_label.setText("0개")
            self.payment_admit_btn.setText("  0원\n  결제하기")
            self.drink_num = 0
            return

        order_df["drink_cnt"] = order_df["drink_cnt"].fillna(0).astype(int)
        order_df["price"] = order_df["price"].fillna(0).astype(int)

        total_count = int(order_df["drink_cnt"].sum())
        total_price = int((order_df["drink_cnt"] * order_df["price"]).sum())

        max_id = 0
        for _, row in order_df.iterrows():
            idx = str(row["id"])
            name = str(row["order_drink"])
            price = str(row["price"])
            add_shopping_item_to_listwidget(
                self.drinks_cart_list_widget, idx, name, price, self.menu_cnt_label, self.payment_admit_btn
            )
            item = self.drinks_cart_list_widget.item(self.drinks_cart_list_widget.count() - 1)
            widget = self.drinks_cart_list_widget.itemWidget(item)
            if widget:
                widget.quantity_label.setText(str(int(row["drink_cnt"])))
                widget.price_label.setText(str(int(row["price"]) * int(row["drink_cnt"])) + "원")
            try:
                max_id = max(max_id, int(idx))
            except ValueError:
                pass

        self.menu_cnt_label.setText(f"{total_count}개")
        self.payment_admit_btn.setText(f"  {total_price}원\n  결제하기")
        self.drink_num = max_id

    def _apply_cart_action(self, action_type: str, menu_name: str, quantity: int, temperature: str = ""):
        if action_type not in ("add", "inc", "dec", "remove", "set", "reset"):
            return
        if action_type in ("add", "inc", "dec", "remove") and not menu_name:
            return

        con = sqlite3.connect(DB_PATH)
        order_df = pd.read_sql("select * from order_table", con)

        if order_df.empty:
            order_df = pd.DataFrame(
                columns=["id", "customer_id", "drink_cnt", "order_drink", "price", "custom_option", "for_here_or_to_go", "discount_price"]
            )

        def match_menu(name):
            norm_name = self._normalize_order_drink(name)
            norm_menu = self._normalize_text(menu_name)
            if "디카페인" not in menu_name and "디카페인" in name:
                return False
            if "디카페인" in menu_name and "디카페인" not in name:
                return False
            if temperature:
                if temperature not in name.upper():
                    return False
            return norm_menu == norm_name or norm_menu in norm_name or norm_name in norm_menu

        matches = order_df[order_df["order_drink"].apply(match_menu)] if not order_df.empty else pd.DataFrame()

        if action_type == "reset":
            self._reset_cart_on_start()
            return
        if action_type in ("add", "inc", "set"):
            if not matches.empty:
                idx = matches.index[0]
                if action_type == "set":
                    order_df.at[idx, "drink_cnt"] = max(1, quantity)
                else:
                    order_df.at[idx, "drink_cnt"] = int(order_df.at[idx, "drink_cnt"]) + max(1, quantity)
            else:
                menu_row = self._find_menu_row(menu_name, temperature)
                if menu_row is None:
                    con.close()
                    return
                display_name = menu_row["menu_name"]
                if temperature:
                    display_name = f"{display_name} ({temperature})"
                new_id = 1
                try:
                    new_id = int(order_df["id"].astype(int).max()) + 1 if not order_df.empty else 1
                except Exception:
                    pass
                new_row = {
                    "id": str(new_id),
                    "customer_id": "",
                    "drink_cnt": max(1, quantity),
                    "order_drink": display_name,
                    "price": str(menu_row["price"]),
                    "custom_option": "[]",
                    "for_here_or_to_go": "",
                    "discount_price": "0",
                }
                order_df = pd.concat([order_df, pd.DataFrame([new_row])], ignore_index=True)
        elif action_type == "dec":
            if matches.empty:
                con.close()
                return
            idx = matches.index[0]
            new_cnt = int(order_df.at[idx, "drink_cnt"]) - max(1, quantity)
            if new_cnt <= 0:
                order_df = order_df.drop(index=idx)
            else:
                order_df.at[idx, "drink_cnt"] = new_cnt
        elif action_type == "remove":
            if matches.empty:
                con.close()
                return
            order_df = order_df.drop(index=matches.index)

        order_df.to_sql("order_table", con, if_exists="replace", index=False)
        con.commit()
        con.close()

        self._refresh_cart_from_db()

    def askRcpt(self):
        try:
            card_num = self.table_widget_qr_code.item(0, 0).text()
        except AttributeError:
            card_num = ''

        if len(card_num) > 0:
            msg_box_page = MSG_Dialog(self, 8)
            msg_box_page.exec_()

    def check_discount_and_move(self):
        d_price = self.get_discount_price()
        t_price = self.get_total_price()
        self.r_price = t_price - d_price
        self.total_payment_price.setText(f'  주문금액: {str(t_price)}원 - 할인금액:{str(d_price)}원')
        self.total_payment_price_2.setText(f'  결제금액: {str(self.r_price)}원')
        self.payment_choose_title_bar.setText(f'    결제수단 선택({str(self.r_price)})원')
        self.stackedWidget.setCurrentWidget(self.payment_choose_page)

    def mobile_pay_msgbox(self):
        msg_box_page = MSG_Dialog(self, 3)
        msg_box_page.exec_()

    def set_table_widget(self, tablewidget, row, column, labels):
        self.table_widget_qr_code.setRowCount(row)
        self.table_widget_qr_code.setColumnCount(column)
        self.table_widget_qr_code.setVerticalHeaderLabels(labels)
        self.table_widget_qr_code.horizontalHeader().setVisible(False)
        self.table_widget_qr_code.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        if self.barcode_type == 'qr_payment':
            self.table_widget_qr_code.setItem(4, 0, QtWidgets.QTableWidgetItem(
                f"{str(int(self.get_total_price()) -self.get_discount_price())}원"))

    def change_card_num(self):
        sender_name = self.sender().text()
        if self.barcode_type == 'kt_discount':
            card_num_row = 3
        if self.barcode_type == 'qr_payment':
            card_num_row = 2
        else:
            card_num_row = 1
        self.insert_value_in_tablewidget(self.table_widget_qr_code, sender_name, card_num_row)

    def insert_value_in_tablewidget(self, tablewidget, sender_obj, card_row):
        keypad_numbers = [str(num) for num in range(10)]
        keypad_numbers.extend(['00', '000'])

        try:
            card_num = tablewidget.item(card_row, 0).text()
        except AttributeError:
            card_num = ''
        string = (card_num + sender_obj).replace('-', '')

        if sender_obj in keypad_numbers and len(string) <= 16:
            string = (card_num + sender_obj).replace('-', '')
            divided_4_letter = '-'.join([string[i:i + 4] for i in range(0, len(string), 4)])
            tablewidget.setItem(card_row, 0, QtWidgets.QTableWidgetItem(divided_4_letter))
        elif sender_obj == 'clear':
            tablewidget.setItem(card_row, 0, QtWidgets.QTableWidgetItem(''))
        elif sender_obj == '승인':
            if card_num.replace('-', '') == '1111222233334444':
                if self.barcode_type == 'kt_discount':
                    kt_info = ['KT할인', '특정할인', '20230601 ~ 20230930', card_num, str(self.get_total_price()) + '원',
                               str(self.get_discount_price()) + '원']
                    self.set_number_in_qr_payment_table(6, kt_info)
                if self.barcode_type == 'qr_payment':
                    coupon_info = [str(self.get_total_price() - self.get_discount_price()) + '원', str(randint(1, 3)), card_num]
                    self.set_number_in_qr_payment_table(3, coupon_info)
            else:
                msg_box_page = MSG_Dialog(self, 6)
                msg_box_page.exec_()
        elif sender_obj == '←':
            card_num = card_num[:-1]
            tablewidget.setItem(card_row, 0, QtWidgets.QTableWidgetItem(card_num))
            tablewidget.item(card_row, 0).setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        else:
            pass

    def set_number_in_qr_payment_table(self, num, value):
        nums = [n for n in range(num)]
        for i in nums:
            self.table_widget_qr_code.setItem(nums[i], 0, QtWidgets.QTableWidgetItem(value[i]))

    def get_discount_price(self):
        # DB 경로 수정됨
        con = sqlite3.connect(DB_PATH)
        order_df = pd.read_sql('select * from order_table', con)
        if self.barcode_type == 'kt_discount':
            order_df.loc[0, 'discount_price'] = 1900

        order_df.to_sql('order_table', con, if_exists='replace', index=False)
        con.commit()
        con.close()
        discount_p = order_df.loc[0, 'discount_price']
        if discount_p == None:
            discount_p = 0
        return discount_p

    def payment_choose_signal(self):
        # CSV 경로 수정됨
        csv_path = os.path.join(DATA_DIR, 'payment_choose.csv')
        payment_btn_df = pd.read_csv(csv_path)
        payment_choose_buttons = self.payment_choose_main_widget.findChildren(QPushButton)
        for btn in payment_choose_buttons:
            con1 = payment_btn_df['btn_name'] == btn.objectName()
            crs_btn = payment_btn_df.loc[con1, ['kor_name', 'type']].to_dict('list')
            btn.clicked.connect(
                lambda x, y=crs_btn['kor_name'][0], z=crs_btn['type'][0]: self.move_to_payment_page(y, z))

    def move_to_payment_page(self, name, type):
        if type == 1:
            self.payment_card_title_bar.setText("  " + name)
            self.stackedWidget.setCurrentWidget(self.charge_page)
            self.update_card_payment_table()
        elif type == 2 or (type == 3 and not self.kt_discount) or type == 4:
            self.move_to_payment_page_for_qr(name, type)
        else:
            msg_box_page = MSG_Dialog(self, 7)
            msg_box_page.exec_()

    def move_to_payment_page_for_qr(self, name, type):
        self.table_widget_qr_code.clearContents()
        self.table_widget_qr_code.setRowCount(0)

        if type == 2:
            self.qr_check_btns_stackwidget.setCurrentWidget(self.coupon_check_btn)
            lab = ['쿠폰번호', '쿠폰명칭', '잔여금액', '받을금액', '결제금액']
            self.barcode_type = 'coupon_payment'

        if type == 4:
            self.qr_check_btns_stackwidget.setCurrentWidget(self.coupon_check_btn)
            lab = ['총 결제금액', '할부개월', '카드번호']
            self.barcode_type = 'qr_payment'

        else:
            self.qr_check_btns_stackwidget.setCurrentWidget(self.kt_check_btn)
            lab = ['제휴사명', '할인종료', '유효기간', '카드번호', '대상금액', '할인금액']
            self.barcode_type = 'kt_discount'
            self.kt_discount = True

        self.set_table_widget(self.table_widget_qr_code, len(lab), 1, lab)
        self.barcord_payment_title_bar.setText("  " + name)
        self.stackedWidget.setCurrentWidget(self.barcod_payment_page)

    def get_total_price(self):
        # DB 경로 수정됨
        con = sqlite3.connect(DB_PATH)
        order_table_df = pd.read_sql('select * from order_table', con)
        order_table_df['drink_cnt'] = order_table_df['drink_cnt'].astype(int)
        order_table_df['price'] = order_table_df['price'].astype(int)
        total_price = (order_table_df['drink_cnt'] * order_table_df['price']).sum()
        return total_price

    def get_total_cnt(self):
        # DB 경로 수정됨
        con = sqlite3.connect(DB_PATH)
        order_table_df = pd.read_sql('select * from order_table', con)
        total_count = order_table_df['drink_cnt'].sum()
        return total_count

    def update_card_payment_table(self):
        total_price = self.get_total_price()
        discount_price = self.get_discount_price()
        f_price = total_price - discount_price
        card_num = self.make_random_card_num()

        self.card_payment_table_widget.setRowCount(3)
        self.card_payment_table_widget.setColumnCount(1)
        self.card_payment_table_widget.horizontalHeader().setVisible(False)
        self.card_payment_table_widget.setItem(0, 0, QtWidgets.QTableWidgetItem(str(f_price) + '원'))
        self.card_payment_table_widget.setItem(1, 0, QtWidgets.QTableWidgetItem('0개월'))
        self.card_payment_table_widget.setItem(2, 0, QtWidgets.QTableWidgetItem(card_num))
        self.card_payment_table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def make_random_card_num(self):
        random_card_num = [str(randint(1000, 9999)) for _ in range(4)]
        random_card_num_for_print = ' '.join(random_card_num)
        mask_card_num = "*" * (len(random_card_num_for_print) - 4) + random_card_num_for_print[-4:]
        return mask_card_num

    def insert_img_for_recommend(self):
        menu_and_price_join_df = pd.merge(self.menu_df, self.img_path_df, on='id')
        con1 = (menu_and_price_join_df['category'] == '디저트')

    def move_to_payment_choose(self, state):
        # DB 경로 수정됨
        con = sqlite3.connect(DB_PATH)
        order_table_df = pd.read_sql('select * from order_table', con)
        order_table_df.loc[:, 'for_here_or_to_go'] = state
        order_table_df.to_sql('order_table', con, if_exists='replace', index=False)
        con.commit()
        con.close()
        self.stackedWidget.setCurrentWidget(self.payment_choose_page)

    def timer_restart_and_go_to_main_page(self):
        self.timer.start()
        self.stackedWidget.setCurrentWidget(self.main_page)

    def fill_the_table_widget(self, table):
        # DB 경로 수정됨
        con = sqlite3.connect(DB_PATH)
        order_table_df = pd.read_sql('select * from order_table', con)
        # CSV 경로 수정됨
        csv_path = os.path.join(DATA_DIR, 'drinks_price.csv')
        price_df = pd.read_csv(csv_path)

        row = order_table_df['id'].count()
        table.setRowCount(row)

        order_table_dict = pd.DataFrame(order_table_df).to_dict()

        for idx in range(row):
            drink_option_df = order_table_df.loc[idx, 'custom_option']
            drink_option_list = ast.literal_eval(drink_option_df)
            option_choices_no_choice = [price_df[price_df['eng_name'] == i]['kor_name'].to_string(index=False)
                                        for i in drink_option_list if 'no_choice' not in i]

            items = [QTableWidgetItem(str(order_table_dict[col][idx])) for col in ['order_drink', 'drink_cnt', 'price']]
            items.append(QTableWidgetItem(','.join(option_choices_no_choice)))

            for item in items:
                item.setTextAlignment(Qt.AlignCenter)

            for col, item in enumerate(items):
                table.setItem(idx, col, item)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

    def move_to_order_check_page(self):
        self.timer.stop()

        # DB 경로 수정됨
        con = sqlite3.connect(DB_PATH)
        order_table_df = pd.read_sql('select * from order_table', con)

        total_price = self.get_total_price()
        total_count = order_table_df['drink_cnt'].sum()
        discount_price = order_table_df['discount_price'].sum()

        self.total_price_for_check_page.setText(str(total_price) + '원')
        self.payment_choose_title_bar.setText(f'  결제수단 선택({str(int(total_price) - int(discount_price))}원)')
        self.total_payment_price.setText(f'  주문금액: {str(total_price)}원 - 할인금액:{str(discount_price)}원')
        self.total_payment_price_2.setText(f'  결제금액: {str(total_price)}원')
        self.total_cnt_for_check_page.setText(str(total_count) + '개')

        if total_count > 0:
            self._send_checkout(order_table_df)
            self.stackedWidget.setCurrentWidget(self.order_check_page)
            self.fill_the_table_widget(self.tableWidget_menu_check)
            self.fill_the_table_widget(self.tableWidget_menu_2_for_qr)
        else:
            msg_box_page = MSG_Dialog(self, 2)
            msg_box_page.exec_()

    def _send_checkout(self, order_table_df):
        items = []
        for _, row in order_table_df.iterrows():
            menu_name = str(row.get("order_drink", "")).strip()
            qty = int(row.get("drink_cnt", 1) or 1)
            price = str(row.get("price", "")).strip()
            if menu_name:
                items.append({"menu_name": menu_name, "quantity": qty, "price": price})
        if items:
            self.comm_thread.send_message({"type": "checkout_preview", "items": items})

    def open_manager_page(self):
        self.manager_page_num += 1
        if self.manager_page_num == 5:
            self.manager_page_num = 0
            manager_page = Manager_Page()
            manager_page.show()
            manager_page.exec_()
            self.stackedWidget.setCurrentWidget(self.opening_page)

    def check_current_page(self, num):
        if num == 1:
            self.category_stackedWidget.setCurrentWidget(self.category_2)
            self.category_btn_11.click()
        elif num == 2:
            self.category_stackedWidget.setCurrentWidget(self.category_1)
            self.category_btn_1.click()
        self.menu_stackedWidget.setCurrentWidget(self.page_1)

    def delete_order_table_values(self):
        self.drinks_cart_list_widget.clear()

        # DB 경로 수정됨
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM 'order_table'")

        cur.execute("SELECT SUM(drink_cnt) FROM 'order_table'")
        result = cur.fetchone()[0]
        if result == None:
            result = 0
        self.menu_cnt_label.setText(str(result) + '개')

        order_df = pd.read_sql('select * from order_table', conn)
        order_df['drink_cnt'] = order_df['drink_cnt'].astype(int)
        order_df['price'] = order_df['price'].astype(int)
        total_price = (order_df['drink_cnt'] * order_df['price']).sum()
        self.payment_admit_btn.setText(f'  {str(total_price)}원\n  결제하기')

        conn.commit()
        cur.close()
        conn.close()

    def start_timer(self, category):
        self.timer.stop()
        self.remaining_time = self.DURATION_INT
        self.timer.start()

    def update_timer(self):
        if self.stackedWidget.currentWidget() == self.main_page:
            self.remaining_time -= 1
            if self.remaining_time == 0:
                self.remaining_time = self.DURATION_INT
                self.stackedWidget.setCurrentWidget(self.main_page)
            self.timer_label.setText(f"{str(self.remaining_time)}초")
        else:
            pass

    def click_frame(self, event, name):
        option_page_df = pd.merge(self.menu_df, self.img_path_df, on='id')
        
        condition1 = (option_page_df['category'] == self.user_clicked_category)
        condition2 = (option_page_df['category_num'] == int(name[11:]))

        sold_out_state = option_page_df.loc[condition1 & condition2, 'sold_out']
        self.send_info = option_page_df.loc[condition1 & condition2]
        
        if sold_out_state.sum() > 0:
            msg_box_page = MSG_Dialog(self, 1)
            msg_box_page.exec_()
        else:
            self.show_sample_label()
            dialog_page = Option_Class(self)
            dialog_page.show()

    def remove_label(self):
        self.sample_label.hide()

    def show_sample_label(self):
        self.sample_label = QLabel(self)
        self.sample_label.setGeometry(0, 0, 768, 1024)
        self.sample_label.setStyleSheet('background-color: rgba(45,45,45,200);')
        self.sample_label.show()

    def show_menu_arrow_btn(self):
        current_page = self.menu_stackedWidget.currentWidget().objectName()
        self.menu_right_btn.setVisible(current_page == 'page_1' and self.menu_arrow_btn_num == 2)
        self.menu_left_btn.setVisible(current_page == 'page_2' and self.menu_arrow_btn_num == 2)

    def set_categroy_num(self):
        menu_frame_list = [getattr(self, f"menu_frame_{i}") for i in range(1, 25)]
        btn_name = self.sender().text()
        self.user_clicked_category = btn_name

        # DB 경로 수정됨
        self.connn = sqlite3.connect(DB_PATH)
        self.menu_df = pd.read_sql('select * from drinks_menu', self.connn)

        category_drinks_num = len(self.menu_df[self.menu_df['category'] == btn_name])

        for index, frame in enumerate(menu_frame_list):
            getattr(self, f'menu_img_{index + 1}').clear()
            getattr(self, f'menu_name_label_{index + 1}').clear()
            getattr(self, f'menu_price_label_{index + 1}').clear()
            if index + 1 <= category_drinks_num:
                frame.setStyleSheet('background-color: white;')
            else:
                frame.setStyleSheet('background-color: rgb(255, 204, 0);')

        self.insert_img(btn_name, category_drinks_num)

    def insert_img(self, btn_name, category_drinks_num):
        menu_and_price_join_df = pd.merge(self.menu_df, self.img_path_df, how='left', on='id')
        con1 = (menu_and_price_join_df['category'] == btn_name)
        user_click_category_df = menu_and_price_join_df.loc[
            con1, ['id', 'category', 'category_num', 'sold_out', 'menu_name_x', 'img_path',
                   'price']]

        con3 = user_click_category_df['sold_out'] == 1
        for i in range(1, category_drinks_num + 1):
            con2 = user_click_category_df['category_num'] == i

            drink_img = user_click_category_df.loc[con2, 'img_path'].values
            drink_name = user_click_category_df.loc[con2, 'menu_name_x'].values
            drink_price = user_click_category_df.loc[con2, 'price'].values
            sold_out_state = user_click_category_df.loc[con2 & con3, 'sold_out'].values

            if sold_out_state.size > 0:
                # 이미지 경로 수정 (절대 경로)
                sold_out_path = os.path.join(IMG_DIR, 'qt자료/sold_out_3.png')
                getattr(self, f'menu_img_{i}').setPixmap(
                    QPixmap(sold_out_path).scaled(175, 180, Qt.IgnoreAspectRatio))
            elif drink_img.size > 0 and drink_name.size > 0 and drink_price.size > 0:
                getattr(self, f'menu_img_{i}').setPixmap(QPixmap(drink_img[0]))
            getattr(self, f'menu_name_label_{i}').setText(str(drink_name[0]))
            getattr(self, f'menu_price_label_{i}').setText(f'{str(drink_price[0])}원')
        self.connn.close()
        
        if category_drinks_num > 12:
            self.menu_arrow_btn_num = 2
        else:
            self.menu_arrow_btn_num = 1

    def change_categroy_btn_color(self):
        btn_object = self.sender()
        for btn in self.category_btn_list:
            if btn == btn_object:
                btn.setStyleSheet('''
                    background-color: rgb(45, 45, 45);
                    border: 2px solid rgb(45, 45, 45);
                    border-radius:15px; color:white; 
                    ''')
            else:
                btn.setStyleSheet('''
                    background-color: rgb(255, 204, 0);
                    border: 2px solid rgb(45, 45, 45);
                    border-radius:15px;
                    color:black;
                    ''')

    def set_ad_image(self):
        self.ad_img_num = 1
        # 이미지 경로 수정 (절대 경로)
        img_path = os.path.join(IMG_DIR, f'ad/ad_img_{self.ad_img_num}')
        self.ad_label.setPixmap(QPixmap(img_path).scaled(QSize(768, 1024)))

        ad_timer = QTimer(self)
        ad_timer.timeout.connect(self.change_ad_image)
        ad_timer.start(3000)

    def change_ad_image(self):
        self.ad_img_num += 1
        if self.ad_img_num == 5:
            self.ad_img_num = 1

        # 이미지 경로 수정 (절대 경로)
        img_path = os.path.join(IMG_DIR, f'ad/ad_img_{self.ad_img_num}')
        pixmap = QPixmap(img_path)
        self.ad_label.setPixmap(QPixmap(pixmap).scaled(QSize(768, 1024)))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
