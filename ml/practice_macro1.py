import sys
from PyQt5.QtWidgets import *             # 위젯관련 함수 사용하기
from PyQt5.QtCore import QTimer           # 시간관련 함수 사용하기
import pyautogui                          # pyautogui 함수 사용하기


class MyApp(QWidget):

    def __init__(self):
        super().__init__()                # 부모 클래스 초기화


        self.x, self.y, self.delay, self.num_click = 0, 0, 0, 0     # x, y, 지연, 클릭 횟수 초기화

        self.start_btn_click()          # 자동클릭 하기

    def start_btn_click(self):
        self.timer = QTimer()          # self.timer에 Qtimer() 기능 인스턴스화
        self.x = 800                   # x축 좌표
        self.y = 500                   # y축 좌표
        self.delay = 3                 # 지연(초)
        self.rutine_number = 3         # 반복 횟수

        self.timer.start(self.delay * 1000)             # timer 시작, self.deay만큼 시간 지연 후 종료 그리고 다시 가동
        self.timer.timeout.connect(self.mouse_click)    # timer가 끝나면 self.mouse_click 명령 생성

    def mouse_click(self):
        pyautogui.click(self.x, self.y)                 # pyautogui의 click 함수 사용 위치는 x,y 좌표
        self.num_click += 1                             # timer가 다시 가동될 때 마다 1씩 증가


        if self.num_click == self.rutine_number:        # 클릭 횟수가 self.rutine_number와 동급을 때 멈춰라
            self.timer.stop()


if __name__ == '__main__':           # import된 것들을 실행시키지 않고 __main__에서 실행하는 것만 실행 시킨다.
                                     # 즉 import된 다른 함수의 코드를 이 화면에서 실행시키지 않겠다는 의미이다.

    app = QApplication(sys.argv)     # PyQt5로 실행할 파일명을 자동으로 설정, PyQt5에서 자동으로 프로그램 실행

    ex = MyApp()                     # ex에 MyApp()을 상속시킨다.

    sys.exit(app.exec_())