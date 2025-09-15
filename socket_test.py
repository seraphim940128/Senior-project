import socket
import keyboard  # pip install keyboard
import time

HOST = '127.0.0.1'  # Unity 的 IP
PORT = 5500         # Unity TCP Server 端口

def send_command(command):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(command.encode())
    except ConnectionRefusedError:
        print("Unity 伺服器未啟動")

# 主迴圈，即時偵測按鍵
print("按 i/j/l/u/o 控制角色，按 esc 離開")
while True:
    if keyboard.is_pressed('i'):
        send_command('i')
        time.sleep(0.1)  # 避免過快重複發送
    elif keyboard.is_pressed('j'):
        send_command('j')
        time.sleep(0.1)
    elif keyboard.is_pressed('l'):
        send_command('l')
        time.sleep(0.1)
    elif keyboard.is_pressed('u'):
        send_command('u')
        time.sleep(0.1)
    elif keyboard.is_pressed('o'):
        send_command('o')
        time.sleep(0.1)
    elif keyboard.is_pressed('esc'):
        print("離開程式")
        break
