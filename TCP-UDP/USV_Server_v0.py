import socket
import json
import threading
import time


def send_data(drl):
    eta_data = [2, 1]
    message_json = json.dumps(eta_data, ensure_ascii=False)
    drl.send(message_json.encode('UTF-8'))
    time.sleep(1)


def get_data(drl):
    global ex
    global ey
    while True:
        sock, addr = drl.accept()
        print('Successfully accept new connection')
        while True:
            send_data(sock)  # send data
            data = sock.recv(1024)
            if not data or data.decode('utf-8') == 'exit':
                break
            drl_data = json.loads(data.decode("UTF-8"), strict=False)
            ex, ey = drl_data[0], drl_data[1]
        sock.close()
        print('Connection is closed')


DRL = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
DRL.bind(('219.216.72.101', 1234))  # get aim
DRL.listen(5)

thread = threading.Thread(target=get_data, args=(DRL,))
thread.start()
ex = 0
ey = 0
while 1:
    print(ex, ey)
    time.sleep(2)

