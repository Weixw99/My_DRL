import socket
import json
import threading
import time


class TCPServer:
    def __init__(self, ipp, posts):
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.bind((ipp, posts))
        self.soc.listen(5)
        self.sock = None
        self.addr = None

    def send_data(self):
        eta_data = [0.2, 0.6]
        data_json = json.dumps(eta_data, ensure_ascii=False)
        self.sock.send(data_json.encode('UTF-8'))

    def get_data(self):
        global ex
        global ey
        while True:
            self.sock, self.addr = self.soc.accept()
            print('Successfully accept new connection')
            while True:
                self.send_data()  # send data
                data = self.sock.recv(1024)
                if not data or data.decode('utf-8') == 'exit':
                    break
                drl_data = json.loads(data.decode("UTF-8"), strict=False)
                ex, ey = drl_data[0], drl_data[1]
            self.close()
            print('Connection is closed')

    def close(self):
        self.soc.close()


if __name__ == '__main__':

    DRL = TCPServer('219.216.72.101', 1230)
    thread = threading.Thread(target=DRL.get_data, args=())
    thread.start()
    ex = 0
    ey = 0
    while 1:
        print(ex, ey)
        time.sleep(2)
