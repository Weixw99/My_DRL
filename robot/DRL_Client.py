import socket
import json
import time


class TCPClient:
    def __init__(self, ipp, posts):
        # 创建客户端
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect((ipp, posts))

    def send_data(self, datas):
        data_json = json.dumps(datas, ensure_ascii=False)
        self.soc.send(data_json.encode('UTF-8'))

    def get_data(self):
        recv_data = self.soc.recv(1024).decode("UTF-8")
        recv_data = json.loads(recv_data, strict=False)
        return recv_data

    def close(self):
        self.soc.close()


if __name__ == '__main__':
    wxw = TCPClient('172.22.0.157', 1230)
    w1 = 1
    with open("agent.txt", "r") as f:
        while True:
            rev_data = wxw.get_data()  # 等待接收数据
            print(rev_data)  # 打印接收的数据
            data = f.readline()
            if not data:
                break
            data = data.encode("UTF-8").decode("UTF-8")
            drl_data = json.loads(data, strict=False)
            drl_data = [drl_data[0] * 100, drl_data[1] * 100]
            wxw.send_data(drl_data)
            if w1:
                time.sleep(15)
                w1 = 0
            else:
                time.sleep(5)
    wxw.close()
