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

    def dev_data(self):
        recv_data = self.soc.recv(1024).decode("UTF-8")
        recv_data = json.loads(recv_data, strict=False)
        return recv_data

    def close(self):
        self.soc.send(b'exit')
        self.soc.close()


wxw = TCPClient('219.216.72.101', 1231)

w1 = 1
with open("agent.txt", "r") as f:
    while True:
        rev_data = wxw.dev_data()  # 等待接收数据
        print(rev_data)  # 打印接收的数据
        data = f.readline()
        if not data:
            break
        data = data.encode("UTF-8").decode("UTF-8")
        drl_data = json.loads(data, strict=False)
        drl_data = [drl_data[0] * 100, drl_data[1] * 100]
        wxw.send_data(drl_data)
wxw.close()
