import socket
import threading
import time


def tcplink(sock):
    while True:
        sock, addr = soc.accept()
        print('Accept new connection from:...')
        sock.send(b'Welcome')
        while True:
            data = sock.recv(1024)
            time.sleep(1)
            if not data or data.decode('utf-8') == 'exit':
                print(data.decode('utf-8'))
                break
            sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
        sock.close()
        print('Connection from closed.')


# 创建客户端
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 监听端口
soc.bind(('127.1.1.0', 1234))

print('Waiting for connection...')
soc.listen(5)

while True:
    # 创建新线程来处理TCP连接DRL
    t = threading.Thread(target=tcplink, args=(soc, ))
    t.start()
    while 1:
        print(2)
        time.sleep(1)
