import socket

# 创建客户端

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

soc.connect(('192.168.43.22', 1234))
# soc.connect(('127.0.0.1', 1111))
print(1)
# 接受欢迎消息
print(soc.recv(1024).decode('utf-8'))
for data in [b'Michael', b'Tracy', b'Sarah']:
    # 发送数据
    print(2)
    soc.send(data)
    print(3)
    print(soc.recv(1024).decode('utf-8'))
soc.send(b'exit')
soc.close()
