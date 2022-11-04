import socket
import os.path
import PIL.Image
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Connecting...')
while True:
    try:
        sock.connect(('192.168.1.35', 16129))
    except Exception as e:
        print(e)
    else:
        break
print('Connected.')
count = 1
begin = b'\x48\x54\x4D\x00'
end = b'\x77\xEE\xEE\x77'
save_path = '/home/nano/nano1/tcdlr/original/'
save_path2= '/home/nano/nano1/tcdlr/result/'
begin_found = False
end_found = False
data_remain = b''
while True:
    data = sock.recv(1024 * 1024)  # 一次接收到的字节数
    if not data:
        break
    i = 0
    if begin_found and not end_found:
        k = 0
        while k < len(data):
            if data[k:k + len(end)] == end:
                end_found = True
                save = data_remain + data[:k]
                save = save[125:125+1024*1024]
                t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
                image: PIL.Image.Image = PIL.Image.frombytes('L', (1024, 1024), save)
                image.save(os.path.join(save_path, f'{count:06}.jpg'))
                image.save(os.path.join(save_path2, f'{t}.jpg'))
                time.sleep(4)

                # with open(os.path.join(save_path, f'{count:06}'), 'wb') as file:
                #     file.write(save)
                count += 1
                break
            k += 1
        if end_found:
            begin_found = False
            end_found = False
            i = k + len(end)
        else:
            data_remain += data
            continue
    while i < len(data):
        if data[i:i+len(begin)] == begin:
            begin_found = True
            k = i+len(begin)
            while k < len(data):
                if data[k:k+len(end)] == end:
                    end_found = True
                    save = data[i+len(begin):k]
                    save = save[125:125 + 1024 * 1024]
                    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
                    image: PIL.Image.Image = PIL.Image.frombytes('L', (1024, 1024), save)
                    image.save(os.path.join(save_path, f'{count:06}.jpg'))
                    image.save(os.path.join(save_path2, f'{t}.jpg'))
                    time.sleep(4)
                    # with open(os.path.join(save_path, f'{count:06}'), 'wb') as file:
                    #     file.write(save)
                    count += 1
                    break
                k += 1
            if end_found:
                begin_found = False
                end_found = False
                i = k + len(end)
                continue
            else:
                data_remain = data[i + len(begin):]
                break
        i += 1
