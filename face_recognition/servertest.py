import socket  # 导入 socket 模块

s = socket.socket()  # 创建 socket 对象
host = '0.0.0.0'  # 获取本地主机名
port = 12350  # 设置端口
s.bind((host, port))  # 绑定端口
s.listen(5)  # 等待客户端连接

def getmessage():
    c, addr = s.accept()  # 建立客户端连接
    print('连接地址：', addr)
    print('send over')
    while True:
        data = c.recv(1024)
        if data:
            ans = data.decode()
            print("message:", ans)
            break
    c.close()  # 关闭连接
    return ans

for i in range(4):
    print(getmessage())
s.shutdown(socket.SHUT_RDWR)
s.close()