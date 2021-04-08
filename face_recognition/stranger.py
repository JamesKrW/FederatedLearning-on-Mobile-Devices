import socket
ip='127.0.0.1'
port = 55533

def stranger_name(ip='127.0.0.1',port = 55533):
    ip_port=(ip,port)
    sk = socket.socket()
    try:
        #请求连接服务端
        sk.connect(ip_port)
        #发送数据
        sk.send(b"end123456789\n")
        print("send success")
        #接收数据
        server_reply = sk.recv(1024)
        print("receive success")
        #打印接受的数据
        server_reply = str(server_reply,'UTF-8')
        #关闭连接
        sk.close()
        return server_reply
    except:
        server_reply = 'connnection failed'
    print (server_reply)
