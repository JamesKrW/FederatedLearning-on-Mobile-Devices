import sys
import socket


class Communication(object):

    def __init__(self, private_ip, public_ip):
        self._private_ip = private_ip.split(':')[0]         # the private(local) IP address
        self._private_port = int(private_ip.split(':')[1])  # the private(local) IP port
        self._public_ip = public_ip.split(':')[0]           # the public(Internet) IP address
        self._public_port = int(public_ip.split(':')[1])    # the public(Internet) IP port

    def start_socket_ps(self):
        """Creates a socket that will act as server.
            Returns:
              sever_socket (socket): ssl secured socket that will act as server.
         """
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # create a socket
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # set the socket to reuse the address for different connections
        server_socket.bind((self._private_ip, self._private_port))          # bind private IP to listen requests
        server_socket.listen(5)                                          # 1000 is the maximum number of connections at the same time
        return server_socket

    def start_socket_client(self):
        """Creates a socket that will act as client.
           Returns:
              sever_socket (socket): ssl secured socket that will work as client.
         """
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   # create a socket
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # 在客户端开启心跳维护
        client_socket.connect((self._public_ip, self._public_port))         # connect to the server's public IP
        return client_socket

    def receiving_subroutine(self, connection_socket):
        timeout = 1.0
        while True:
            ultimate_buffer = b''   # buffer for saving data
            connection_socket.settimeout(240)   # set timeout for the first round receiving operation
            first_round = True
            while True:
                try:
                    receiving_buffer = connection_socket.recv(8192*2)   # try receiving data of SRC.buffer size and save it to receiving_buffer
                except Exception as e:
                    if str(e) != 'timed out':   # if the exception is not timed out, then error may have occured
                        print(e)
                        sys.stdout.flush()
                    break
                if first_round: # if we have just finished the first round with 240s timeout, we set a short timeout for the second round to check if all data has been received.
                    connection_socket.settimeout(timeout)
                    first_round = False
                if not receiving_buffer:    # if we have not received anything, then all data has been transferred
                    break
                ultimate_buffer += receiving_buffer # append the received data in this round to the ultimate_buffer
            if(ultimate_buffer[0:5]==b'10111'):
                message = ultimate_buffer[5:int(0 - int(5))]  # this is the actual massage
                connection_socket.send(b'RECEIVED')
                print('Recived right message!')
            else:
                connection_socket.send(b'ERRORrrr')
                print("Received wrong message!")
                continue
            return message


    def get_message(self, connection_socket):
        message = self.receiving_subroutine(connection_socket)
        return message

    def send_message(self, message_to_send, connection_socket):
        message = b'10111'+ message_to_send + b'EOF\r\n' # create the message with signature and EOF mark
        connection_socket.settimeout(240)       # set timeout for the sendall operation
        connection_socket.sendall(message)      # send the message using the socket connection
        #print('sended!!!!!!!!!!!')
        while True:
            check = connection_socket.recv(8)  # receive status signal
            print(check)
            if check == b'ERRORrrr':              # if error signal is received, retry
                connection_socket.sendall(message)
            elif check == b'RECEIVED':             # message received successfully, end
                break
