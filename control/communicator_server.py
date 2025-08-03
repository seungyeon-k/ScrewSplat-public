import socket
import time
import pickle
import struct

class Listener():
    def __init__(self, tcp_ip, tcp_port):
        self.BUFFER_SIZE = 368640
        self.tcp_ip = tcp_ip
        self.tcp_port = tcp_port

        self.socket = None
        self.conn   = None

        self.isOpen = False

        self.payload_size = struct.calcsize("L")
        self.data = b''

    def open_connection(self):
        while True:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.bind((self.tcp_ip , self.tcp_port))
                self.socket.listen(1)
                print("[UI => Robot] waiting for connection..")

                self.conn, _ = self.socket.accept()
                print("[UI => Robot] connected")
                self.isOpen = True
                break
            except socket.error as e:
                print("Cannot open socket")
                time.sleep(1)
                continue

    def recv_vision(self):
        if not self.isOpen:
            self.open_connection()

        while len(self.data) < self.payload_size:
            self.data += self.conn.recv(4096)

        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(self.data) < msg_size:
            self.data += self.conn.recv(4096)

        image_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Extract frame
        image = pickle.loads(image_data, encoding='bytes') # encoding='bytes' should be needed for python3
        # image = pickle.loads(image_data)
        return image
    
    def send_grasp(self, grasp):
        data = pickle.dumps(grasp, protocol=2)

        # Send message length first
        message_size = struct.pack("L", len(data))

        # Then data
        self.conn.sendall(message_size + data)

    # def send_dict_data(self, dict_data):
    #     dict_data_dumped = pickle.dumps(dict_data)
    #     self.conn.send(dict_data_dumped)

    def send_status(self, status):
        if self.isOpen:
            try:
                self.conn.send(status.encode())
            except socket.error as e:
                self.close_connection()

    def close_connection(self):
        if self.isOpen:
            self.socket.shutdown(socket.SHUT_RDWR)
            time.sleep(1)
            self.socket.close()
            self.isOpen = False
