from multiprocessing.connection import Client

from query_pb2 import ProtoQueryList, FALSE, UNKNOWN, TRUE

class MixtapeClient(object):
    def __init__(self, port, authkey):
        self.port = port
        self.authkey = authkey

    def connect(self):
        address = ('localhost', self.port)
        self.conn = Client(address, authkey=self.authkey)

    def should_prune(self, query):
        protolist = ProtoQueryList()
        protolist.append(query.to_proto())

        self.conn.send_bytes(protolist.SerializeToString())
        msg = self.conn.recv_bytes()
        response = ProtoResult()
        response.ParseFromString(msg)
        return (response.results[0] == FALSE)

    def close(self):
        self.conn.send_bytes(b'close')
        self.conn.close()
