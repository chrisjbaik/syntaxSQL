from multiprocessing.connection import Client

from query_pb2 import ProtoQueryList, ProtoResult, FALSE, UNKNOWN, TRUE

class DuoquestClient(object):
    def __init__(self, port, authkey):
        self.port = port
        self.authkey = authkey

    def connect(self):
        address = ('localhost', self.port)
        self.conn = Client(address, authkey=self.authkey)

    def should_prune(self, query):
        protolist = ProtoQueryList()
        protolist.queries.append(query.pq)

        self.conn.send_bytes(protolist.SerializeToString())
        msg = self.conn.recv_bytes()
        response = ProtoResult()
        response.ParseFromString(msg)
        return (response.results[0] == FALSE)

    def is_verified(self, query):
        protolist = ProtoQueryList()
        protolist.queries.append(query.pq)

        self.conn.send_bytes(protolist.SerializeToString())
        msg = self.conn.recv_bytes()
        response = ProtoResult()
        response.ParseFromString(msg)
        return (response.results[0] == TRUE), response.answer_found

    def close(self):
        self.conn.send_bytes(b'close')
        self.conn.close()
