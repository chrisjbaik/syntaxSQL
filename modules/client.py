from multiprocessing.connection import Client

from duoquest_pb2 import ProtoQueryList, ProtoResult, FALSE, UNKNOWN, TRUE

class StopException(Exception):
    pass

class DuoquestClient(object):
    def __init__(self, host, port, authkey):
        self.host = host
        self.port = port
        self.authkey = authkey
        self.tsq_level = None

    def init_cache(self):
        self.cache = {}

    def connect(self):
        address = (self.host, self.port)
        self.conn = Client(address, authkey=self.authkey)

    def should_prune(self, query):
        cache_key = str(query.pq)
        if cache_key in self.cache:
            return self.cache[cache_key]
        else:
            protolist = ProtoQueryList()
            protolist.queries.append(query.pq)

            self.conn.send_bytes(protolist.SerializeToString())
            msg = self.conn.recv_bytes()
            response = ProtoResult()
            response.ParseFromString(msg)

            result = (response.results[0] == FALSE)
            self.cache[cache_key] = result

            if response.answer_found:
                raise StopException('Early termination triggered.')
        return result

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
