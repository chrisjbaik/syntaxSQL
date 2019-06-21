import os
import sqlite3

class Database(object):
    def __init__(self, db_path, dataset):
        self.db_path = db_path
        self.dataset = dataset

        if dataset == 'spider':
            self.db_names = os.listdir(self.db_path)
            self.conn = None
        else:
            self.conn = sqlite3.connect(self.db_path)
            cur = self.conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            self.db_names = [row[0] for row in cur.fetchall()]
            cur.close()

    def has_db(self, db_name):
        return db_name in self.db_names

    def get_conn(self, db_name=None):
        if self.dataset == 'spider':
            db_path = os.path.join(self.db_path, db_name,
                '{}.sqlite'.format(db_name))
            conn = sqlite3.connect(db_path)
        else:
            db_path = self.db_path
            conn = self.conn
        return conn

    def find_literals(self, db_name, tbl_name, col_name, str, b):
        if col_name == '*':
            return []

        conn = self.get_conn(db_name)
        cur = conn.cursor()

        q = 'SELECT "{}" FROM "{}" WHERE "{}" LIKE ? ESCAPE \'\\\''.format(
            col_name, tbl_name, col_name
        )
        cur.execute(q, ('%{}%'.format(str.replace('%', '\%')),))

        results = []
        rows = cur.fetchmany(size=b)
        for row in rows:
            results.append(row[0])

        cur.close()
        conn.close()
        return results
