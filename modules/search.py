class Query(object):
    def __init__(self, set_op=None):
        # set operation
        self.set_op = set_op    # 'none', 'intersect', 'except', 'union'
        self.left = None        # left subquery
        self.right = None       # right subquery

        # query components
        # states:
        #  - None: if invalid/unknown, or if set_op != 'none'
        #  - False: if isn't in query
        #  - True: in query, but not yet inferred
        #  - Anything else: inferred value
        self.select = True
        self.where = None
        self.group_by = None
        self.having = None
        self.order_by = None
        self.limit = None

    # in format expected by SyntaxSQLNet
    def as_dict(self):
        if self.set_op != 'none':
            sql = self.left.as_dict()
            sql['nested_sql'] = self.right.as_dict()
            sql['nested_label'] = self.set_op
        else:
            sql = {
                'select': self.select
            }

            if isinstance(self.where, list):
                where = []
                for item in self.where:
                    if instanceof(item, Query):
                        where.append(item.as_dict())
                    else:
                        where.append(item)
                sql['where'] = where

            if self.group_by:
                sql['groupBy'] = self.group_by

            if isinstance(self.having, list):
                having = []
                for item in self.having:
                    if instanceof(item, Query):
                        having.append(item.as_dict())
                    else:
                        having.append(item)
                sql['having'] = having

            if self.order_by:
                sql['orderBy'] = self.order_by

        # add another 'sql' layer as prescribed
        sql = {
            'sql': sql
        }

        return sql

    # next is array, e.g. [left, select, ...]
    # returns smallest subquery Query object
    def find_subquery(self, next):
        if next[0] == 'left':
            return self.left.find_subquery(next[1:])
        elif next[0] == 'right':
            return self.right.find_subquery(next[1:])
        elif next[0] == 'where' and isinstance(self.where, list):
            return self.where[-1].find_subquery(next[1:])
        elif next[0] == 'having' and isinstance(self.having, list):
            return self.having[-1].find_subquery(next[1:])
        else:
            return self

class SearchState(object):
    def __init__(self, next, parent=None, history=None, query=None):
        # a chain of keys storing next item to infer, e.g.:
        #   left, select: select in left subquery
        #   left, where: left subquery, where clause, last subquery
        self.next = next

        # parent SearchState if in subquery
        self.parent = None

        # next col index, e.g. for where column predicates
        self.next_col_idx = None
        # columns to iterate for where predicates
        self.iter_cols = None

        # next op index, e.g. for where col predicates
        self.next_op_idx = None
        # ops to iterate for where predicates
        self.iter_ops = None

        # similar for agg
        self.next_agg_idx = None
        self.iter_aggs = None

        if not history:
            history = [["root"]]*2
        self.history = history

        if not query:
            query = Query()
        self.query = query
