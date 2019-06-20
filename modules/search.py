from itertools import permutations

NEW_WHERE_OPS = ('=','>','<','>=','<=','!=','like','not in','in','between')

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

    def copy(self):
        copied = Query()
        copied.set_op = self.set_op
        if copied.left:
            copied.left = self.left.copy()
        if copied.right:
            copied.right = self.right.copy()

        if isinstance(self.select, list):
            copied.select = list(self.select)
        else:
            copied.select = self.select

        if isinstance(self.where, list):
            where = []
            for item in self.where:
                if isinstance(item, Query):
                    where.append(item.copy())
                else:
                    where.append(item)
            copied.where = where
        else:
            copied.where = self.where

        if isinstance(self.group_by, list):
            copied.group_by = list(self.group_by)
        else:
            copied.group_by = self.group_by

        if isinstance(self.having, list):
            having = []
            for item in self.having:
                if isinstance(item, Query):
                    having.append(item.copy())
                else:
                    having.append(item)
            copied.having = having
        else:
            copied.having = self.having

        if isinstance(self.order_by, list):
            copied.order_by = list(self.order_by)
        else:
            copied.order_by = self.order_by

        copied.limit = self.limit
        return copied

    # in format expected by SyntaxSQLNet
    def as_dict(self, sql_key=True):
        if self.set_op != 'none':
            sql = {
                'sql': self.left.as_dict(sql_key=False),
                'nested_sql': self.right.as_dict(sql_key=False),
                'nested_label': self.set_op
            }
        else:
            sql = {
                'select': self.select
            }

            if isinstance(self.where, list):
                where = []
                for item in self.where:
                    if isinstance(item, Query):
                        where.append(item.as_dict())
                    else:
                        where.append(item)
                sql['where'] = where

            if self.group_by:
                sql['groupBy'] = self.group_by

            if isinstance(self.having, list):
                having = []
                for item in self.having:
                    if isinstance(item, Query):
                        having.append(item.as_dict())
                    else:
                        having.append(item)
                sql['having'] = having

            if self.order_by:
                sql['orderBy'] = self.order_by

            # add another 'sql' layer in outermost layer
            if sql_key:
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
        self.parent = parent

        # next column to use (from col_cands)
        self.next_col = None
        # candidate columns for current clause
        self.col_cands = None
        # used candidates for current clause
        self.used_cols = None
        # number of column slots for current clause
        self.num_cols = None

        # next agg to use (from agg_cands)
        self.next_agg = None
        # candidate aggs for current clause
        self.agg_cands = None
        # used candidates for current clause
        self.used_aggs = None
        # number of agg slots for current clause
        self.num_aggs = None

        # index of next op to use from iter_ops
        self.next_op_idx = None
        # ops to use for current clause
        self.iter_ops = None

        if not history:
            history = [['root']] * 2
        self.history = history

        if not query:
            query = Query()
        self.query = query

    def copy(self):
        history_copy = [list(self.history[0])] * 2

        copied = SearchState(self.next, history=history_copy,
            query=self.query.copy())

        if self.parent:
            copied.parent = self.parent.copy()

        copied.next_col = self.next_col
        copied.col_cands = self.col_cands       # will not be modified
        if self.used_cols is not None:
            copied.used_cols = set(self.used_cols)
        copied.num_cols = self.num_cols

        copied.next_agg = self.next_agg
        copied.agg_cands = self.agg_cands       # will not be modified
        if self.used_aggs is not None:
            copied.used_aggs = set(self.used_aggs)
        copied.num_aggs = self.num_aggs

        copied.next_op_idx = self.next_op_idx
        copied.iter_ops = self.iter_ops         # will not be modified

        return copied

    def next_agg_states(self):
        states = []
        for agg in self.agg_cands:
            if agg in self.used_aggs:
                continue
            new = self.copy()
            new.next_agg = agg
            states.append(new)
        return states

    def next_col_states(self):
        states = []
        for col in self.col_cands:
            if col in self.used_cols:
                continue
            new = self.copy()
            new.next_col = col
            states.append(new)
        return states

    def next_op_states(self, next, num_ops, op_cands, col_name):
        states = []
        self.next[-1] = next
        self.next_op_idx = 0
        for ops in permutations(op_cands, num_ops):
            new = self.copy()

            for i, op in enumerate(ops):
                if i != 0:
                    new.history[0].append(col_name)
                new.history[0].append(NEW_WHERE_OPS[op])

            new.iter_ops = ops
            states.append(new)

        return states

    def clear_agg_info(self):
        self.next_agg = None
        self.agg_cands = None
        self.used_aggs = None
        self.num_aggs = None

    def clear_col_info(self):
        self.next_col = None
        self.col_cands = None
        self.used_cols = None
        self.num_cols = None

    def clear_op_info(self):
        self.next_op_idx = None
        self.iter_ops = None
