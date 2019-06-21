from itertools import permutations

from query_pb2 import *

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

    def to_proto(self):
        pq = ProtoQuery()

        if self.set_op is None or self.set_op == 'none':
            pq.set_op = NO_SET_OP
        elif self.set_op == 'intersect':
            pq.set_op = INTERSECT
        elif self.set_op == 'except':
            pq.set_op = EXCEPT
        elif self.set_op == 'union':
            pq.set_op = UNION
        else:
            raise Exception('Unrecognized set_op: {}'.format(self.set_op))

        if self.left:
            pq.left = self.left.to_proto()
        if self.right:
            pq.right = self.right.to_proto()

        self.to_proto_select(pq)
        self.to_proto_where(pq)
        self.to_proto_group_by(pq)
        self.to_proto_having(pq)
        self.to_proto_order_by(pq)

        return pq

    def to_proto_order_by(self, pq):
        if self.order_by is None:
            pq.has_order_by = UNKNOWN
        elif self.order_by == False:
            pq.has_order_by = FALSE
        else:
            pq.has_order_by = TRUE

        if self.limit is None:
            pq.has_limit = UNKNOWN
        elif self.limit == False:
            pq.has_limit = FALSE
        else:
            pq.has_limit = TRUE

        if isinstance(self.order_by, list):
            cur_col_id = None
            cur_agg = None
            cur_dir = None
            for item in self.order_by:
                if cur_col_id is None:
                    cur_col_id = item[2]
                    continue

                if cur_agg is None:
                    cur_agg = item
                    continue

                if cur_dir is None:
                    cur_dir = item
                    continue

                orderedcol = OrderedColumn()
                orderedcol.agg_col.col_id = cur_col_id
                if cur_agg == 'none_agg':
                    orderedcol.agg_col.has_agg = FALSE
                else:
                    orderedcol.agg_col.has_agg = TRUE
                    orderedcol.agg_col.agg = self.to_proto_agg(cur_agg)

                if cur_dir == 'asc':
                    orderedcol.dir = ASC
                elif cur_dir == 'desc':
                    orderedcol.dir = DESC
                else:
                    raise Exception('Unrecognized dir: {}'.format(dir))

                cur_col_id = None
                cur_agg = None
                cur_dir = None
                pq.order_by.append(orderedcol)

    def to_proto_having(self, pq):
        if self.having is None:
            pq.has_having = UNKNOWN
        elif self.having == False:
            pq.has_having = FALSE
        else:
            pq.has_having = TRUE

        if isinstance(self.having, list):
            having = SelectionClause()

            cur_col_id = None
            cur_agg = None
            cur_op = None
            for item in self.having:
                if cur_col_id is None:
                    cur_col_id = item[2]
                    continue

                if cur_agg is None:
                    cur_agg = item
                    continue

                if cur_op is None:
                    cur_op = item
                    continue

                pred = Predicate()
                pred.col_id = cur_col_id

                if cur_agg == 'none_agg':
                    pred.has_agg = FALSE
                else:
                    pred.has_agg = TRUE
                    pred.agg = self.to_proto_agg(cur_agg)

                pred.op = self.to_proto_op(cur_op)

                if isinstance(item, Query):
                    pred.has_subquery = TRUE
                    pred.subquery = item.to_proto()
                else:
                    pred.has_subquery = FALSE
                    pred.value = str(item)

                cur_col_id = None
                cur_agg = None
                cur_op = None

                having.predicates.append(pred)

            pq.having = having

    def to_proto_group_by(self, pq):
        if self.group_by is None:
            pq.has_group_by = UNKNOWN
        elif self.group_by == False:
            pq.has_group_by = FALSE
        else:
            pq.has_group_by = TRUE

        if isinstance(self.group_by, list):
            for col in self.group_by:
                pq.group_by.append(col[2])

    def to_proto_where(self, pq):
        if self.where is None:
            pq.has_where = UNKNOWN
        elif self.where == False:
            pq.has_where = FALSE
        else:
            pq.has_where = TRUE

        if isinstance(self.where, list):
            where = SelectionClause()

            cur_col_id = None
            cur_op = None
            for item in self.where:
                if item == 'and':
                    where.logical_op = AND
                elif item == 'or':
                    where.logical_op = OR
                else:
                    if cur_col_id is None:
                        cur_col_id = item[2]
                        continue

                    if cur_op is None:
                        cur_op = item
                        continue

                    pred = Predicate()
                    pred.col_id = cur_col_id
                    pred.op = self.to_proto_op(cur_op)
                    pred.has_agg = FALSE

                    if isinstance(item, Query):
                        pred.has_subquery = TRUE
                        pred.subquery = item.to_proto()
                    else:
                        pred.has_subquery = FALSE
                        pred.value = str(item)

                    cur_col_id = None
                    cur_op = None

                    where.predicates.append(pred)

            pq.where = where

    def to_proto_op(self, op):
        if op == '=':
            return EQUALS
        elif op == '>':
            return GT
        elif op == '<':
            return LT
        elif op == '>=':
            return GEQ
        elif op == '<=':
            return LEQ
        elif op == '!=':
            return NEQ
        elif op == 'like':
            return LIKE
        elif op == 'in':
            return IN
        elif op == 'not in':
            return NOT_IN
        elif op == 'between':
            return BETWEEN
        else:
            raise Exception('Unrecognized op: {}'.format(op))

    def to_proto_agg(self, agg):
        if agg == 'max':
            return MAX
        elif agg == 'min':
            return MIN
        elif agg == 'count':
            return COUNT
        elif agg == 'sum':
            return SUM
        elif agg == 'avg':
            return AVG
        else:
            raise Exception('Unrecognized agg: {}'.format(agg))

    def to_proto_select(self, pq):
        if isinstance(self.select, list):
            # [(tbl_name, col_name, col_id), agg] repeated, agg may be missing
            cur_col_id = None
            for item in self.select:
                if cur_col_id is None:
                    cur_col_id = item[2]
                    continue

                agg = item

                aggcol = AggregatedColumn()
                aggcol.col_id = cur_col_id
                if agg == 'none_agg':
                    aggcol.has_agg = FALSE
                else:
                    aggcol.has_agg = TRUE
                    aggcol.agg = self.to_proto_agg(agg)

                pq.select.append(aggcol)
                cur_col_id = None

            # hanging column with no aggregate means we don't know
            if cur_col_id:
                aggcol = AggregatedColumn()
                aggcol.col_id = cur_col_id
                aggcol.has_agg = UNKNOWN
                pq.select.append(aggcol)

    def copy(self):
        copied = Query()
        copied.set_op = self.set_op
        if self.left:
            copied.left = self.left.copy()
        if self.right:
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
        if self.set_op is not None and self.set_op != 'none':
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
        elif next[0] == 'where_op' and \
            len(next) > 1 and isinstance(next[1], int):
            return self.where[next[1]].find_subquery(next[2:])
        elif next[0] == 'having_op' and \
            len(next) > 1 and isinstance(next[1], int):
            return self.having[next[1]].find_subquery(next[2:])
        else:
            return self

class SearchState(object):
    def __init__(self, next, history=None, query=None):
        # a chain of keys storing next item to infer, e.g.:
        #   left, select: select in left subquery
        #   where_op, 2, select: first subquery at index 2 of where clause
        #   having_op, 5, select: first subquery at index 5 of having clause
        self.next = next

        # store parent to return to it after subquery
        self.parent = None

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

    def set_parent(self, parent):
        # link parent history, query with current values
        self.parent = parent.copy()
        self.parent.history = self.history
        self.parent.query = self.query

    def copy(self):
        history_copy = [list(self.history[0])] * 2

        copied = SearchState(list(self.next), history=history_copy,
            query=self.query.copy())

        if self.parent:
            copied.set_parent(self.parent)

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

        # if no candidate states, next_agg to None
        if not states:
            self.next_agg = None
            return [self]
        else:
            return states

    def next_col_states(self):
        states = []
        for col in self.col_cands:
            if col in self.used_cols:
                continue
            new = self.copy()
            new.next_col = col
            states.append(new)

        # if no candidate states, next_col to None
        if not states:
            self.next_col = None
            return [self]
        else:
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
