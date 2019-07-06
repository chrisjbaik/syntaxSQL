from itertools import permutations
from query import Query, join_path_needs_update, with_updated_join_paths
from query_pb2 import TRUE

NEW_WHERE_OPS = ('=','>','<','>=','<=','!=','like','not in','in','between')

class SearchState(object):
    def __init__(self, next, query, history=None):
        # a chain of keys storing next item to infer, e.g.:
        #   left, select: select in left subquery
        #   where_op, 2, select: first subquery at index 2 of where clause
        #   having_op, 5, select: first subquery at index 5 of having clause
        self.next = next

        # store parent to return to it after subquery
        self.parent = None

        # Next keyword to use
        self.next_kw = None
        # candidate keywords, in order
        self.kw_cands = None
        # used kws for current query
        self.used_kws = None
        # number of keywords for current query
        self.num_kws = None

        # next column to use (from col_cands)
        self.next_col = None
        # candidate columns for current clause, in order
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
        # number of op slots for current col
        self.num_ops = None

        self.query = query

        if not history:
            history = [['root']] * 2
        self.history = history

    def set_subquery(self, pq, next, set_pq):
        if next[0] == 'left':
            self.set_subquery(pq.left, next[1:], set_pq)
        elif next[0] == 'right':
            self.set_subquery(pq.right, next[1:], set_pq)
        elif next[0] == 'where_op' and \
            len(next) > 1 and isinstance(next[1], int):
            pred = pq.where.predicates[next[1]]
            if pred.has_subquery != TRUE:
                raise Exception('No subquery at {}'.format(next[1]))
            self.set_subquery(pred.subquery, next[2:], set_pq)
        elif next[0] == 'having_op' and \
            len(next) > 1 and isinstance(next[1], int):
            pred = pq.having.predicates[next[1]]
            if pred.has_subquery != TRUE:
                raise Exception('No subquery at {}'.format(next[1]))
            self.set_subquery(pred.subquery, next[2:], set_pq)
        else:
            pq.CopyFrom(set_pq)

    # next is array, e.g. [left, select, ...]
    # returns smallest subquery ProtoQuery object
    def find_protoquery(self, pq, next):
        if next[0] == 'left':
            return self.find_protoquery(pq.left, next[1:])
        elif next[0] == 'right':
            return self.find_protoquery(pq.right, next[1:])
        elif next[0] == 'where_op' and \
            len(next) > 1 and isinstance(next[1], int):
            pred = pq.where.predicates[next[1]]
            if pred.has_subquery != TRUE:
                raise Exception('No subquery at {}'.format(next[1]))
            return self.find_protoquery(pred.subquery, next[2:])
        elif next[0] == 'having_op' and \
            len(next) > 1 and isinstance(next[1], int):
            pred = pq.having.predicates[next[1]]
            if pred.has_subquery != TRUE:
                raise Exception('No subquery at {}'.format(next[1]))
            return self.find_protoquery(pred.subquery, next[2:])
        else:
            return pq

    def update_join_paths(self):
        states = []

        pq = self.find_protoquery(self.query.pq, self.next)

        needs_update = join_path_needs_update(self.query.schema, pq)
        if needs_update is None:
            return None, False
        elif needs_update:
            try:
                new_pqs = with_updated_join_paths(self.query.schema, pq)
            except Exception as e:
                print(traceback.format_exc())
                return None, False

            for new_pq in new_pqs:
                new = self.copy()
                new.set_subquery(new.query.pq, self.next, new_pq)
                states.append(new)
            return states, True
        else:
            return self, False

    def set_parent(self, parent):
        # link parent history, query with current values
        self.parent = parent.copy()
        self.parent.history = self.history
        self.parent.query = self.query

    def copy(self, query=None):
        history_copy = [list(self.history[0])] * 2

        if query is None:
            query = self.query.copy()
        copied = SearchState(list(self.next), query, history=history_copy)

        if self.parent:
            copied.set_parent(self.parent)

        copied.next_kw = self.next_kw
        copied.kw_cands = self.kw_cands         # will not be modified
        if self.used_kws is not None:
            copied.used_kws = set(self.used_kws)
        copied.num_kws = self.num_kws

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
        copied.num_ops = self.num_ops

        return copied

    def next_num_kw_states(self, num_kw_cands, b):
        states = []
        for num_kws in num_kw_cands:
            if b and len(states) >= b:
                break
            new = self.copy()
            new.num_kws = num_kws
            states.append(new)

        return states

    def next_kw_states(self, b):
        states = []
        if len(self.used_kws) < self.num_kws:
            for kw in self.kw_cands:
                if b and len(states) >= b:
                    break
                if kw in self.used_kws:
                    continue
                new = self.copy()
                new.next_kw = kw
                states.append(new)

        # if no candidate states, next_kw to None
        if not states:
            self.next_kw = None
            return [self]
        else:
            return states

    def next_num_agg_states(self, num_agg_cands, b):
        states = []
        for num_aggs in num_agg_cands:
            if b and len(states) >= b:
                break
            new = self.copy()
            new.num_aggs = num_aggs
            states.append(new)

        return states

    def next_agg_states(self, b):
        states = []
        if len(self.used_aggs) < self.num_aggs:
            for agg in self.agg_cands:
                if b and len(states) >= b:
                    break
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

    def next_num_col_states(self, num_col_cands, b):
        states = []
        for num_cols in num_col_cands:
            if b and len(states) >= b:
                break
            new = self.copy()
            new.num_cols = num_cols
            states.append(new)

        return states

    def next_col_states(self, b):
        states = []
        if len(self.used_cols) < self.num_cols:
            for col in self.col_cands:
                if b and len(states) >= b:
                    break
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

    def next_num_op_states(self, num_op_cands, b):
        states = []
        for num_ops in num_op_cands:
            if b and len(states) >= b:
                break
            new = self.copy()
            new.num_ops = num_ops
            states.append(new)

        return states

    def next_op_states(self, next, num_ops, op_cands, col_name, b):
        states = []
        self.next[-1] = next
        self.next_op_idx = 0
        for ops in permutations(op_cands, num_ops):
            if b and len(states) >= b:
                break
            new = self.copy()

            for i, op in enumerate(ops):
                if i != 0:
                    new.history[0].append(col_name)
                new.history[0].append(NEW_WHERE_OPS[op])

            new.iter_ops = ops
            states.append(new)

        return states

    def clear_kw_info(self):
        self.next_kw = None
        self.kw_cands = None
        self.used_kws = None
        self.num_kws = None

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
        self.num_ops = None
