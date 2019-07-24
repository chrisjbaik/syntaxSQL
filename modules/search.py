import traceback
from itertools import chain, permutations
from query import Query, join_path_needs_update, with_updated_join_paths, \
    to_proto_op, to_proto_tribool, to_proto_agg, to_str_agg, to_proto_dir
from duoquest_pb2 import TRUE, UNKNOWN, AggregatedColumn, Predicate, \
    OrderedColumn, AND, OR

AGG_OPS = ('max', 'min', 'count', 'sum', 'avg')
NEW_WHERE_OPS = ('=','>','<','>=','<=','!=','like','not in','in','between')
DIR_LIMIT_OPS = (("asc",True),("asc",False),("desc",True),("desc",False))

def index_to_column_name(index, table):
    column_name = table["column_names"][index][1]
    table_index = table["column_names"][index][0]
    table_name = table["table_names"][table_index]
    return table_name, column_name, index

class SearchState(object):
    def __init__(self, next, query, history=None):
        # a chain of keys storing next item to infer, e.g.:
        #   left, select: select in left subquery
        #   where_op, 2, select: first subquery at index 2 of where clause
        #   having_op, 5, select: first subquery at index 5 of having clause
        self.next = next

        # stores the cumulative probability of the current state
        self.prob = 1

        # join path ranking if multiple join paths generated
        self.join_path_ranking = 0

        # store parent to return to it after subquery
        self.parent = None

        # Next keyword to use
        self.next_kw = None
        # scores for candidate keywords
        self.kw_scores = None
        # used kws for current query
        self.used_kws = None
        # number of keywords for current query
        self.num_kws = None

        # next column id to use
        self.next_col = None
        # scores for candidate columns
        self.col_scores = None
        # used candidates for current clause
        self.used_cols = None
        # number of column slots for current clause
        self.num_cols = None

        # next agg to use
        self.next_agg = None
        # scores for candidate aggs
        self.agg_scores = None
        # used candidates for current clause
        self.used_aggs = None
        # number of agg slots for current clause
        self.num_aggs = None

        # index of next op to use from iter_ops
        self.next_op_idx = None
        # offset of next_op_idx in complete list of predicates
        self.next_op_offset = None
        # scores for each op
        self.op_scores = None
        # ops to use for current clause
        self.iter_ops = None
        # number of op slots for current col
        self.num_ops = None

        # scores for order by dir and limit presence
        self.dir_limit_scores = None

        # scores for and_or for predicates
        self.and_or_scores = None

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

    def update_join_paths(self, cur_pq):
        states = []

        try:
            needs_update = join_path_needs_update(self.query.schema, cur_pq)
        except Exception as e:
            # print(traceback.format_exc())
            return None, False

        if needs_update:
            try:
                new_pqs = with_updated_join_paths(self.query.schema, cur_pq)
            except Exception as e:
                print(traceback.format_exc())
                return None, False

            for rank, new_pq in enumerate(new_pqs):
                new = self.copy()
                new.set_subquery(new.query.pq, self.next, new_pq)
                new.join_path_ranking = rank
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

        copied.prob = self.prob
        copied.join_path_ranking = self.join_path_ranking

        copied.next_kw = self.next_kw
        copied.kw_scores = self.kw_scores         # will not be modified
        if self.used_kws is not None:
            copied.used_kws = set(self.used_kws)
        copied.num_kws = self.num_kws

        copied.next_col = self.next_col
        copied.col_scores = self.col_scores       # will not be modified
        if self.used_cols is not None:
            copied.used_cols = set(self.used_cols)
        copied.num_cols = self.num_cols

        copied.next_agg = self.next_agg
        copied.agg_scores = self.agg_scores       # will not be modified
        if self.used_aggs is not None:
            copied.used_aggs = set(self.used_aggs)
        copied.num_aggs = self.num_aggs

        copied.next_op_idx = self.next_op_idx
        copied.next_op_offset = self.next_op_offset
        copied.op_scores = self.op_scores       # will not be modified
        copied.iter_ops = self.iter_ops         # will not be modified
        copied.num_ops = self.num_ops

        copied.and_or_scores = self.and_or_scores  # will not be modified
        copied.dir_limit_scores = self.dir_limit_scores  # will not be modified

        return copied

    def next_num_kw_states(self, num_kw_scores):
        states = []
        for num_kws, score in enumerate(num_kw_scores):
            # if not client and b and len(states) >= b:
            #     break
            new = self.copy()
            new.num_kws = num_kws
            new.prob = new.prob * score
            states.append(new)

        return states

    def next_kw_states(self, client):
        states = []
        if len(self.used_kws) < self.num_kws:
            for kw, score in enumerate(self.kw_scores):
                # if not client and b and len(states) >= b:
                #     break
                if kw in self.used_kws:
                    continue
                new = self.copy()
                new.next_kw = kw

                new.prob = new.prob * score

                new_pq = new.find_protoquery(new.query.pq, new.next)
                if client and client.should_prune(new.query):
                    continue

                states.append(new)

        # if no candidate states, next_kw to None
        if not states:
            self.next_kw = None
            return [self]
        else:
            return states

    def next_num_agg_states(self, clause, num_agg_scores, client):
        states = []

        # subqueries can only have one projected column (i.e. 0 or 1 agg)
        if clause == 'select' and self.parent:
            num_agg_scores = num_agg_scores[0:2]

        # ORDER BY can only have one column
        if clause == 'order_by':
            num_agg_scores = num_agg_scores[0:2]

        for num_aggs, score in enumerate(num_agg_scores):
            # cannot have HAVING without aggs
            if clause == 'having' and num_aggs == 0:
                continue

            # if not client and b and len(states) >= b:
            #     break
            new = self.copy()
            new.num_aggs = num_aggs
            new.used_aggs = set()

            new.prob = new.prob * score

            if clause == 'select':
                new_pq = new.find_protoquery(new.query.pq, new.next)
                new_pq.min_select_cols = len(new_pq.select) + \
                    (max(num_aggs, 1) - 1) + (new.num_cols - len(new.used_cols))
                if new.num_aggs == 0:
                    new_pq.select[-1].has_agg = to_proto_tribool(False)

            if client and client.should_prune(new.query):
                continue

            states.append(new)

        return states

    def next_select_agg_states(self, client):
        states = []

        if len(self.used_aggs) == self.num_aggs:
            if len(self.used_aggs) == 0:
                cur_pq = self.find_protoquery(self.query.pq, self.next)
                cur_pq.select[-1].has_agg = to_proto_tribool(False)
            self.next_agg = None
            return [self]
        elif len(self.used_aggs) < self.num_aggs:
            states = []
            for agg, score in enumerate(self.agg_scores):
                # if not client and b and len(states) >= b:
                #     break
                if agg in self.used_aggs:
                    continue

                new = self.copy()
                new_pq = new.find_protoquery(new.query.pq, new.next)

                new.next_agg = agg
                new.prob = new.prob * score

                if len(new.used_aggs) > 0:
                    agg_col = AggregatedColumn()
                    agg_col.col_id = new.next_col
                    agg_col.has_agg = to_proto_tribool(True)
                    agg_col.agg = to_proto_agg(AGG_OPS[agg])
                    new_pq.select.append(agg_col)
                else:
                    new_pq.select[-1].has_agg = to_proto_tribool(True)
                    new_pq.select[-1].agg = to_proto_agg(AGG_OPS[agg])

                new.used_aggs.add(new.next_agg)

                if client and client.should_prune(new.query):
                    continue

                states.append(new)
            return states
        else:
            raise Exception('Exceeded number of aggs.')

    def next_dir_limit_states(self, tsq_level, ordered_col, client):
        states = []

        can_prune_order = (client and tsq_level in ('default', 'no_range'))

        for dir_limit, score in enumerate(self.dir_limit_scores):
            # if not can_prune_order and b and len(states) >= b:
            #     break
            new = self.copy()
            new.prob = new.prob * score

            new_oc = OrderedColumn()
            new_oc.CopyFrom(ordered_col)

            new_pq = new.find_protoquery(new.query.pq, new.next)

            dir, has_limit = DIR_LIMIT_OPS[dir_limit]
            new.history[0].append(dir)

            new_oc.dir = to_proto_dir(dir)
            new_pq.order_by.append(new_oc)

            if new_pq.has_limit == UNKNOWN:
                new_pq.has_limit = to_proto_tribool(has_limit)

            if can_prune_order and client.should_prune(new.query):
                continue

            states.append(new)

        return states

    def next_agg_states(self, client):
        if len(self.used_aggs) == self.num_aggs:
            self.next_agg = None
            return [self]
        elif len(self.used_aggs) < self.num_aggs:
            states = []
            for agg, score in enumerate(self.agg_scores):
                # if not client and b and len(states) >= b:
                #     break
                if agg in self.used_aggs:
                    continue
                new = self.copy()
                new.next_agg = agg
                new.prob = new.prob * score
                states.append(new)
            return states
        else:
            raise Exception('Exceeded number of aggs.')

    def next_num_col_states(self, clause, num_col_scores, client):
        states = []

        # Subqueries can only have one projection
        if clause == 'select' and self.parent:
            num_col_scores = [1]

        # ORDER BY can only have 1 column max
        if clause == 'order_by':
            num_col_scores = [1]

        for num_cols, score in enumerate(num_col_scores):
            # 0 is not a feasible option for num_cols
            num_cols = num_cols + 1

            # if not client and b and len(states) >= b:
            #     break
            new = self.copy()
            new.num_cols = num_cols

            new.prob = new.prob * score

            new_pq = new.find_protoquery(new.query.pq, new.next)

            if clause == 'select':
                new_pq.min_select_cols = num_cols
            elif clause == 'where':
                new_pq.min_where_preds = num_cols
            elif clause == 'group_by':
                new_pq.min_group_by_cols = num_cols
            elif clause == 'having':
                new_pq.min_having_preds = num_cols
            elif clause == 'order_by':
                new_pq.min_order_by_cols = num_cols
            else:
                raise Exception('Unknown clause: {}'.format(clause))

            if client and client.should_prune(new.query):
                continue

            states.append(new)

        return states

    # get history while inferring select (updated all at once at the end)
    def get_select_history(self, tables):
        history = [list(self.history[0])] * 2

        cur_pq = self.find_protoquery(self.query.pq, self.next)

        # order by position in col_cands, then original order in select
        sorted_select = sorted(enumerate(cur_pq.select),
            key=lambda (i, x): (-self.col_scores[x.col_id], i))

        for i, agg_col in sorted_select:
            col_name = index_to_column_name(agg_col.col_id, tables)
            history[0].append(col_name)

            if agg_col.has_agg == TRUE:
                history[0].append(to_str_agg(agg_col.agg))

        return history

    def next_select_col_states(self, client):
        if len(self.used_cols) == self.num_cols:
            self.next_col = None
            return [self]
        elif len(self.used_cols) < self.num_cols:
            states = []

            for col, score in enumerate(self.col_scores):
                # if not client and b and len(states) >= b:
                #     break
                if col in self.used_cols:
                    continue

                new = self.copy()
                new_pq = new.find_protoquery(new.query.pq, new.next)

                new.next_col = col
                new.used_cols.add(col)
                new.prob = new.prob * score

                agg_col = AggregatedColumn()
                agg_col.col_id = new.next_col
                new_pq.select.append(agg_col)

                if client and client.should_prune(new.query):
                    continue

                states.append(new)

            return states
        else:
            raise Exception('Exceeded number of columns.')

    def next_col_states(self, client):
        if len(self.used_cols) == self.num_cols:
            self.next_col = None
            return [self]
        elif len(self.used_cols) < self.num_cols:
            states = []

            for col, score in enumerate(self.col_scores):
                # if not client and b and len(states) >= b:
                #     break
                if col in self.used_cols:
                    continue
                new = self.copy()
                new.next_col = col
                new.prob = new.prob * score
                states.append(new)
            return states
        else:
            raise Exception('Exceeded number of columns.')

    def next_num_op_states(self, clause, num_op_scores, client):
        states = []
        for num_ops, score in enumerate(num_op_scores):
            # Cannot have 0 ops
            num_ops = num_ops + 1

            # if not client and b and len(states) >= b:
            #     break
            new = self.copy()
            new.num_ops = num_ops
            new.prob = new.prob * score

            new_pq = new.find_protoquery(new.query.pq, new.next)

            if clause == 'where':
                new_pq.min_where_preds = len(new_pq.where.predicates) + \
                    num_ops + (new.num_cols - len(new.used_cols))
            elif clause == 'having':
                new_pq.min_having_preds = len(new_pq.having.predicates) + \
                    num_ops + (new.num_cols - len(new.used_cols))
            else:
                raise Exception('Unknown clause: {}'.format(clause))

            if client and client.should_prune(new.query):
                continue

            states.append(new)

        return states

    def next_op_states(self, clause, col_name, client):
        states = []
        # self.next[-1] = next
        self.next_op_idx = 0

        or_op = False
        cur_pq = self.find_protoquery(self.query.pq, self.next)
        if clause == 'where':
            self.next_op_offset = len(cur_pq.where.predicates)
            or_op = (cur_pq.where.logical_op == OR)
        elif clause == 'having':
            self.next_op_offset = len(cur_pq.having.predicates)
            or_op = (cur_pq.having.logical_op == OR)
        else:
            raise Exception('Unknown clause: {}'.format(clause))

        op_score_list = enumerate(self.op_scores)

        # Disallow invalid OP 10
        op_score_list = filter(lambda x: x[0] != 10, op_score_list)

        cand_iter = permutations(op_score_list, self.num_ops)

        # HACK: because SyntaxSQLNet can't do multiple = ops
        if or_op and self.num_ops > 1:
            cand_iter = chain([[(0, self.op_scores[0])] * self.num_ops],
                cand_iter)

        for op_scores in cand_iter:
            # if b and len(states) >= b:
            #     break
            new = self.copy()

            new_pq = new.find_protoquery(new.query.pq, new.next)

            for i, op_score in enumerate(op_scores):
                op, score = op_score

                if i != 0:
                    new.history[0].append(col_name)
                new.history[0].append(NEW_WHERE_OPS[op])

                new.prob = new.prob * score

                pred = Predicate()
                pred.col_id = new.next_col
                pred.op = to_proto_op(NEW_WHERE_OPS[op])

                if clause == 'where':
                    pred.has_agg = to_proto_tribool(False)
                    new_pq.where.predicates.append(pred)
                elif clause == 'having':
                    pred.has_agg = to_proto_tribool(True)
                    pred.agg = to_proto_agg(AGG_OPS[new.next_agg])
                    new_pq.having.predicates.append(pred)

            if client and client.should_prune(new.query):
                continue

            new.iter_ops = op_scores
            states.append(new)

        return states

    def clear_kw_info(self):
        self.next_kw = None
        self.kw_scores = None
        self.used_kws = None
        self.num_kws = None

    def clear_agg_info(self):
        self.next_agg = None
        self.agg_scores = None
        self.used_aggs = None
        self.num_aggs = None

    def clear_col_info(self):
        self.next_col = None
        self.col_scores = None
        self.used_cols = None
        self.num_cols = None

    def clear_op_info(self):
        self.next_op_idx = None
        self.next_op_offset = None
        self.op_scores = None
        self.iter_ops = None
        self.num_ops = None

    def clear_and_or_info(self):
        self.and_or_scores = None
