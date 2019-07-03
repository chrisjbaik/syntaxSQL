import traceback

from query_pb2 import *
from schema import JoinEdge

def to_proto_tribool(boolval):
    if boolval is None:
        return UNKNOWN
    elif boolval:
        return TRUE
    else:
        return FALSE

def to_proto_set_op(set_op):
    if set_op == 'none':
        return NO_SET_OP
    elif set_op == 'intersect':
        return INTERSECT
    elif set_op == 'except':
        return EXCEPT
    elif set_op == 'union':
        return UNION
    else:
        raise Exception('Unknown set_op: {}'.format(set_op))

def to_proto_agg(agg):
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

def to_str_agg(proto_agg):
    if proto_agg == MAX:
        return 'max'
    elif proto_agg == MIN:
        return 'min'
    elif proto_agg == COUNT:
        return 'count'
    elif proto_agg == SUM:
        return 'sum'
    elif proto_agg == AVG:
        return 'avg'
    else:
        raise Exception('Unrecognized agg: {}'.format(proto_agg))

def to_proto_logical_op(logical_op):
    if logical_op == 'and':
        return AND
    elif logical_op == 'or':
        return OR
    else:
        raise Exception('Unknown logical_op: {}'.format(logical_op))

def to_str_logical_op(proto_logical_op):
    if proto_logical_op == AND:
        return 'and'
    elif proto_logical_op == OR:
        return 'or'
    else:
        raise Exception('Unknown logical_op: {}'.format(proto_logical_op))

def to_proto_op(op):
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

def to_str_op(proto_op):
    if proto_op == EQUALS:
        return '='
    elif proto_op == GT:
        return '>'
    elif proto_op == LT:
        return '<'
    elif proto_op == GEQ:
        return '>='
    elif proto_op == LEQ:
        return '<='
    elif proto_op == NEQ:
        return '!='
    elif proto_op == LIKE:
        return 'like'
    elif proto_op == IN:
        return 'in'
    elif proto_op == NOT_IN:
        return 'not in'
    elif proto_op == BETWEEN:
        return 'between'
    else:
        raise Exception('Unrecognized op: {}'.format(proto_op))

def to_proto_dir(dir):
    if dir == 'desc':
        return DESC
    elif dir == 'asc':
        return ASC
    else:
        raise Exception('Unrecognized dir: {}'.format(dir))

def to_str_dir(proto_dir):
    if proto_dir == DESC:
        return 'desc'
    elif proto_dir == ASC:
        return 'asc'
    else:
        raise Exception('Unrecognized dir: {}'.format(proto_dir))

def gen_alias(alias_idx, alias_prefix):
    if alias_prefix:
        return '{}t{}'.format(alias_prefix, alias_idx)
    else:
        return 't{}'.format(alias_idx)

def from_clause_str(pq, schema, alias_prefix):
    aliases = {}
    join_exprs = ['FROM']

    tables = map(lambda x: schema.get_table(x), pq.from_clause.edge_map.keys())
    tbl = min(tables, key=lambda x: x.syn_name)
    alias = gen_alias(len(aliases) + 1, alias_prefix)
    aliases[tbl.syn_name] = alias
    join_exprs.append(u'{} AS {}'.format(tbl.syn_name, alias))

    stack = [tbl]

    while stack:
        tbl = stack.pop()
        for edge in pq.from_clause.edge_map[tbl.id].edges:
            edge = JoinEdge(
                schema.get_col(edge.fk_col_id),
                schema.get_col(edge.pk_col_id)
            )
            other_tbl = edge.other(tbl)
            if other_tbl.syn_name in aliases:
                continue

            alias = gen_alias(len(aliases) + 1, alias_prefix)
            aliases[other_tbl.syn_name] = alias
            join_exprs.append(
                u'JOIN {} AS {} ON {}.{} = {}.{}'.format(
                    other_tbl.syn_name, alias,
                    aliases[tbl.syn_name], edge.key(tbl).syn_name,
                    aliases[other_tbl.syn_name], edge.key(other_tbl).syn_name
                )
            )
            stack.append(other_tbl)

    return u' '.join(join_exprs), aliases

def select_clause_str(pq, schema, aliases):
    projs = []
    for agg_col in pq.select:
        if agg_col.has_agg == TRUE:
            if agg_col.agg == COUNT and \
                schema.get_col(agg_col.col_id).syn_name != '*':
                proj_str = u'{}(DISTINCT {})'.format(
                    to_str_agg(agg_col.agg),
                    schema.get_aliased_col(aliases, agg_col.col_id)
                )
            else:
                proj_str = u'{}({})'.format(
                    to_str_agg(agg_col.agg),
                    schema.get_aliased_col(aliases, agg_col.col_id)
                )
            projs.append(proj_str)
        else:
            projs.append(schema.get_aliased_col(aliases, agg_col.col_id))

    if pq.distinct:
        return u'SELECT DISTINCT ' + ', '.join(projs)
    else:
        return u'SELECT ' + ', '.join(projs)

def where_clause_str(pq, schema, aliases):
    where_exprs = ['WHERE']
    for i, pred in enumerate(pq.where.predicates):
        if i != 0:
            where_exprs.append(to_str_logical_op(pq.where.logical_op))

        where_val = None
        if pred.has_subquery:
            where_val = u'({})'.format(
                generate_sql_str(pred.subquery, schema,
                    alias_prefix='w{}'.format(i))
            )
        else:
            if not pred.value:
                raise Exception('Value is empty when generating where clause.')

            if pred.op in (IN, NOT_IN):
                where_val = u"({})".format(
                        u','.join(map(lambda x: u"'{}'".format(x), pred.value)))
            elif pred.op == BETWEEN:
                where_val = u"{} AND {}".format(pred.value[0], pred.value[1])
            else:
                where_val = u"'{}'".format(pred.value[0])

        pred_str = u' '.join([
            schema.get_aliased_col(aliases, pred.col_id),
            to_str_op(pred.op),
            where_val
        ])
        where_exprs.append(pred_str)

    return u' '.join(where_exprs)

def group_by_clause_str(pq, schema, aliases):
    group_by_exprs = ['GROUP BY']
    for col_id in pq.group_by:
        group_by_exprs.append(schema.get_aliased_col(aliases, col_id))
    return u' '.join(group_by_exprs)

def having_clause_str(pq, schema, aliases):
    having_exprs = ['HAVING']
    for i, pred in enumerate(pq.having.predicates):
        if i != 0:
            having_exprs.append(to_str_logical_op(pq.having.logical_op))

        having_col = u'{}({})'.format(
            to_str_agg(pred.agg),
            schema.get_aliased_col(aliases, pred.col_id)
        )

        having_val = None
        if pred.has_subquery:
            having_val = '({})'.format(
                generate_sql_str(pred.subquery, schema,
                    alias_prefix='h{}'.format(i))
            )
        elif pred.op in (IN, NOT_IN):
            having_val = u"({})".format(
                    u','.join(map(lambda x: u"'{}'".format(x), pred.value)))
        elif pred.op == BETWEEN:
            having_val = u"{} AND {}".format(pred.value[0], pred.value[1])
        else:
            having_val = u"'{}'".format(pred.value[0])

        pred_str = u' '.join([having_col, to_str_op(pred.op), having_val])
        having_exprs.append(pred_str)

    return u' '.join(having_exprs)

def order_by_clause_str(pq, schema, aliases):
    order_by_exprs = ['ORDER BY']
    for ordered_col in pq.order_by:
        if ordered_col.agg_col.has_agg:
            order_by_exprs.append('{}({}) {}'.format(
                to_str_agg(ordered_col.agg_col.agg),
                schema.get_aliased_col(aliases, ordered_col.agg_col.col_id),
                to_str_dir(ordered_col.dir)
            ))
        else:
            order_by_exprs.append('{} {}'.format(
                schema.get_aliased_col(aliases, ordered_col.agg_col.col_id),
                to_str_dir(ordered_col.dir)
            ))
    return u' '.join(order_by_exprs)

def limit_clause_str(pq):
    if pq.limit == 0:       # if not set, default to 1
        pq.limit = 1
    return u'LIMIT {}'.format(pq.limit)

def generate_sql_str(pq, schema, alias_prefix=None):
    if pq.set_op != NO_SET_OP:
        set_op_str = None
        if pq.set_op == INTERSECT:
            set_op_str = 'INTERSECT'
        elif pq.set_op == UNION:
            set_op_str = 'UNION'
        elif pq.set_op == EXCEPT:
            set_op_str = 'EXCEPT'

        return u'{} {} {}'.format(
            generate_sql_str(pq.left),
            set_op_str,
            generate_sql_str(pq.right, alias_prefix=set_op_str[0])
        )

    from_clause, aliases = from_clause_str(pq, schema, alias_prefix)
    if from_clause is None:
        raise Exception('FROM clause not generated.')

    clauses = []
    clauses.append(select_clause_str(pq, schema, aliases))
    clauses.append(from_clause)
    if pq.has_where == TRUE:
        clauses.append(where_clause_str(pq, schema, aliases))
    if pq.has_group_by == TRUE:
        clauses.append(group_by_clause_str(pq, schema, aliases))
    if pq.has_having == TRUE:
        clauses.append(having_clause_str(pq, schema, aliases))
    if pq.has_order_by == TRUE:
        clauses.append(order_by_clause_str(pq, schema, aliases))
    if pq.has_limit == TRUE:
        clauses.append(limit_clause_str(pq))

    return u' '.join(clauses)

# Get all tables used in PQ. Does not consider subqueries.
def get_tables(schema, pq):
    # assuming no duplicate tables, change to list() if allowing self-join
    tables = set()
    for agg_col in pq.select:
        tbl = schema.get_col(agg_col.col_id).table
        if tbl:         # check in case tbl is None for '*' column case
            tables.add(tbl)
    if pq.has_where == TRUE:
        for pred in pq.where.predicates:
            tbl = schema.get_col(pred.col_id).table
            if tbl:
                tables.add(tbl)
    if pq.has_group_by == TRUE:
        for col_id in pq.group_by:
            tbl = schema.get_col(col_id).table
            if tbl:
                tables.add(tbl)
    if pq.has_having == TRUE:
        for pred in pq.having.predicates:
            tbl = schema.get_col(pred.col_id).table
            if tbl:
                tables.add(tbl)
    if pq.has_order_by == TRUE:
        for ordered_col in pq.order_by:
            tbl = schema.get_col(ordered_col.agg_col.col_id).table
            if tbl:
                tables.add(tbl)
    return tables

# Only considers whether join path for current localized pq needs updating.
# Does not consider for subqueries or set op children
# Returns:
# - True: if join path needs to be and can be updated
# - False: if join path needs no updating
# - None: if cannot be updated by expanding current join path
def join_path_needs_update(schema, pq):
    tables_in_cur_jp = set(map(lambda x: schema.get_table(x),
        pq.from_clause.edge_map.keys()))

    # if SELECT has a column (i.e. inference started) and there are no tables
    if pq.select and len(tables_in_cur_jp) == 0:
        return True

    # if the current join path doesn't account for all tables in protoquery
    tables = get_tables(schema, pq)
    if tables_in_cur_jp < tables:
        return True
    elif tables_in_cur_jp >= tables:
        return False
    else:
        return None

def with_updated_join_paths(schema, pq):
    # Prioritize subqueries.
    if pq.has_where == TRUE:
        for i, pred in enumerate(pq.where.predicates):
            if pred.has_subquery:
                should_update_s = join_path_needs_update(schema, pred.subquery)
                if should_update_s is None:
                    return None
                elif should_update_s:
                    subqs = with_updated_join_paths(schema, pred.subquery)
                    if subqs is None:
                        return None
                    elif len(subqs) == 1:
                        pred.subquery.CopyFrom(subqs[0])
                        return [pq]
                    else:
                        new_pqs = []
                        for subq in subqs:
                            new_pq = ProtoQuery()
                            new_pq.CopyFrom(pq)
                            new_pq.where.predicates[i].subquery.CopyFrom(subq)
                            new_pqs.append(new_pq)
                        return new_pqs
    if pq.has_having == TRUE:
        for i, pred in enumerate(pq.having.predicates):
            if pred.has_subquery:
                should_update_s = join_path_needs_update(schema, pred.subquery)
                if should_update_s is None:
                    return None
                elif should_update_s:
                    subqs = with_updated_join_paths(schema, pred.subquery)
                    if not subqs:
                        return None
                    elif len(subqs) == 1:
                        pred.subquery.CopyFrom(subqs[0])
                        return [pq]
                    else:
                        new_pqs = []
                        for subq in subqs:
                            new_pq = ProtoQuery()
                            new_pq.CopyFrom(pq)
                            new_pq.having.predicates[i].subquery.CopyFrom(subq)
                            new_pqs.append(new_pq)
                        return new_pqs

    # Then set op children.
    if pq.set_op != NO_SET_OP:
        should_update_left = join_path_needs_update(schema, pq.left)
        if should_update_left is None:
            return None
        elif should_update_left:
            subqs = with_updated_join_paths(schema, pq.left)
            if not subqs:
                return None
            elif len(subqs) == 1:
                pq.left = subqs[0]
                return [pq]
            else:
                new_pqs = []
                for subq in subqs:
                    new_pq = ProtoQuery()
                    new_pq.CopyFrom(pq)
                    new_pq.left = subq
                    new_pqs.append(new_pq)
                return new_pqs
        should_update_right = join_path_needs_update(schema, pq.right)
        if should_update_right is None:
            return None
        elif should_update_right:
            subqs = with_updated_join_paths(schema, pq.right)
            if not subqs:
                return None
            elif len(subqs) == 1:
                pq.right = subqs[0]
                return [pq]
            else:
                new_pqs = []
                for subq in subqs:
                    new_pq = ProtoQuery()
                    new_pq.CopyFrom(pq)
                    new_pq.right = subq
                    new_pqs.append(new_pq)
                return new_pqs

    # Then the main query.
    should_update = join_path_needs_update(schema, pq)
    if should_update is None:
        return None
    elif should_update:
        try:
            jps = schema.get_join_paths(get_tables(schema, pq))
        except Exception as e:
            traceback.print_exc()
            return None

        new_pqs = []
        for jp in jps:
            new_pq = ProtoQuery()
            new_pq.CopyFrom(pq)
            set_proto_from(new_pq, jp)
            new_pqs.append(new_pq)
        return new_pqs

    # If no change, return current query
    return [pq]

def set_proto_from(pq, jp):
    # reset from clause
    del pq.from_clause.edge_list.edges[:]
    for key in pq.from_clause.edge_map.keys():
        del pq.from_clause.edge_map[key]

    if jp.distinct:
        pq.distinct = True

    for edge in jp.edges:
        proto_edge = ProtoJoinEdge()
        proto_edge.fk_col_id = edge.fk_col.id
        proto_edge.pk_col_id = edge.pk_col.id
        pq.from_clause.edge_list.edges.append(proto_edge)

    for tbl, edges in jp.edge_map.items():
        # initialize table in protobuf even if edges don't exist
        pq.from_clause.edge_map.get_or_create(tbl.id)
        for edge in edges:
            proto_edge = ProtoJoinEdge()
            proto_edge.fk_col_id = edge.fk_col.id
            proto_edge.pk_col_id = edge.pk_col.id
            pq.from_clause.edge_map[tbl.id].edges.append(proto_edge)

class Query(object):
    def __init__(self, schema, protoquery=None):
        self.schema = schema

        if protoquery is None:
            protoquery = ProtoQuery()
            protoquery.has_where = UNKNOWN
            protoquery.has_group_by = UNKNOWN
            protoquery.has_having = UNKNOWN
            protoquery.has_order_by = UNKNOWN
            protoquery.has_limit = UNKNOWN
        self.pq = protoquery

        # set operation
        # self.set_op = set_op    # 'none', 'intersect', 'except', 'union'
        # self.left = None        # left subquery
        # self.right = None       # right subquery

        # query components
        # states:
        #  - None: if invalid/unknown, or if set_op != 'none'
        #  - False: if isn't in query
        #  - True: in query, but not yet inferred
        #  - Anything else: inferred value
        # self.select = True
        # self.where = None
        # self.group_by = None
        # self.having = None
        # self.order_by = None
        # self.limit = None

    def with_updated_join_paths(self):
        return with_updated_join_paths(self.schema, self.pq)

    def copy(self):
        new_query = Query(self.schema)
        new_query.pq.CopyFrom(self.pq)
        return new_query

    # def to_proto(self):
    #     pq = ProtoQuery()
    #
    #     if self.set_op is None or self.set_op == 'none':
    #         pq.set_op = NO_SET_OP
    #     elif self.set_op == 'intersect':
    #         pq.set_op = INTERSECT
    #     elif self.set_op == 'except':
    #         pq.set_op = EXCEPT
    #     elif self.set_op == 'union':
    #         pq.set_op = UNION
    #     else:
    #         raise Exception('Unrecognized set_op: {}'.format(self.set_op))
    #
    #     if self.left:
    #         pq.left.CopyFrom(self.left.to_proto())
    #     if self.right:
    #         pq.right.CopyFrom(self.right.to_proto())
    #
    #     self.to_proto_select(pq)
    #     self.to_proto_where(pq)
    #     self.to_proto_group_by(pq)
    #     self.to_proto_having(pq)
    #     self.to_proto_order_by(pq)
    #
    #     return pq
    #
    # def to_proto_order_by(self, pq):
    #     if self.order_by is None:
    #         pq.has_order_by = UNKNOWN
    #     elif self.order_by == False:
    #         pq.has_order_by = FALSE
    #     else:
    #         pq.has_order_by = TRUE
    #
    #     if self.limit is None:
    #         pq.has_limit = UNKNOWN
    #     elif self.limit == False:
    #         pq.has_limit = FALSE
    #     else:
    #         pq.has_limit = TRUE
    #
    #     if isinstance(self.order_by, list):
    #         cur_col_id = None
    #         cur_agg = None
    #         cur_dir = None
    #         for item in self.order_by:
    #             if cur_col_id is None:
    #                 cur_col_id = item[2]
    #                 continue
    #
    #             if cur_agg is None:
    #                 cur_agg = item
    #                 continue
    #
    #             if cur_dir is None:
    #                 cur_dir = item
    #                 continue
    #
    #             orderedcol = OrderedColumn()
    #             orderedcol.agg_col.col_id = cur_col_id
    #             if cur_agg == 'none_agg':
    #                 orderedcol.agg_col.has_agg = FALSE
    #             else:
    #                 orderedcol.agg_col.has_agg = TRUE
    #                 orderedcol.agg_col.agg = self.to_proto_agg(cur_agg)
    #
    #             if cur_dir == 'asc':
    #                 orderedcol.dir = ASC
    #             elif cur_dir == 'desc':
    #                 orderedcol.dir = DESC
    #             else:
    #                 raise Exception('Unrecognized dir: {}'.format(dir))
    #
    #             cur_col_id = None
    #             cur_agg = None
    #             cur_dir = None
    #             pq.order_by.append(orderedcol)
    #
    # def to_proto_having(self, pq):
    #     if self.having is None:
    #         pq.has_having = UNKNOWN
    #     elif self.having == False:
    #         pq.has_having = FALSE
    #     else:
    #         pq.has_having = TRUE
    #
    #     if isinstance(self.having, list):
    #         cur_col_id = None
    #         cur_agg = None
    #         cur_op = None
    #         for item in self.having:
    #             if cur_col_id is None:
    #                 cur_col_id = item[2]
    #                 continue
    #
    #             if cur_agg is None:
    #                 cur_agg = item
    #                 continue
    #
    #             if cur_op is None:
    #                 cur_op = item
    #                 continue
    #
    #             pred = Predicate()
    #             pred.col_id = cur_col_id
    #
    #             if cur_agg == 'none_agg':
    #                 pred.has_agg = FALSE
    #             else:
    #                 pred.has_agg = TRUE
    #                 pred.agg = self.to_proto_agg(cur_agg)
    #
    #             pred.op = self.to_proto_op(cur_op)
    #
    #             if isinstance(item, Query):
    #                 pred.has_subquery = TRUE
    #                 pred.subquery.CopyFrom(item.to_proto())
    #             else:
    #                 pred.has_subquery = FALSE
    #                 if isinstance(item, list):
    #                     for val in item:
    #                         pred.value.append(unicode(val))
    #                 else:
    #                     pred.value.append(unicode(item))
    #
    #             cur_col_id = None
    #             cur_agg = None
    #             cur_op = None
    #
    #             pq.having.predicates.append(pred)
    #
    # def to_proto_group_by(self, pq):
    #     if self.group_by is None:
    #         pq.has_group_by = UNKNOWN
    #     elif self.group_by == False:
    #         pq.has_group_by = FALSE
    #     else:
    #         pq.has_group_by = TRUE
    #
    #     if isinstance(self.group_by, list):
    #         for col in self.group_by:
    #             pq.group_by.append(col[2])
    #
    # def to_proto_where(self, pq):
    #     if self.where is None:
    #         pq.has_where = UNKNOWN
    #     elif self.where == False:
    #         pq.has_where = FALSE
    #     else:
    #         pq.has_where = TRUE
    #
    #     if isinstance(self.where, list):
    #         cur_col_id = None
    #         cur_op = None
    #         for item in self.where:
    #             if item == 'and':
    #                 pq.where.logical_op = AND
    #             elif item == 'or':
    #                 pq.where.logical_op = OR
    #             else:
    #                 if cur_col_id is None:
    #                     cur_col_id = item[2]
    #                     continue
    #
    #                 if cur_op is None:
    #                     cur_op = item
    #                     continue
    #
    #                 pred = Predicate()
    #                 pred.col_id = cur_col_id
    #                 pred.op = self.to_proto_op(cur_op)
    #                 pred.has_agg = FALSE
    #
    #                 if isinstance(item, Query):
    #                     pred.has_subquery = TRUE
    #                     pred.subquery.CopyFrom(item.to_proto())
    #                 else:
    #                     pred.has_subquery = FALSE
    #                     if isinstance(item, list):
    #                         for val in item:
    #                             pred.value.append(unicode(val))
    #                     else:
    #                         pred.value.append(unicode(item))
    #
    #                 cur_col_id = None
    #                 cur_op = None
    #
    #                 pq.where.predicates.append(pred)
    #
    # def to_proto_op(self, op):
    #     if op == '=':
    #         return EQUALS
    #     elif op == '>':
    #         return GT
    #     elif op == '<':
    #         return LT
    #     elif op == '>=':
    #         return GEQ
    #     elif op == '<=':
    #         return LEQ
    #     elif op == '!=':
    #         return NEQ
    #     elif op == 'like':
    #         return LIKE
    #     elif op == 'in':
    #         return IN
    #     elif op == 'not in':
    #         return NOT_IN
    #     elif op == 'between':
    #         return BETWEEN
    #     else:
    #         raise Exception('Unrecognized op: {}'.format(op))
    #
    # def to_proto_agg(self, agg):
    #     if agg == 'max':
    #         return MAX
    #     elif agg == 'min':
    #         return MIN
    #     elif agg == 'count':
    #         return COUNT
    #     elif agg == 'sum':
    #         return SUM
    #     elif agg == 'avg':
    #         return AVG
    #     else:
    #         raise Exception('Unrecognized agg: {}'.format(agg))
    #
    # def to_proto_select(self, pq):
    #     if isinstance(self.select, list):
    #         # [(tbl_name, col_name, col_id), agg] repeated, agg may be missing
    #         cur_col_id = None
    #         for item in self.select:
    #             if cur_col_id is None:
    #                 cur_col_id = item[2]
    #                 continue
    #
    #             agg = item
    #
    #             aggcol = AggregatedColumn()
    #             aggcol.col_id = cur_col_id
    #             if agg == 'none_agg':
    #                 aggcol.has_agg = FALSE
    #             else:
    #                 aggcol.has_agg = TRUE
    #                 aggcol.agg = self.to_proto_agg(agg)
    #
    #             pq.select.append(aggcol)
    #             cur_col_id = None
    #
    #         # hanging column with no aggregate means we don't know
    #         if cur_col_id:
    #             aggcol = AggregatedColumn()
    #             aggcol.col_id = cur_col_id
    #             aggcol.has_agg = UNKNOWN
    #             pq.select.append(aggcol)

    # def copy(self):
    #     copied = Query()
    #     copied.set_op = self.set_op
    #     if self.left:
    #         copied.left = self.left.copy()
    #     if self.right:
    #         copied.right = self.right.copy()
    #
    #     if isinstance(self.select, list):
    #         copied.select = list(self.select)
    #     else:
    #         copied.select = self.select
    #
    #     if isinstance(self.where, list):
    #         where = []
    #         for item in self.where:
    #             if isinstance(item, Query):
    #                 where.append(item.copy())
    #             else:
    #                 where.append(item)
    #         copied.where = where
    #     else:
    #         copied.where = self.where
    #
    #     if isinstance(self.group_by, list):
    #         copied.group_by = list(self.group_by)
    #     else:
    #         copied.group_by = self.group_by
    #
    #     if isinstance(self.having, list):
    #         having = []
    #         for item in self.having:
    #             if isinstance(item, Query):
    #                 having.append(item.copy())
    #             else:
    #                 having.append(item)
    #         copied.having = having
    #     else:
    #         copied.having = self.having
    #
    #     if isinstance(self.order_by, list):
    #         copied.order_by = list(self.order_by)
    #     else:
    #         copied.order_by = self.order_by
    #
    #     copied.limit = self.limit
    #     return copied
    #
    # # in format expected by SyntaxSQLNet
    # def as_dict(self, sql_key=True):
    #     if self.set_op is not None and self.set_op != 'none':
    #         sql = {
    #             'sql': self.left.as_dict(sql_key=False),
    #             'nested_sql': self.right.as_dict(sql_key=False),
    #             'nested_label': self.set_op
    #         }
    #     else:
    #         sql = {
    #             'select': self.select
    #         }
    #
    #         if isinstance(self.where, list):
    #             where = []
    #             for item in self.where:
    #                 if isinstance(item, Query):
    #                     where.append(item.as_dict())
    #                 else:
    #                     where.append(item)
    #             sql['where'] = where
    #
    #         if self.group_by:
    #             sql['groupBy'] = self.group_by
    #
    #         if isinstance(self.having, list):
    #             having = []
    #             for item in self.having:
    #                 if isinstance(item, Query):
    #                     having.append(item.as_dict())
    #                 else:
    #                     having.append(item)
    #             sql['having'] = having
    #
    #         if self.order_by:
    #             sql['orderBy'] = self.order_by
    #
    #         # add another 'sql' layer in outermost layer
    #         if sql_key:
    #             sql = {
    #                 'sql': sql
    #             }
    #
    #     return sql
