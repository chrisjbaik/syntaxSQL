from duoquest_pb2 import *
from schema import JoinEdge

def to_str_tribool(proto_tribool):
    if proto_tribool == UNKNOWN:
        return None
    elif proto_tribool == TRUE:
        return True
    else:
        return False

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

    # single table case, no aliases
    if len(tables) == 1:
        join_exprs.append(u'{}'.format(tbl.syn_name))
        return u' '.join(join_exprs), aliases

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
        if pred.has_subquery == TRUE:
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
    group_by_cols = []
    for col_id in pq.group_by:
        group_by_cols.append(schema.get_aliased_col(aliases, col_id))
    return u'GROUP BY {}'.format(u', '.join(group_by_cols))

def having_clause_str(pq, schema, aliases):
    having_exprs = ['HAVING']
    for i, pred in enumerate(pq.having.predicates):
        if i != 0:
            having_exprs.append(to_str_logical_op(pq.having.logical_op))

        assert(pred.has_agg == TRUE)

        having_col = u'{}({})'.format(
            to_str_agg(pred.agg),
            schema.get_aliased_col(aliases, pred.col_id)
        )

        having_val = None
        if pred.has_subquery == TRUE:
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
    order_by_cols = []
    for ordered_col in pq.order_by:
        if ordered_col.agg_col.has_agg == TRUE:
            order_by_cols.append('{}({}) {}'.format(
                to_str_agg(ordered_col.agg_col.agg),
                schema.get_aliased_col(aliases, ordered_col.agg_col.col_id),
                to_str_dir(ordered_col.dir)
            ))
        else:
            order_by_cols.append('{} {}'.format(
                schema.get_aliased_col(aliases, ordered_col.agg_col.col_id),
                to_str_dir(ordered_col.dir)
            ))
    return u'ORDER BY {}'.format(u', '.join(order_by_cols))

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
            generate_sql_str(pq.left, schema),
            set_op_str,
            generate_sql_str(pq.right, schema, alias_prefix=set_op_str[0])
        )

    from_clause, aliases = from_clause_str(pq, schema, alias_prefix)
    if from_clause is None:
        raise Exception('FROM clause not generated.')

    clauses = []
    clauses.append(select_clause_str(pq, schema, aliases))
    clauses.append(from_clause)
    if pq.has_where == TRUE and pq.where.predicates:
        clauses.append(where_clause_str(pq, schema, aliases))
    if pq.has_group_by == TRUE and pq.group_by:
        clauses.append(group_by_clause_str(pq, schema, aliases))
    if pq.has_having == TRUE and pq.having.predicates:
        clauses.append(having_clause_str(pq, schema, aliases))
    if pq.has_order_by == TRUE and pq.order_by:
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
def join_path_needs_update(schema, pq):
    tables_in_cur_jp = set(map(lambda x: schema.get_table(x),
        pq.from_clause.edge_map.keys()))

    # if SELECT has a column (i.e. inference started) and there are no tables
    if pq.select and len(tables_in_cur_jp) == 0:
        return True

    # if the current join path doesn't account for all tables in protoquery
    tables = get_tables(schema, pq)
    if tables_in_cur_jp >= tables:
        return False
    else:
        return True

def with_updated_join_paths(schema, pq, minimal_join_paths=False):
    for agg_col in pq.select:
        if agg_col.agg == COUNT and agg_col.col_id == 0:
            minimal_join_paths = False
    jps = schema.get_join_paths(get_tables(schema, pq),
        minimal_join_paths=minimal_join_paths)

    new_pqs = []
    for jp in jps:
        new_pq = ProtoQuery()
        new_pq.CopyFrom(pq)
        set_proto_from(new_pq, jp)
        new_pqs.append(new_pq)
    return new_pqs

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
        self.pq = protoquery

    def copy(self):
        new_query = Query(self.schema)
        new_query.pq.CopyFrom(self.pq)
        return new_query
