from Queue import Queue

class FromClause(object):
    def __init__(self, aliases, clause, distinct=False):
        self.aliases = aliases
        self.clause = clause
        self.distinct = distinct

class JoinPath(object):
    def __init__(self):
        # If SELECT should be distinct
        self.distinct = False

        self.edges = list()

        # table -> [ join edges ]
        self.edge_map = {}

    def add_single_table(self, table):
        if table not in self.edge_map:
            self.edge_map[table] = []

    def __len__(self):
        return len(self.edges)

    def copy(self):
        new_jp = JoinPath()
        for table, edge_list in self.edge_map.items():
            new_jp.edge_map[table] = list(edge_list)
        new_jp.edges = list(self.edges)
        return new_jp

    def add_edge(self, edge):
        # base case
        fk_table = edge.fk_col.table
        pk_table = edge.pk_col.table

        if len(self.edge_map) == 0:
            self.edge_map[fk_table] = [edge]
            self.edge_map[pk_table] = [edge]
            self.edges.append(edge)
        else:
            # TODO: prevent f -> p <- f join paths
            if fk_table in self.edge_map and pk_table in self.edge_map:
                raise Exception('Cannot generate cycle in JoinPath.')
            elif fk_table in self.edge_map:
                self.edge_map[fk_table].append(edge)
                self.edge_map[pk_table] = [edge]
                self.edges.append(edge)
            elif pk_table in self.edge_map:
                self.edge_map[pk_table].append(edge)
                self.edge_map[fk_table] = [edge]
                self.edges.append(edge)
            else:
                raise Exception('Edge does not link with existing JoinPath.')

    def merge(self, other):
        edges_left = list(other.edges)
        while edges_left:
            edge = edges_left.pop(0)
            if edge not in self.edges:
                try:
                    self.add_edge(edge)
                except Exception:
                    edges_left.append(edge)

class JoinEdge(object):
    def __init__(self, fk_col, pk_col):
        self.fk_col = fk_col
        self.pk_col = pk_col

    def key(self, tbl):
        if self.fk_col.table == tbl:
            return self.fk_col
        elif self.pk_col.table == tbl:
            return self.pk_col
        else:
            raise Exception('Table <{}> not in this JoinEdge.'.format(tbl.syn_name))

    def other(self, tbl):
        if self.fk_col.table == tbl:
            return self.pk_col.table
        elif self.pk_col.table == tbl:
            return self.fk_col.table
        else:
            raise Exception('Table <{}> not in this JoinEdge.'.format(tbl.syn_name))

    def __str__(self):
        return '{} -> {}'.format(str(self.fk_col), str(self.pk_col))

    def __hash__(self):
        return hash((self.fk_col, self.pk_col))

class Table(object):
    def __init__(self, tbl_id, sem_name, syn_name):
        self.id = tbl_id
        self.sem_name = sem_name
        self.syn_name = syn_name
        self.columns = []

        self.fk_edges = []
        self.pk_edges = []

    def num_cols(self):
        return len(self.columns)

    def add_col(self, col):
        self.columns.append(col)

    def add_fk_edge(self, fk_edge):
        self.fk_edges.append(fk_edge)

    def add_pk_edge(self, pk_edge):
        self.pk_edges.append(pk_edge)

    def __hash__(self):
        return hash((self.id, self.sem_name))

    def __str__(self):
        return self.sem_name

class Column(object):
    def __init__(self, col_id, table, col_type, sem_name, syn_name, pk=False,
        fk=False, fk_ref=None):
        self.id = col_id
        self.table = table
        self.type = col_type
        self.sem_name = sem_name
        self.syn_name = syn_name
        self.pk = pk
        self.fk = fk
        self.fk_ref = fk_ref   # id of PK referred to if this is FK

    def set_fk(self, fk):
        self.fk = fk

    def set_fk_ref(self, fk_ref):
        self.fk = True
        self.fk_ref = fk_ref

    def __str__(self):
        if self.syn_name == '*':
            return '*'
        else:
            return '{}.{}'.format(self.table.sem_name, self.sem_name)

    def __hash__(self):
        return hash((self.id, self.table, self.sem_name))

class Schema(object):
    def __init__(self, schema_info):
        self.db_id = schema_info['db_id']
        self.tables = []
        self.columns = []

        tbl_syn_names = schema_info['table_names_original']
        tbl_sem_names = schema_info['table_names']

        for i, sem_name in enumerate(tbl_sem_names):
            tbl = Table(i, sem_name, tbl_syn_names[i])
            self.tables.append(tbl)

        col_syn_names = schema_info['column_names_original']
        col_sem_names = schema_info['column_names']

        for i, col_info in enumerate(col_sem_names):
            tbl_id, sem_name = col_info
            syn_name = col_syn_names[i][1]
            col_type = schema_info['column_types'][i]
            pk = i in schema_info['primary_keys']

            if syn_name == '*':
                col = Column(i, None, col_type, 'all', syn_name, pk=pk)
            else:
                tbl = self.get_table(tbl_id)
                col = Column(i, tbl, col_type, sem_name, syn_name, pk=pk)
                tbl.add_col(col)
            self.columns.append(col)

        for fk, pk in schema_info['foreign_keys']:
            fk_col = self.get_col(fk)
            fk_col.set_fk_ref(pk)
            pk_col = self.get_col(pk)

            edge = JoinEdge(fk_col, pk_col)
            fk_col.table.add_fk_edge(edge)
            pk_col.table.add_pk_edge(edge)

    def pk_ids(self):
        ids = []
        for col in self.columns:
            if col.pk:
                ids.append(col.id)
        return ids

    def fk_ids(self):
        ids = []
        for col in self.columns:
            if col.fk:
                ids.append(col.id)
        return ids

    def num_cols(self):
        return len(self.columns)

    def get_table(self, tbl_id):
        return self.tables[tbl_id]

    def get_col(self, col_id):
        return self.columns[col_id]

    def get_shortest_paths(self, tables):
        # frozenset(table1, table2) -> JoinPath
        shortest = {}
        for tbl in tables:
            tbls_left = set(tables)
            tbls_left.remove(tbl)

            to_remove = set()
            for other_tbl in tbls_left:
                if frozenset([tbl, other_tbl]) in shortest:
                    to_remove.add(other_tbl)
            tbls_left -= to_remove

            # perform breadth-first search
            # TODO: change to more efficient shortest path alg if needed
            queue = Queue()
            queue.put((tbl, JoinPath()))
            visited = set()
            visited.add(tbl)

            while not queue.empty():
                if not tbls_left:
                    break

                cur_tbl, jp = queue.get_nowait()
                all_edges = cur_tbl.fk_edges + cur_tbl.pk_edges
                for edge in all_edges:
                    other_tbl = edge.other(cur_tbl)

                    if other_tbl in visited:
                        continue

                    new_jp = jp.copy()
                    new_jp.add_edge(edge)

                    if other_tbl in tbls_left:
                        key = frozenset([tbl, other_tbl])
                        shortest[key] = new_jp
                        tbls_left.remove(other_tbl)

                    queue.put((other_tbl, new_jp))
                    visited.add(other_tbl)

        return shortest

    def steiner(self, tables):
        # TODO: later, produce a self-join (!) if there are duplicate
        #       projections with same agg/col (bleghhhhhhh)

        # STEP 1: Get shortest paths between each table in col_idxs
        # shortest: frozenset(table1, table2) -> JoinPath
        # TODO: extend to get multiple shortest paths per pair if needed
        shortest = self.get_shortest_paths(tables)

        # STEPS 2-3: Get MST of shortest, replace shortest paths with join edges
        # TODO: extend to get multiple MSTs if needed
        tbls_in = set([next(iter(tables))])

        mst = JoinPath()

        while len(tbls_in) < len(tables):
            min_path_len = 9999
            min_path = None
            min_path_tbl = None
            for tbl in tbls_in:
                for other_tbl in tables:
                    if other_tbl in tbls_in:
                        continue
                    key = frozenset([tbl, other_tbl])
                    if key not in shortest:
                        raise Exception('Join path fail ({}) <{}>, <{}>'.format(
                            self.db_id, tbl.syn_name, other_tbl.syn_name
                        ))

                    if len(shortest[key]) < min_path_len:
                        min_path_len = len(shortest[key])
                        min_path = shortest[key]
                        min_path_tbl = other_tbl

            mst.merge(min_path)

            tbls_in.add(min_path_tbl)

        # TODO: omitting the following, if most cases are handled anyway
        # STEP 4: Find minimal spanning tree of `mst`
        # STEP 5: Delete edges so that all leaves are Steiner points

        return mst

    def get_join_paths(self, tables, minimal_join_paths=False):
        if len(tables) == 0:
            # when there's 0 tables (due to * being the only column),
            #   generate join path for each table in the schema
            return self.zero_table_join_paths()
        else:
            jps = []

            # first, get the default shortest join path
            if len(tables) == 1:
                jp = JoinPath()
                jp.add_single_table(table)
                jps.append(jp)
            else:
                jp = self.steiner(tables)

            # get alternative extensions with FK 2 layers deep
            if not minimal_join_paths:
                for table in tables:
                    if len(table.pk_edges) > 0:
                        for edge in table.pk_edges:
                            other_tbl = edge.other(table)
                            new_tables = list(tables)
                            new_tables.append(other_tbl)
                            jp = self.steiner(new_tables)
                            jp.distinct = True
                            jps.append(jp)

                            if len(other_tbl.pk_edges) > 0:
                                for edge in other_tbl.pk_edges:
                                    other2_tbl = edge.other(other_tbl)
                                    if other2_tbl not in tables:
                                        newer_tables = list(new_tables)
                                        newer_tables.append(other2_tbl)
                                        jp = self.steiner(newer_tables)
                                        jp.distinct = True
                                        jps.append(jp)
            return jps


    def zero_table_join_paths(self):
        jps = []
        for tbl in self.tables:
            jp = JoinPath()
            jp.add_single_table(tbl)
            jps.append(jp)
        return jps

    # def single_table_join_paths(self, table):
    #     join_paths = []
    #
    #     # Case 1: only use single table
    #     # jp = JoinPath()
    #     # jp.add_single_table(table)
    #     # join_paths.append(jp)
    #
    #     # Case 2: join with other table and count distinct ones
    #     if len(table.pk_edges) > 0:
    #         for edge in table.pk_edges:
    #             other_tbl = edge.other(table)
    #
    #             jp = self.steiner([table, other_tbl])
    #             jp.distinct = True
    #             join_paths.append(jp)
    #
    #             if len(other_tbl.pk_edges) > 0:
    #                 for edge in other_tbl.pk_edges:
    #                     other2_tbl = edge.other(other_tbl)
    #
    #                     if other2_tbl != table:
    #                         jp = self.steiner([table, other_tbl, other2_tbl])
    #                         jp.distinct = True
    #                         join_paths.append(jp)
    #
    #     return join_paths

    def get_aliased_col(self, aliases, col_idx):
        col = self.get_col(col_idx)
        if col.syn_name == '*':
            return '*'

        # change col if it is an fk and primary key is in table set already
        if col.fk and self.get_col(col.fk_ref).table.syn_name in aliases:
            col = self.get_col(col.fk_ref)

        if col.table.syn_name in aliases and aliases[col.table.syn_name]:
            return '{}."{}"'.format(aliases[col.table.syn_name],
                col.syn_name)
        else:
            return '"{}"'.format(col.syn_name)
