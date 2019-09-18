import argparse
import ConfigParser
import json
from multiprocessing.connection import Listener
import re
import sqlite3
import torch
import traceback

from process_sql import tokenize
from supermodel import SuperModel
from utils import load_word_emb

from modules.client import DuoquestClient, StopException
from modules.database import Database
from modules.query import generate_sql_str
from modules.duoquest_pb2 import ProtoSchema, ProtoTask, ProtoCandidates, \
    COL_TEXT, COL_NUMBER, COL_TIME, COL_BOOLEAN

def proto_col_type_to_text(proto_col_type):
    if proto_col_type == COL_TEXT:
        return 'text'
    elif proto_col_type == COL_NUMBER:
        return 'number'
    elif proto_col_type == COL_TIME:
        return 'time'
    elif proto_col_type == COL_BOOLEAN:
        return 'boolean'
    else:
        raise Exception('Unrecognized type: {}'.format(proto_col_type))

def load_schemas(schemas_path):
    data = json.load(open(schemas_path))
    schemas = {}
    for item in data:
        schemas[item['db_id']] = item
    return schemas

def load_schemas_from_proto(schema_proto_str):
    schema_proto = ProtoSchema()
    schema_proto.ParseFromString(schema_proto_str)
    schema = {}

    schema['db_id'] = schema_proto.name
    schema['table_names'] = []
    schema['table_names_original'] = []
    schema['primary_keys'] = []

    column_names = [(0, [-1, '*'])]
    column_names_original = [(0, [-1, '*'])]
    column_types = [(0, 'text')]
    for table_id, table in enumerate(schema_proto.tables):
        schema['table_names'].append(table.sem_name)
        schema['table_names_original'].append(table.syn_name)

        for col in table.columns:
            if col.is_pk:
                schema['primary_keys'].append(col.id)
            column_names.append((col.id, [table_id, col.sem_name]))
            column_names_original.append((col.id, [table_id, col.syn_name]))
            column_types.append((col.id, proto_col_type_to_text(col.type)))

    schema['column_names'] = map(lambda x: x[1],
        sorted(column_names, key=lambda x: x[0]))
    schema['column_names_original'] = map(lambda x: x[1],
        sorted(column_names_original, key=lambda x: x[0]))
    schema['column_types'] = map(lambda x: x[1],
        sorted(column_types, key=lambda x: x[0]))

    schema['foreign_keys'] = []
    for fkpk in schema_proto.fkpks:
        schema['foreign_keys'].append([fkpk.fk_col_id, fkpk.pk_col_id])

    schemas = {}
    schemas[schema_proto.name] = schema
    return schemas

def load_model(models_path, glove_path, toy=False):
    ### CONFIGURABLE
    GPU = True           # GPU activated
    B_word = 42          # GloVE corpus size
    N_word = 300         # word embedding dimension
    N_h = 300            # hidden layer size
    N_depth = 2          # num LSTM layers

    print("Loading GloVE word embeddings...")
    word_emb = load_word_emb('{}/glove.{}B.{}d.txt'.format(glove_path,
        B_word, N_word), load_used=False, use_small=toy)

    model = SuperModel(word_emb, N_word=N_word, gpu=GPU, trainable_emb=False,
        table_type='std', use_hs=True)

    print("Loading trained models...")
    model.multi_sql.load_state_dict(
        torch.load("{}/multi_sql_models.dump".format(models_path)))
    model.key_word.load_state_dict(
        torch.load("{}/keyword_models.dump".format(models_path)))
    model.col.load_state_dict(
        torch.load("{}/col_models.dump".format(models_path)))
    model.op.load_state_dict(
        torch.load("{}/op_models.dump".format(models_path)))
    model.agg.load_state_dict(
        torch.load("{}/agg_models.dump".format(models_path)))
    model.root_teminal.load_state_dict(
        torch.load("{}/root_tem_models.dump".format(models_path)))
    model.des_asc.load_state_dict(
        torch.load("{}/des_asc_models.dump".format(models_path)))
    model.having.load_state_dict(
        torch.load("{}/having_models.dump".format(models_path)))
    return model

def translate(id, model, db, schemas, client, db_name, nlq, literals,
    timeout=None, _old=False, debug=False, fake_literals=False):
    if db_name not in schemas:
        raise Exception("Error: %s not in schemas" % db_name)

    schema = schemas[db_name]

    if isinstance(nlq, list):
        tokens = nlq
    else:
        tokens = tokenize(nlq)

    results = []
    if _old:
        cq = model.full_forward([tokens] * 2, [], schema)
        results.append(model.gen_sql(cq, schemas[db_name]))
    else:
        if debug:
            print('Database: {} || NLQ: {}'.format(db_name, nlq))
            print('LITERALS')
            print(literals)
        cqs = model.search(id, db, [tokens] * 2, literals, [], schema, client,
            timeout=timeout, debug=debug, fake_literals=fake_literals)

        for cq in cqs:
            results.append(cq.pq)

    return results

def get_dataset_paths(config, dataset, mode):
    schemas_path = None
    db_path = None
    if dataset == 'spider':
        schemas_path = config.get('spider', '{}_tables_path'.format(mode))
        db_path = config.get('spider', '{}_db_path'.format(mode))
    elif dataset == 'wikisql':
        pass  # TODO
    return schemas_path, db_path

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', default='../../src/config.ini')
    parser.add_argument('--toy', action='store_true',
        help='Use toy word embedding set to save load time')
    parser.add_argument('--debug', action='store_true',
        help='Enable debug output')

    args = parser.parse_args()

    config = ConfigParser.RawConfigParser()
    config.read(args.config_path)

    model = load_model(config.get('syntaxsql', 'models_path'),
        config.get('syntaxsql', 'glove_path'), args.toy)
    client = DuoquestClient(int(config.get('duoquest', 'port')),
        config.get('duoquest', 'authkey'))

    while True:
        try:
            address = ('localhost', int(config.get('nlq', 'port')))
            listener = Listener(address, authkey=config.get('nlq', 'authkey'))
            print('Listening on port {}...'.format(config.get('nlq', 'port')))
            conn = listener.accept()
            print('Connection accepted from:', listener.last_accepted)
            while True:
                msg = conn.recv_bytes()

                if msg == 'close':
                    conn.close()
                    break

                task = ProtoTask()
                task.ParseFromString(msg)

                if task.dataset and task.mode:
                    schemas_path, db_path = \
                        get_dataset_paths(config, task.dataset, task.mode)

                    schemas = load_schemas(schemas_path)
                    db = Database(db_path, task.dataset)
                else:
                    task_db_path = config.get('db', 'path')

                    task_conn = sqlite3.connect(task_db_path)
                    cur = task_conn.cursor()
                    cur.execute('''SELECT schema_proto, path FROM databases
                                   WHERE name = ?''', (task.db_name,))
                    row = cur.fetchone()
                    if row is None:
                        raise Exception('Database <{}> not found!'.format(
                            task.db_name))

                    schema_proto, db_path = row
                    task_conn.close()

                    schemas = load_schemas_from_proto(schema_proto)
                    db = Database(db_path, None, db_name=task.db_name)

                tokens_list = list(task.nlq_tokens)
                nlq = None
                if len(tokens_list) == 1:
                    nlq = tokens_list[0]
                else:
                    nlq = tokens_list

                client.tsq_level = task.tsq_level

                client.init_cache()
                client.connect()

                try:
                    sqls = translate(task.id, model, db, schemas, client,
                        task.db_name, nlq, task.literals, timeout=task.timeout,
                        debug=args.debug)
                except StopException as e:
                    sqls = []
                finally:
                    proto_cands = ProtoCandidates()
                    for sql in sqls:
                        proto_cands.cqs.append(sql)
                    conn.send_bytes(proto_cands.SerializeToString())
                    client.close()
        except Exception as e:
            traceback.print_exc()
        finally:
            listener.close()

if __name__ == '__main__':
    main()
