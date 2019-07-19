import argparse
import ConfigParser
import json
from multiprocessing.connection import Listener
import re
import torch
import traceback

from process_sql import tokenize
from supermodel import SuperModel
from utils import load_word_emb

from modules.client import DuoquestClient
from modules.database import Database
from modules.query import generate_sql_str
from modules.task_pb2 import ProtoTask, ProtoCandidates

def load_schemas(schemas_path):
    data = json.load(open(schemas_path))
    schemas = {}
    for item in data:
        schemas[item['db_id']] = item
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

def translate(id, model, db, schemas, client, db_name, nlq, n, b, tsq_level,
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
        cqs = model.enumerate(id, db, [tokens] * 2, [], schema, client, n,
            b, tsq_level, timeout=timeout, debug=debug,
            fake_literals=fake_literals)

        for cq in cqs:
            results.append(generate_sql_str(cq.pq, cq.schema))

    return results

def get_dataset_paths(config, dataset, mode):
    schemas_path = None
    db_path = None
    if dataset == 'spider':
        schemas_path = config.get('spider',
            '{}_tables_path'.format(mode))
        db_path = config.get('spider',
            '{}_db_path'.format(mode))
    elif dataset == 'wikisql':
        pass  # TODO
    return schemas_path, db_path

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset', choices=['spider', 'wikisql'])
    # parser.add_argument('mode', choices=['dev', 'test'])

    parser.add_argument('--config_path', default='../../src/config.ini')
    parser.add_argument('--toy', action='store_true',
        help='Use toy word embedding set to save load time')
    parser.add_argument('--debug', action='store_true',
        help='Enable debug output')
    # parser.add_argument('--test_manual', action='store_true',
    #     help='For manual command line testing')
    parser.add_argument('--test_path', help='Path for dataset to test')

    args = parser.parse_args()

    config = ConfigParser.RawConfigParser()
    config.read(args.config_path)

    model = load_model(config.get('syntaxsql', 'models_path'),
        config.get('syntaxsql', 'glove_path'), args.toy)
    client = DuoquestClient(int(config.get('duoquest', 'port')),
        config.get('duoquest', 'authkey'))

    # if args.test_manual:
    #     n = 1
    #     b = 1
    #     test(model, db, schemas, client, n, b, args.debug, timeout=args.timeout)
    #     exit()

    if args.test_path:
        schemas_path, db_path = \
            get_dataset_paths(config, 'spider', 'dev')
        schemas = load_schemas(schemas_path)
        db = Database(db_path, 'spider')
        data = json.load(open(args.test_path))
        test_old_and_new(data, model, db, schemas, 10, 1, debug=args.debug)
        exit()

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

                schemas_path, db_path = \
                    get_dataset_paths(config, task.dataset, task.mode)

                schemas = load_schemas(schemas_path)
                db = Database(db_path, task.dataset)

                tokens_list = list(task.nlq_tokens)
                nlq = None
                if len(tokens_list) == 1:
                    nlq = tokens_list[0]
                else:
                    nlq = tokens_list

                dqc = client if task.tsq_level != 'no_duoquest' else None

                sqls = translate(task.id, model, db, schemas, dqc, task.db_name,
                    nlq, task.n, task.b, task.tsq_level, timeout=task.timeout,
                    debug=args.debug)

                proto_cands = ProtoCandidates()
                for sql in sqls:
                    proto_cands.cq.append(sql)
                conn.send_bytes(proto_cands.SerializeToString())
        except Exception as e:
            traceback.print_exc()
        finally:
            listener.close()

def new_to_old(new):
    return re.sub('(w[0-9]+|I|E|U)(t[0-9]+)', '\g<2>', new)

def test_old_and_new(data, model, db, schemas, n, b, debug=False):
    correct = 0
    for i, task in enumerate(data):
        print('{}/{} || {}, {}'.format(i+1, len(data), task['db_id'],
            task['question_toks']))
        dqc = None
        old = translate(i+1, model, db, schemas, dqc, task['db_id'],
            task['question_toks'], n, b, _old=True)
        new = translate(i+1, model, db, schemas, dqc, task['db_id'],
            task['question_toks'], n, b, debug=debug, fake_literals=True)

        if new and old and new_to_old(new[0]).lower() == old[0].lower():
            correct += 1
            print('Correct!\n')
        else:
            print(old)
            print(new)
            print('Incorrect!\n')
    print('Correct: {}/{}'.format(correct, len(data)))

# def test(model, db, schemas, client, n, b, debug, timeout=None):
#     while True:
#         db_name = raw_input('Database (hit enter for default) > ')
#         if not db_name:
#             db_name = 'course_teach'
#         print('Database: {}'.format(db_name))
#
#         nlq = raw_input('NLQ (hit enter for default) > ')
#         if not nlq:
#             nlq = [u'Show', u'the', u'hometowns', u'shared', u'by', u'at',
#                 u'least', u'two', u'teachers', u'.']
#         print('NLQ: {}'.format(nlq))
#
#         old = translate(model, db, schemas, client, db_name, nlq, n, b,
#             _old=True, debug=debug)
#         print('--- OLD ---')
#         for cq in old:
#             print(u' - {}'.format(cq))
#         print
#
#         new = translate(model, db, schemas, client, db_name, nlq, n, b,
#             timeout=timeout, debug=debug)
#         print('--- NEW ---')
#
#         for cq in new:
#             print(u' - {}'.format(cq))
#         print

if __name__ == '__main__':
    main()
