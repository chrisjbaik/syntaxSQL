import argparse
import json
from multiprocessing.connection import Listener
import torch

from process_sql import tokenize
from supermodel import SuperModel
from utils import load_word_emb

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

def translate(model, schemas, db_name, nlq, n, b, _old=False, debug=False):
    if db_name not in schemas:
        raise Exception("Error: %s not in schemas" % db_name)

    schema = schemas[db_name]

    if isinstance(nlq, list):
        tokens = nlq
    else:
        tokens = tokenize(nlq)

    # 06/13/2019: not sure why multiply by 2 is necessary for tokens
    results = []
    if _old:
        cq = model.full_forward([tokens] * 2, [], schema)
        results.append(model.gen_sql(cq, schemas[db_name]))
    else:
        cqs = model.dfs_beam_search([tokens] * 2, [], schema, n, b, debug=debug)

        for cq in cqs:
            results.append(model.gen_sql(cq.as_dict(), schemas[db_name]))

    return results

# Listens for: `{db_name}\t{NLQ}' (plaintext) on port 6000
# Response: `{SQL}\t{SQL}\t{SQL}...`
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=6000)
    parser.add_argument('--authkey', default='mixtape')
    parser.add_argument('--schemas_path',
        default='../../data/spider/tables.json')
    parser.add_argument('--models_path',
        default='generated_data_augment/saved_models')
    parser.add_argument('--glove_path', default='glove')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--n', default=1, type=int,
        help='Max number of final queries to output')
    parser.add_argument('--b', default=1, type=int,
        help='Beam search parameter')

    parser.add_argument('--test_manual', action='store_true',
        help='For manual command line testing')
    parser.add_argument('--test_path', help='Path for dataset to test')
    parser.add_argument('--debug', help='Enable debug output for test_manual')

    args = parser.parse_args()

    print('Running with n = {} and b = {}'.format(args.n, args.b))

    schemas = load_schemas(args.schemas_path)
    model = load_model(args.models_path, args.glove_path, args.toy)

    if args.test_manual:
        test(model, schemas, args.n, args.b, args.debug)
        exit()
    elif args.test_path:
        data = json.load(open(args.test_path))
        test_old_and_new(data, model, schemas, args.n, args.b)
        exit()

    while True:
        address = ('localhost', args.port)  # family is deduced to be 'AF_INET'
        listener = Listener(address, authkey=args.authkey)
        print('Listening on port {}...'.format(args.port))
        conn = listener.accept()
        print('Connection accepted from:', listener.last_accepted)
        while True:
            msg = conn.recv_bytes()

            if msg == 'close':
                conn.close()
                break

            db_name, nlq = msg.split('\t')
            sqls = translate(model, schemas, db_name, nlq, args.n, args.b)
            conn.send_bytes('\t'.join(sqls))
        listener.close()

def test_old_and_new(data, model, schemas, n, b):
    correct = 0
    for task in data:
        print('{}, {}'.format(task['db_id'], task['question_toks']))
        old = translate(model, schemas, task['db_id'], task['question_toks'],
            n, b, _old=True)
        new = translate(model, schemas, task['db_id'], task['question_toks'],
            n, b)
        if new[0] == old[0]:
            correct += 1
            print('Correct!\n')
        else:
            print(old)
            print(new)
            print('Incorrect!\n')
    print('Correct: {}/{}'.format(correct, len(data)))

def test(model, schemas, n, b, debug):
    while True:
        db_name = raw_input('Database (hit enter for default) > ')
        if not db_name:
            db_name = 'concert_singer'
        print('Database: {}'.format(db_name))

        nlq = raw_input('NLQ (hit enter for default) > ')
        if not nlq:
            nlq = [u'What', u'is', u'the', u'average', u',', u'minimum', u',', u'and', u'maximum', u'age', u'of', u'all', u'singers', u'from', u'France', u'?']
        print('NLQ: {}'.format(nlq))

        old = translate(model, schemas, db_name, nlq, n, b, _old=True,
            debug=debug)
        print('--- OLD ---')
        for cq in old:
            print(' - {}'.format(cq))
        print

        new = translate(model, schemas, db_name, nlq, n, b, debug=debug)
        print('--- NEW ---')
        for cq in new:
            print(' - {}'.format(cq))
        print

if __name__ == '__main__':
    main()
