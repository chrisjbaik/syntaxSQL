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

def translate(model, schemas, db_name, nlq, n, b):
    if db_name not in schemas:
        raise Exception("Error: %s not in schemas" % db_name)

    schema = schemas[db_name]

    tokens = tokenize(nlq)

    # 06/13/2019: not sure why multiply by 2 is necessary for tokens
    cqs = model.forward([tokens] * 2, [], schema, n, b)

    results = []
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
    parser.add_argument('--n', default=10, type=int,
        help='Max number of final queries to output')
    parser.add_argument('--b', default=5, type=int,
        help='Beam search parameter')
    parser.add_argument('--test', action='store_true', help='For sanity check')
    args = parser.parse_args()

    schemas = load_schemas(args.schemas_path)
    model = load_model(args.models_path, args.glove_path, args.toy)

    if args.test:
        test(model, schemas, args.n, args.b)
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

def test(model, schemas, n, b):
    while True:
        input = input('Test (hit enter for default) > ')
        if not input:
            db_name = 'concert_singer'
            nlq = 'How many singers do we have?'
        else:
            db_name, nlq = input.split('\t')
        print('Database: {}'.format(db_name))
        print('NLQ: {}'.format(nlq))
        
        sqls = translate(model, schemas, db_name, nlq, n, b)
        for sql in sqls:
            print(sql)

if __name__ == '__main__':
    main()
