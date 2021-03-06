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

def translate(model, schemas, db_name, nlq):
    if db_name not in schemas:
        raise Exception("Error: %s not in schemas" % db_name)

    schema = schemas[db_name]

    tokens = tokenize(nlq)

    # 06/13/2019: not sure why multiply by 2 is necessary for tokens
    sql, conf = model.forward([tokens] * 2, [], schema)

    if sql is not None:
        sql = model.gen_sql(sql, schemas[db_name])
    else:
        sql = None

    return sql, conf

# listens for `{db_name}\t{NLQ}' in plaintext on port 6000
# responds with `{SQL}\t{confidence score}`
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=6000)
    parser.add_argument('--authkey', default='adversql')
    parser.add_argument('--schemas_path',
        default='../../data/spider/tables.json')
    parser.add_argument('--models_path',
        default='generated_data_augment/saved_models')
    parser.add_argument('--glove_path', default='glove')
    parser.add_argument('--toy', action='store_true')
    args = parser.parse_args()

    schemas = load_schemas(args.schemas_path)
    model = load_model(args.models_path, args.glove_path, args.toy)

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
            sql, conf = translate(model, schemas, db_name, nlq)
            conn.send_bytes('{}\t{}'.format(sql, conf))
        listener.close()

if __name__ == '__main__':
    main()
