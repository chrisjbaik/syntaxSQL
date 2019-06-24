from nltk import everygrams
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

stopwords_bank = set(stopwords.words('english'))

def to_number(str):
    try:
        val = float(str)
        if val.is_integer():
            return int(val)
        else:
            return val
    except ValueError:
        return None

def get_col_info(schema, col_id):
    col_type = schema['column_types'][col_id]
    col_name = schema['column_names_original'][col_id][1]
    tbl_id = schema['column_names_original'][col_id][0]
    tbl_name = schema['table_names_original'][tbl_id]

    return tbl_name, col_name, col_type

def find_literal_candidates(nlq_toks, db, schema, col_id, cache, b):
    tbl_name, col_name, col_type = get_col_info(schema, col_id)
    if col_type == 'number':
        cands = []
        for tok in nlq_toks:
            val = to_number(tok)
            if val is not None:
                cands.append(val)
        cands.sort()        # ascending for between queries
        return cands
    else:
        cached = cache.get(col_id)
        if cached:
            return cached
        else:
            lits = find_string_literals(nlq_toks, db, schema['db_id'], tbl_name,
                col_name, b)
            cache.set(col_id, lits)
            return lits

def find_string_literals(nlq_toks, db, db_name, tbl_name, col_name, b):
    d = TreebankWordDetokenizer()
    ngrams = list(everygrams(nlq_toks, min_len=1, max_len=6))

    lits = []
    for ngram in reversed(ngrams):
        # HACK: single length tokens sometimes include random punctuation
        if len(ngram) == 1:
            ngram = (ngram[0].replace("'", ''),)

        str = d.detokenize(ngram)
        if str in stopwords_bank:
            continue

        lit = db.find_literals(db_name, tbl_name, col_name, str, b)
        if lit:
            lits.extend(lit)
    return lits

class LiteralsCache(object):
    def __init__(self):
        # col_id -> set()
        self.cache = {}

    def get(self, col_id):
        if col_id in self.cache:
            return list(self.cache[col_id])
        return None

    def set(self, col_id, lits):
        self.cache[col_id] = set(lits)
