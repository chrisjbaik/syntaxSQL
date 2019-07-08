from nltk import everygrams
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer

from word2number import w2n

stopwords_bank = set(stopwords.words('english'))

def to_number(tok):
    try:
        val = float(tok)
        if val.is_integer():
            return int(val)
        else:
            return val
    except ValueError:
        try:
            val = w2n.word_to_num(str(tok))
            return val
        except Exception:
            return None

def find_literal_candidates(nlq_toks, db, schema, col_id, cache, b, agg=None,
    like=False):
    col = schema.get_col(col_id)

    # no literals for *
    if col.syn_name == '*':
        return []

    if agg == 'count' or col.type == 'number':
        cached = cache.get('_num')
        if cached:
            return cached
        else:
            cands = []
            for tok in nlq_toks:
                val = to_number(tok)
                if val is not None:
                    cands.append(val)
            cands.sort()        # ascending for between queries
            lits = list(map(lambda x: str(x), cands))
            cache.set('_num', lits)
    else:
        cached = cache.get(col_id)
        if cached:
            return cached
        else:
            lits = find_string_literals(nlq_toks, db, schema.db_id,
                col.table.syn_name, col.syn_name, b, like=like)
            cache.set(col_id, lits)

    if not lits:
        print('Warning: no literals for {}.{}'.format(
            col.table.syn_name, col.syn_name
        ))

    return lits

def find_string_literals(nlq_toks, db, db_name, tbl_name, col_name, b, like):
    d = TreebankWordDetokenizer()
    ngrams = list(everygrams(nlq_toks, min_len=1, max_len=6))

    lits = []
    for ngram in reversed(ngrams):
        # HACK: single length tokens sometimes include random punctuation
        if len(ngram) == 1:
            ngram = (ngram[0].replace("'", ''),)

        token_str = d.detokenize(ngram)
        if token_str in stopwords_bank:
            continue

        lit = db.find_literals(db_name, tbl_name, col_name, token_str, b)
        if lit:
            if like:
                lits.append(u'%{}%'.format(token_str))
            else:
                lits.extend(map(lambda x: unicode(x), lit))
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
