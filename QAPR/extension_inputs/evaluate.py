"""Evaluate TREC runs with trec_eval semantics (pytrec_eval): Recall/NDCG/MAP @10,100,1000."""
import sys, collections, pytrec_eval

def load_qrels(path):
    q = collections.defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) >= 4:
            q[p[0].strip()][p[2].strip()] = int(p[3])
    return q

def load_run(path, drop_self=True, use_rank=True):
    run = collections.defaultdict(dict)
    for line in open(path):
        p = line.split()
        if len(p) < 4: continue
        qid, did, rank = p[0], p[2], int(p[3])
        if drop_self and qid == did: continue
        # score = -rank so trec_eval ordering follows the file's rank column exactly
        run[qid][did] = -float(rank) if use_rank else float(p[4])
    return run

MEAS = {'recall.10,100,1000', 'ndcg_cut.10,100,1000', 'map_cut.10,100,1000'}
KEYS = ['recall_10','recall_100','recall_1000','ndcg_cut_10','ndcg_cut_100','ndcg_cut_1000','map_cut_10','map_cut_100','map_cut_1000']

def evaluate(qrels, run, topics=None):
    topics = topics or sorted(run)
    qrels = {t: qrels[t] for t in topics if t in qrels}
    ev = pytrec_eval.RelevanceEvaluator(qrels, MEAS)
    res = ev.evaluate({t: run.get(t, {}) for t in qrels})
    # queries with empty runs score 0 (trec_eval -c behaviour)
    agg = {k: sum(res.get(t, {}).get(k, 0.0) for t in qrels) / len(qrels) for k in KEYS}
    return agg, res

if __name__ == '__main__':
    qrels = load_qrels(sys.argv[1])
    for f in sys.argv[2:]:
        run = load_run(f)
        agg, _ = evaluate(qrels, run)
        print(f.split('/')[-1], len(run), ' '.join(f'{k}={agg[k]:.4f}' for k in KEYS))
