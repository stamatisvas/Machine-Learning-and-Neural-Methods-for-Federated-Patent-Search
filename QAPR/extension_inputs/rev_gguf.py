"""Open LLM pointwise rerankers via llama.cpp (Q4_K_M GGUF, CPU): score = logit(Yes) - logit(No) for
'is this candidate relevant prior art for the query patent?'. Query/doc = title + abstract, 230 tokens each.
Same 50 random test topics and BM25 top-20 as the Qwen3-Reranker run. usage: rev_gguf.py llama|mistral <run> <depth> <n>"""
import sys, json, time, random, numpy as np
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from rev_common import *
which, rname, depth, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
REPO = {'llama': ('bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'),
        'mistral': ('bartowski/Mistral-7B-Instruct-v0.3-GGUF', 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf')}[which]
path = hf_hub_download(*REPO)
llm = Llama(model_path=path, n_ctx=1024, n_threads=4, n_batch=512, verbose=False)
tok = lambda s: llm.tokenize(s.encode(), add_bos=False, special=True)
cut = lambda s, k: llm.detokenize(tok(s)[:k]).decode(errors='ignore')
ids = lambda words: sorted({tok(w)[0] for w in words})
YES, NO = ids(['Yes', ' Yes', 'yes']), ids(['No', ' No', 'no'])
if which == 'llama':
    wrap = lambda u: f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
else:
    wrap = lambda u: f'<s>[INST] {u} [/INST]'
def prompt(q, d):
    return wrap(f'You are a patent examiner searching for prior art.\n\nQuery patent:\n{q}\n\nCandidate document:\n{d}\n\n'
                'Is the candidate document relevant prior art that an examiner would cite against the query patent? Answer Yes or No.')
def score(text):
    t = tok(text); p = 0
    for a, b in zip(llm.input_ids[:llm.n_tokens], t[:-1]):
        if a != b: break
        p += 1
    llm.n_tokens = p; llm.eval(t[p:])
    lg = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(llm.n_vocab(),)).astype(np.float64)
    lse = lambda ix: np.log(np.exp(lg[ix] - lg.max()).sum())
    return float(lse(YES) - lse(NO))
topics = sorted(random.Random(0).sample(TEST, n))
run = read_run(RUNS + rname + '.txt', set(topics))
need = {d for q in run for d, _ in sorted(run[q].items(), key=lambda kv: -kv[1])[:depth]}
T = {t['id']: t['title'] + '. ' + t['abstract'] for t in map(json.loads, open('/home/user/data/topics.jsonl')) if t['id'] in run}
D = {}
for l in open('/home/user/data/docs.jsonl'):
    i = l[8:l.index('"', 8)]
    if i in need: t = json.loads(l); D[i] = t['title'] + '. ' + t['abstract']
t0 = time.time(); out = {}
for k, q in enumerate(topics):
    ranked = [d for d, _ in sorted(run[q].items(), key=lambda kv: -kv[1])]; qt = cut(T[q], 230)
    out[q] = {d: 100 + score(prompt(qt, cut(D[d], 230))) for d in ranked[:depth] if d in D}
    for r, d in enumerate(ranked):
        if d not in out[q]: out[q][d] = -r / len(ranked)
    print(k, f'{time.time() - t0:.0f}s', flush=True)
secs = time.time() - t0; tag = f'{which}_{rname}_top{depth}_n{n}'
store(tag, out, topics)
json.dump({'seconds': secs, 'topics': n, 'pairs': n * depth, 'model': REPO}, open(f'{REV}{tag}_time.json', 'w'))
print('DONE', secs)
