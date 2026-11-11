"""Neural reranker baselines (reviewer R2.1(2)), query = topic title+abstract, doc = title+abstract.
  ce  : cross-encoder/ms-marco-MiniLM-L-6-v2 over the top-<depth> of a run, all 923 test topics
  llm : Qwen/Qwen3-Reranker-0.6B (LLM-based pointwise reranker, P(yes)) over the top-<depth>, on a fixed random
        sample of <n> test topics (CPU-only container); candidates below <depth> keep their first-stage order.
usage: rev_neural.py ce|llm <run_name_in_runs/> <depth> [n_topics]"""
import sys, json, time, random, numpy as np, torch
from rev_common import *
kind, rname, depth = sys.argv[1], sys.argv[2], int(sys.argv[3]); n = int(sys.argv[4]) if len(sys.argv) > 4 else None
topics = TEST if n is None else sorted(random.Random(0).sample(TEST, n))
run = read_run(RUNS + rname + '.txt', set(topics))
need = {d for q in run for d, _ in sorted(run[q].items(), key=lambda kv: -kv[1])[:depth]}
T = {}
for l in open('/home/user/data/topics.jsonl'):
    t = json.loads(l)
    if t['id'] in run: T[t['id']] = t['title'] + '. ' + t['abstract']
Dt = {}
for l in open('/home/user/data/docs.jsonl'):
    i = l[8:l.index('"', 8)]
    if i in need: t = json.loads(l); Dt[i] = t['title'] + '. ' + t['abstract']
print('topics', len(run), 'docs', len(Dt), flush=True)
torch.set_num_threads(4)
if kind == 'ce':
    from sentence_transformers import CrossEncoder
    m = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    score = lambda q, ds: m.predict([(q, d) for d in ds], batch_size=32)
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    name = 'Qwen/Qwen3-Reranker-0.6B'; tok = AutoTokenizer.from_pretrained(name, padding_side='left')
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    yes, no = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
    pre = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
           'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suf = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    ins = 'Given a patent application, judge whether the document is relevant prior art that a patent examiner would cite against it'
    cut = lambda s, k: tok.decode(tok(s, add_special_tokens=False)['input_ids'][:k])
    def score(q, ds):
        out = []
        for b in range(0, len(ds), 8):
            x = tok([pre + f'<Instruct>: {ins}\n<Query>: {cut(q, 230)}\n<Document>: {cut(d, 230)}' + suf for d in ds[b:b + 8]],
                    return_tensors='pt', padding=True)
            with torch.no_grad(): lg = m(**x).logits[:, -1, [no, yes]]
            out += torch.softmax(lg, -1)[:, 1].tolist()
        return out
t0 = time.time(); out = {}
for i, q in enumerate(sorted(run)):
    ranked = [d for d, _ in sorted(run[q].items(), key=lambda kv: -kv[1])]
    top = [d for d in ranked[:depth] if d in Dt]; s = score(T[q], [Dt[d] for d in top])
    out[q] = {d: 10 + float(v) for d, v in zip(top, s)}
    for r, d in enumerate(ranked):
        if d not in out[q]: out[q][d] = -r / len(ranked)
    if i % 20 == 0: print(i, f'{time.time() - t0:.0f}s', flush=True)
secs = time.time() - t0
tag = f'{kind}_{rname}_top{depth}' + (f'_n{n}' if n else '')
store(tag, out, topics)
store(f'{rname}_sample_n{n}' if n else rname + '_all', run, topics, save_run=False)
json.dump({'seconds': secs, 'topics': len(run), 'pairs': sum(min(depth, len(v)) for v in run.values())}, open(f'{REV}{tag}_time.json', 'w'))
print('DONE', secs)
