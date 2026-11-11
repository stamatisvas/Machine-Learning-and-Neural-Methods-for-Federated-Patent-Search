# Inputs for the neural baseline runs (QAPR extension)

- `bm25_top100_all_topics.txt.gz`: TREC run, BM25 (Anserini k1=0.9, b=0.4, full topic text as query) top-100 for all 2,592 WPI-PR topics over WPI-60K.
- `Prior_art_qrels.txt`: relevance judgements. Test topics = the 923 with >= 3 relevant documents; training topics = the 1,669 with exactly 2.
- `convert_topics.py`, `parse_collection.py`: how the topic XMLs and the WPI-60K SGML were converted to `topics.jsonl` / `docs.jsonl` (paths inside are from the original environment and need adjusting).
- `rev_gguf.py`, `rev_neural.py`: the earlier zero-shot LLM and cross-encoder scripts (prompt, scoring by logit(Yes) - logit(No)), for reference.
- `evaluate.py`: pytrec_eval wrapper used for all reported metrics.

Data sources (Google Drive): WPI-60K SGML zip, file id `1DhKnQdwugwJorjokZjzlU8B2A4XsiF-Y`; topic XMLs, folder id `1WxEbo5WDWswTPqWlyOpjxd7mOhbq14Ig`.
