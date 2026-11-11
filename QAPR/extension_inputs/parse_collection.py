"""Parse SGML docs (collection or converted topics) into JSONL: id,title,abstract,description,claims."""
import os, re, sys, json
SRC, OUT = sys.argv[1], sys.argv[2]
def sec(x, t):
    m = re.search(rf'<{t}>\n?(.*?)\n?</{t}>', x, re.S)
    return ' '.join(m.group(1).split()) if m else ''
with open(OUT, 'w') as o:
    for f in sorted(os.listdir(SRC)):
        x = open(os.path.join(SRC, f), encoding='utf-8').read()
        o.write(json.dumps({'id': sec(x,'DOCNO'), 'title': sec(x,'TITLE'), 'abstract': sec(x,'ABSTRACT'),
                            'description': sec(x,'DESCRIPTION'), 'claims': sec(x,'CLAIMS')}) + '\n')
