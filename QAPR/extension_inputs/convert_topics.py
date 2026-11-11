"""Convert WPI topic XML files to the same SGML layout/cleaning as the WPI-60K collection
(WPI-PR/2_convert_to_sgml.py clean_text; EN sections only)."""
import os, re, sys
from lxml import etree
SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
CH = ["\n","\t","'","-",".","!","@","#","$","%","&","*","=","+","0","1","2","3","4","5","6","7","8","9",
      "(",")","{","}","[","]",",",'"',"?","<",">",";","¬","°",":"]
def clean(t):
    t = t.lower()
    for c in CH: t = t.replace(c, ' ')
    return re.sub(r' +', ' ', t).strip()
def en(root, tag):
    for e in root.iter(tag):
        if e.get('lang') == 'EN':
            return clean(' '.join(e.itertext()))
    return ''
n = 0; missing = 0
for f in sorted(os.listdir(SRC)):
    root = etree.parse(os.path.join(SRC, f)).getroot()
    did = root.get('ucid')
    secs = {k: en(root, t) for k, t in [('TITLE','invention-title'),('ABSTRACT','abstract'),('DESCRIPTION','description'),('CLAIMS','claims')]}
    if not all(secs.values()): missing += 1
    with open(os.path.join(DST, did + '.txt'), 'w') as o:
        o.write(f"<DOC>\n<DOCNO>\n{did}\n</DOCNO>\n<TEXT>\n<DATE>\n{root.get('date','')}\n</DATE>\n")
        for k, v in secs.items(): o.write(f"<{k}>\n{v}\n</{k}>\n")
        o.write("</TEXT>\n</DOC>\n")
    n += 1
print(n, 'topics;', missing, 'with an empty EN section')
