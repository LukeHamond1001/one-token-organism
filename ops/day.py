import json, sys, re
LOG='/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl'
target=int(sys.argv[1])
rows=[]
with open(LOG) as f:
    for L in f:
        L=L.strip()
        if not L: continue
        try: r=json.loads(L)
        except: continue
        if r.get('day')==target: rows.append(r)
lines=[r for r in rows if r.get('action')=='line']
cues=[r for r in rows if r.get('action')=='cue']
a=[r for r in lines if r.get('voice','a')=='a']
b=[r for r in lines if r.get('voice')=='b']
aq=[r for r in a if r.get('text','').rstrip().endswith('?')]
sm=sum(1 for r in rows if r.get('action')=='smile')
fr=sum(1 for r in rows if r.get('action')=='frown')
wh=sum(1 for r in rows if r.get('action')=='withheld')
print(f"day {target}: lines={len(lines)} (A={len(a)} B={len(b)}) A-questions={len(aq)} cues={len(cues)} smiles={sm} frowns={fr} withheld={wh}")
# A question answered by the child before B spoke: its_after of the A-question row
ans=0; ansq=[]
for i,r in enumerate(lines):
    if r.get('voice','a')!='a': continue
    t=r.get('text','')
    if not t.rstrip().endswith('?'): continue
    af=(r.get('its_after') or '').replace('_',' ')
    low=af.lower()
    if re.search(r'(^|[^a-z])(i |yes|no |please)', low):
        ans+=1; ansq.append((t,af.strip()))
print(f"A-questions the child answered with I/yes/no/please before B: {ans}")
for t,af in ansq[:60]: print("   Q:",t,"| its:",repr(af))
# its own questions
own=[]
for r in rows:
    af=r.get('its_after') or ''
    if '?' in af: own.append((r.get('ts','')[11:], r.get('text',r.get('frag','')), af))
print(f"child question marks on the page: {len(own)}")
for ts,t,af in own: print("   ",ts,repr(af),"| after:",t)
# cue answers
print("cue answers:")
for r in cues: print("   ", r.get('ts','')[11:], repr(r.get('text','')), "->", repr(r.get('its_after','')))
# repetition of its strings
from collections import Counter
c=Counter()
for r in rows:
    af=(r.get('its_after') or '').replace('_',' ')
    for m in re.finditer(r'i see milk|i see m|i see', af.lower()): c[m.group(0)]+=1
print("i-see occurrences:", dict(c))
