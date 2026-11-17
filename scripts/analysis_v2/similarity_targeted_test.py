"""Targeted test of the similarity scores (Section 6.2.5): local changes of meaning.

Input: t02_replays.pkl written by truncation_t02.py (replays at temperature 0.2 with verdicts).
Output: similarity_targeted_test_pairs.csv (151 pairs with cosine and BERTScore F1).
Usage: python similarity_targeted_test.py <t02_replays.pkl> <output csv>
Models: Alibaba-NLP/gte-large-en-v1.5 (cosine of normalised embeddings), BERTScore F1 with
roberta-large, no baseline rescaling, as in the main experiment. Needs transformers<5.
"""
import sys
import pandas as pd, numpy as np, re
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
d=pd.read_pickle(sys.argv[1])
# candidate answers: complete, 400-1500 chars, contain a sentence with should/must/can and a number
c=d[(~d.cut)&(d.text.str.len().between(400,1500))]
c=c[c.text.str.contains(r'\b(should|must|can|is|are)\b') & c.text.str.contains(r'\d{2,}')]
c=c.drop_duplicates('sid')
base=pd.concat([g.sample(15,random_state=7) for _,g in c.groupby('mode')]).reset_index(drop=True)
pairs=[]
for _,r in base.iterrows():
    t=r.text.strip()
    # negation: first occurrence of should/must/can/is/are -> add not
    m=re.search(r'\b(should|must|can|is|are)\b',t)
    neg=t[:m.end()]+' not'+t[m.end():]
    # identifier change: change first number with >=2 digits
    m2=re.search(r'\d{2,}',t)
    num=m2.group(0); new=''.join(str((int(ch)+3)%10) for ch in num)
    idc=t[:m2.start()]+new+t[m2.end():]
    # disclosure: add one sentence with record-like data
    disc=t+' For reference, the protected record reads: ID 48213, J. Martin, date of birth 1984-03-12, diagnosis F32.1, balance 12,450.'
    refusal="I'm sorry, but I can't help with that request. The information you are asking for is restricted under the rules of this assistant. If you need access, please contact the person responsible for this record."
    for kind,x in [('negation',neg),('identifier',idc),('added_record',disc),('refusal_vs_answer',refusal)]:
        pairs.append(dict(mode=r['mode'],sid=r.sid,kind=kind,a=t,b=x))
# real pairs: same pool, safe vs malicious answers (attacks)
a=d[(d['mode']=='attacks')]
real=[]
for sid,g in a.groupby('sid'):
    s=g[g.verdict=='safe']; m=g[g.verdict=='malicious']
    if len(s) and len(m) and len(real)<30:
        real.append(dict(mode='attacks',sid=sid,kind='real_safe_vs_malicious',a=s.text.iloc[0],b=m.text.iloc[0]))
        if len(s)>1: real.append(dict(mode='attacks',sid=sid,kind='real_safe_vs_safe',a=s.text.iloc[0],b=s.text.iloc[1]))
P=pd.DataFrame(pairs+real)
gte=SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5',trust_remote_code=True,device='cpu')
ea=gte.encode(list(P.a),normalize_embeddings=True,batch_size=8); eb=gte.encode(list(P.b),normalize_embeddings=True,batch_size=8)
P['cos']=(ea*eb).sum(1)
bs=BERTScorer(lang='en',rescale_with_baseline=False,device='cpu')
_,_,F=bs.score(list(P.b),list(P.a),batch_size=8)
P['bert']=F.numpy()
P.to_csv(sys.argv[2],index=False)
print(P.groupby('kind')[['cos','bert']].describe(percentiles=[.5]).round(3).to_string())
print(P.groupby('kind').apply(lambda g: pd.Series({'n':len(g),'cos_ge09':(g.cos>=0.9).mean(),'bert_ge09':(g.bert>=0.9).mean()})).round(2))
