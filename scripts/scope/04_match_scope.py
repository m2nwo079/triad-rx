"""Estimate dictionary matching coverage on abstracts (C3 part 2). Exploratory scope script."""
import pandas as pd, json, re, collections as C, numpy as np
cnt=json.load(open('data/scope/pwc_term_counts.json'))
m=pd.read_parquet('data/scope/methods.parquet')
spam_pat=re.compile(r'[☎→【】]|\+\d|\d{3}\D{0,3}\d{3}\D{0,3}\d{4}|customer|airlines|phone|call ',re.I)
def norm(s): return re.sub(r'[\s\-_]+',' ',s.strip().lower())
# Case-insensitive surface form -> canonical term
ci={}
# Case-sensitive acronym -> canonical term
cs={}
for t,v in cnt['tasks'].items():
    if v>=50 and not spam_pat.search(t) and len(t)>=4: ci.setdefault(norm(t),'T:'+norm(t))
mm=m[(m.num_papers>=10)&~m.name.str.contains(spam_pat)&~m.full_name.fillna('').str.contains(spam_pat)]
for _,r in mm.iterrows():
    canon='M:'+r['name']
    fn=r['full_name'] if isinstance(r['full_name'],str) else ''
    if len(fn)>=6 and ' ' in fn: ci.setdefault(norm(fn),canon)
    n=r['name']
    # Acronyms are matched case-sensitively to avoid collisions with common words
    if n.isupper() and len(n)>=3:
        cs.setdefault(n,canon)
    elif ' ' in n and len(n)>=6: ci.setdefault(norm(n),canon)
cols=['title','abstract','date']
d=pd.concat([pd.read_parquet(f'data/scope/pwa_{i}.parquet',columns=cols) for i in range(4)],ignore_index=True)
d['year']=pd.to_datetime(d.date,errors='coerce').dt.year
pat_ci=re.compile(r'(?<![a-z0-9])('+'|'.join(re.escape(k) for k in sorted(ci,key=len,reverse=True))+r')(?![a-z0-9])')
pat_cs=re.compile(r'(?<![A-Za-z0-9])('+'|'.join(re.escape(k) for k in sorted(cs,key=len,reverse=True))+r')(?![A-Za-z0-9])')
out={}
for y in [2016,2020,2024]:
    s=d[d.year==y].sample(10000,random_state=0)
    sets=[]
    for t,a in zip(s.title,s.abstract):
        raw=(t if isinstance(t,str) else '')+' . '+(a if isinstance(a,str) else '')
        f={ci[x] for x in pat_ci.findall(norm(raw))}|{cs[x] for x in pat_cs.findall(raw)}
        sets.append(f)
    df=C.Counter(x for f in sets for x in f)
    generic={k for k,v in df.items() if v/len(sets)>0.02}
    n=np.array([len(f-generic) for f in sets])
    out[y]={'n':len(sets),'generic_removed':sorted(generic),'mean':float(n.mean()),'ge3':float((n>=3).mean()),'zero':float((n==0).mean()),'vocab_used':sum(1 for k in df if k not in generic)}
    print(y,out[y]['mean'],out[y]['ge3'],out[y]['zero'],len(generic),out[y]['vocab_used'])
    print('  generic:',sorted(generic)[:40])
    print('  top kept:',[k for k,v in df.most_common(60) if k not in generic][:25])
print('vocab ci',len(ci),'cs',len(cs))
json.dump(out,open('data/scope/match_by_year.json','w'))
