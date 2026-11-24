import os; os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'), exist_ok=True); os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'out'))
import json, numpy as np, pandas as pd
rng=np.random.default_rng(11)
df=pd.read_pickle('draws.pkl'); D=np.array(df.d.tolist()); n=len(D)
X=np.zeros((n,60),dtype=np.int16)
for i,r in enumerate(D): X[i,r-1]=1
def hot_hits(X,w,start=300,tie=None):
    cs=np.vstack([np.zeros((1,60),dtype=np.int32),np.cumsum(X,0)])
    hits=[]
    for t in range(start,len(X)):
        score=cs[t]-cs[t-w]+tie[t]
        s=np.argpartition(-score,6)[:6]; hits.append(X[t,s].sum())
    return np.array(hits)
tie=rng.random((n,60))*1e-3
res={}
for w in [10,20,30,50,75,100,150,200,400]:
    h=hot_hits(X,w,tie=tie); res[w]={'mean':round(float(h.mean()),4),'z':round(float((h.mean()-0.6)/(h.std(ddof=1)/np.sqrt(len(h)))),2)}
print('window sweep', res)
# permutation null for w=50: shuffle draw order
obs=hot_hits(X,50,tie=tie).mean(); null=[]
for k in range(300):
    Xp=X[rng.permutation(n)]; null.append(hot_hits(Xp,50,tie=tie).mean())
null=np.array(null); print('w50 obs',obs,'perm p (one-sided)',(null>=obs).mean(),'null mean',null.mean(), 'null 97.5',np.percentile(null,97.5))
# per era breakdown w=50
h=hot_hits(X,50,tie=tie); t=np.arange(300,n)
for a,b in [(300,1000),(1000,1700),(1700,2400),(2400,n)]:
    m=(t>=a)&(t<b); print('era',df.c[a],'-',df.c[b-1],'mean',round(h[m].mean(),4),'z',round((h[m].mean()-0.6)/(h[m].std()/np.sqrt(m.sum())),2))
json.dump({'sweep':res,'w50_perm_p':float((null>=obs).mean()),'w50_obs':float(obs)},open('d2_results.json','w'))
