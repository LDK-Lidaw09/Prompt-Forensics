import pandas as pd, numpy as np
d=pd.read_csv("/mnt/project-files/analyse_runs_v2/validation_evaluator_resultats_Lys.csv")
d["hu"]=d.human.astype(str).str.lower().isin(["suspicious","malicious"]); d["hm"]=d.human.astype(str).str.lower().eq("malicious")
print(d.human.value_counts())
S=d.groupby("stratum").agg(N=("stratum_population","first"),n=("hu","size"),human_unsafe=("hu","sum"),human_malicious=("hm","sum"),mode=("run_mode","first"),verdict=("verdict","first")).reset_index()
S["human_safe"]=S.n-S.human_unsafe
print(S.to_string())
rng=np.random.default_rng(1); B=20000; out=[]
for mode,g in S.groupby("mode"):
    U=g.verdict.isin(["suspicious","malicious"]).to_numpy(); N=g.N.to_numpy()
    for name,a0,b0 in [("jeffreys",0.5,0.5),("uniform",1,1)]:
        h=rng.beta(g.human_unsafe.to_numpy()+a0,(g.n-g.human_unsafe).to_numpy()+b0,size=(B,len(g)))
        num=(N*h*U).sum(1); prec=num/(N*U).sum(); rec=num/(N*h).sum(1); rate=(N*h).sum(1)/N.sum()
        q=lambda x: np.quantile(x,[.025,.5,.975]).round(3).tolist()
        out.append(dict(mode=mode,prior=name,precision=q(prec),recall=q(rec),human_rate=q(rate)))
    hp=g.human_unsafe/g.n; num=(N*hp*U).sum()
    print(mode,"point prec %.3f rec %.3f rate %.3f"%(num/(N*U).sum(), num/(N*hp).sum(), (N*hp).sum()/N.sum()))
print(pd.DataFrame(out).to_string())
S.to_csv("/home/user/repo/results/v2/tables/manual_check_strata.csv",index=False)
pd.DataFrame(out).to_csv("/home/user/repo/results/v2/tables/manual_check_bayes.csv",index=False)
