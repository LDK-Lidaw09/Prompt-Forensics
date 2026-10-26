#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math
from collections import defaultdict

FAM_ORDER = ["belief_elicitation","data_leakage","deceptive_delight","jailbreak","prompt_injection","reverse_psychology"]
FAM_SHORT = {"belief_elicitation":"BE","data_leakage":"DL","deceptive_delight":"DD","jailbreak":"JB","prompt_injection":"PI","reverse_psychology":"RP"}
THRESH = 0.9

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def mean_std(vals):
    vals=[v for v in vals if v is not None]
    if not vals: return (float("nan"),float("nan"))
    m=sum(vals)/len(vals)
    sd=math.sqrt(sum((v-m)**2 for v in vals)/len(vals)) if len(vals)>1 else 0.0
    return (m,sd)

def frac_above(rows_sub,col,thr=THRESH):
    vals=[fnum(r.get(col)) for r in rows_sub]; vals=[v for v in vals if v is not None]
    if not vals: return float("nan")
    return sum(1 for v in vals if v>=thr)/len(vals)

def agg(rows_sub):
    return {
        "n":len(rows_sub),
        "cons_cos":mean_std([fnum(r.get("cons_score")) for r in rows_sub]),
        "fid_cos":mean_std([fnum(r.get("fid_score")) for r in rows_sub]),
        "cons_cos_a":frac_above(rows_sub,"cons_score"),"fid_cos_a":frac_above(rows_sub,"fid_score"),
        "cons_bert":mean_std([fnum(r.get("cons_bert_score")) for r in rows_sub]),
        "fid_bert":mean_std([fnum(r.get("fid_bert_score")) for r in rows_sub]),
        "cons_bert_a":frac_above(rows_sub,"cons_bert_score"),"fid_bert_a":frac_above(rows_sub,"fid_bert_score"),
    }

def build_rows(rows):
    by_fam=defaultdict(list)
    for r in rows: by_fam[r.get("attack_type","")].append(r)
    out=[]
    for fam in FAM_ORDER:
        if fam in by_fam: out.append((FAM_SHORT[fam],agg(by_fam[fam])))
    out.append(("Micro",agg(rows)))
    return out

def print_console(title,table_rows):
    print("\n"+"="*100); print(f"  {title}"); print("="*100)
    print(f"  {'Att':<6}{'Cons_cos':>14}{'Fid_cos':>14}{'Cc>=.9':>9}{'Fc>=.9':>9}{'Cons_BERT':>14}{'Fid_BERT':>14}{'Cb>=.9':>9}{'Fb>=.9':>9}")
    print("  "+"-"*96)
    for label,a in table_rows:
        print(f"  {label:<6}{a['cons_cos'][0]:.3f}±{a['cons_cos'][1]:.3f} {a['fid_cos'][0]:.3f}±{a['fid_cos'][1]:.3f} {a['cons_cos_a']:>8.2f}{a['fid_cos_a']:>9.2f}  {a['cons_bert'][0]:.3f}±{a['cons_bert'][1]:.3f} {a['fid_bert'][0]:.3f}±{a['fid_bert'][1]:.3f} {a['cons_bert_a']:>8.2f}{a['fid_bert_a']:>9.2f}")

def write_csv(path,table_rows):
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["family","n","cons_cos_mean","cons_cos_std","fid_cos_mean","fid_cos_std","cons_cos_frac_ge_0.9","fid_cos_frac_ge_0.9","cons_bert_mean","cons_bert_std","fid_bert_mean","fid_bert_std","cons_bert_frac_ge_0.9","fid_bert_frac_ge_0.9"])
        for label,a in table_rows:
            w.writerow([label,a["n"],f"{a['cons_cos'][0]:.4f}",f"{a['cons_cos'][1]:.4f}",f"{a['fid_cos'][0]:.4f}",f"{a['fid_cos'][1]:.4f}",f"{a['cons_cos_a']:.4f}",f"{a['fid_cos_a']:.4f}",f"{a['cons_bert'][0]:.4f}",f"{a['cons_bert'][1]:.4f}",f"{a['fid_bert'][0]:.4f}",f"{a['fid_bert'][1]:.4f}",f"{a['cons_bert_a']:.4f}",f"{a['fid_bert_a']:.4f}"])

def latex_table(title,label,table_rows):
    L=[]
    L.append("\\begin{table}[t]"); L.append("\\centering")
    L.append("\\caption{"+title+"}"); L.append("\\label{"+label+"}"); L.append("\\small")
    L.append("\\begin{tabular}{lcccccccc}"); L.append("\\toprule")
    L.append(" & $\\mathrm{Cons}_{\\cos}$ & $\\mathrm{Fid}_{\\cos}$ & $\\mathrm{Cons}_{\\cos}\\!\\geq\\!0.9$ & $\\mathrm{Fid}_{\\cos}\\!\\geq\\!0.9$ & $\\mathrm{Cons}_{\\mathrm{BERT}}$ & $\\mathrm{Fid}_{\\mathrm{BERT}}$ & $\\mathrm{Cons}_{\\mathrm{BERT}}\\!\\geq\\!0.9$ & $\\mathrm{Fid}_{\\mathrm{BERT}}\\!\\geq\\!0.9$ \\\\")
    L.append("\\midrule")
    for lr,a in table_rows:
        if lr=="Micro": L.append("\\midrule")
        L.append(f"{lr} & {a['cons_cos'][0]:.3f}$\\pm${a['cons_cos'][1]:.3f} & {a['fid_cos'][0]:.3f}$\\pm${a['fid_cos'][1]:.3f} & {a['cons_cos_a']:.2f} & {a['fid_cos_a']:.2f} & {a['cons_bert'][0]:.3f}$\\pm${a['cons_bert'][1]:.3f} & {a['fid_bert'][0]:.3f}$\\pm${a['fid_bert'][1]:.3f} & {a['cons_bert_a']:.2f} & {a['fid_bert_a']:.2f} \\\\")
    L.append("\\bottomrule"); L.append("\\end{tabular}"); L.append("\\end{table}")
    return "\n".join(L)

def load(path):
    return list(csv.DictReader(open(path,encoding="utf-8",errors="replace")))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--attacks",required=True)
    ap.add_argument("--safe",required=True)
    ap.add_argument("--out-prefix",default="replay_robustness")
    args=ap.parse_args()
    atk=load(args.attacks); saf=load(args.safe)
    print(f"[INFO] attacks: {len(atk)} | safe: {len(saf)}")
    t1=build_rows(atk); t2=build_rows(saf)
    title1="Replay robustness on attack prompts (mean$\\pm$std; fraction $\\geq 0.9$)."
    title2="Replay robustness on safe prompts (10 replays per scenario). Same metrics as Table~1."
    print_console("TABLE 1 — Replay robustness on ATTACK prompts",t1)
    print_console("TABLE 2 — Replay robustness on SAFE prompts",t2)
    write_csv(f"{args.out_prefix}_table1_attacks.csv",t1)
    write_csv(f"{args.out_prefix}_table2_safe.csv",t2)
    tex=latex_table(title1,"tab:replay_attacks",t1)+"\n\n"+latex_table(title2,"tab:replay_safe",t2)+"\n"
    open(f"{args.out_prefix}_tables.tex","w",encoding="utf-8").write(tex)
    print("\n  Fichiers écrits :",f"{args.out_prefix}_table1_attacks.csv, _table2_safe.csv, _tables.tex\n")

if __name__=="__main__":
    main()
