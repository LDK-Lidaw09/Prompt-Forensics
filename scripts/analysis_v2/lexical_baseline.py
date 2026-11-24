"""Lexical baseline on the prompt text alone (no system prompt, document or answer).

A TF-IDF representation with a logistic regression is fitted on the development split
(29 documents) and applied unchanged to the test split (169 documents), as for the
cutoff of Section 5.2.6. It measures how far the two classes of the benchmark can be
told apart from surface features of the prompt, which bounds what the recall and
false-positive rate of an Investigator say about the analysis of an interaction.
A second analysis leaves the attack prompts of one source collection out of the fitting
and tests on that source, to see whether the separation rests on templates seen in fitting.

Output: results/v2/tables/lexical_baseline.csv
Usage: python3 scripts/analysis_v2/lexical_baseline.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/v2/tables/lexical_baseline.csv"
SEED = 2026


def load():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/scenarios_v2/*/*.json"))):
        for s in json.load(open(f)):
            if s.get("attack_prompt"):
                rows.append(dict(text=s["attack_prompt"], y=1, split=s["split"], kind="attack",
                                 source=s["benchmark_source"]))
            if s.get("safe_prompt"):
                rows.append(dict(text=s["safe_prompt"], y=0, split=s["split"], kind=s["safe_kind"],
                                 source=""))
    return pd.DataFrame(rows)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def evaluate(name, vec, d):
    dev, test = d[d.split == "dev"], d[d.split == "test"].copy()
    X = vec.fit_transform(dev.text)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED).fit(X, dev.y)
    test["p"] = clf.predict_proba(vec.transform(test.text))[:, 1]
    test["dec"] = (test.p >= 0.5).astype(int)
    dev_texts = set(dev.text)
    out = []

    def row(group, sub, positive):
        k = int((sub.dec == 1).sum())
        r, lo, hi = wilson(k, len(sub))
        out.append(dict(features=name, group=group, n=len(sub),
                        measure="recall" if positive else "false_positive_rate",
                        flagged=k, rate=round(r, 4), lo=round(lo, 4), hi=round(hi, 4)))

    att, leg = test[test.y == 1], test[test.y == 0]
    row("attack, test", att, True)
    row("attack, test, text not in dev", att[~att.text.isin(dev_texts)], True)
    for src, g in att.groupby("source"):
        row(f"attack, test, {src}", g, True)
    row("legitimate, test", leg, False)
    for kind, g in leg.groupby("kind"):
        row(f"legitimate, test, {kind}", g, False)
    auc = roc_auc_score(test.y, test.p)
    out.append(dict(features=name, group="all, test", n=len(test), measure="auc",
                    flagged=None, rate=round(auc, 4), lo=None, hi=None))
    return out


def cross_source(d):
    """Word model fitted without the attack prompts of one source, tested on that source."""
    out = []
    dev, test = d[d.split == "dev"], d[d.split == "test"]
    for held in sorted(d[d.y == 1].source.unique()):
        tr = dev[(dev.y == 0) | (dev.source != held)]
        te = test[(test.y == 0) | (test.source == held)]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED).fit(vec.fit_transform(tr.text), tr.y)
        dec = clf.predict_proba(vec.transform(te.text))[:, 1] >= 0.5
        a = (te.y == 1).values
        for group, mask, measure in ((f"attack, test, {held} (source left out)", a, "recall"),
                                     (f"legitimate, test ({held} left out)", ~a, "false_positive_rate")):
            k = int(dec[mask].sum())
            r, lo, hi = wilson(k, int(mask.sum()))
            out.append(dict(features="word 1-2 grams, source left out", group=group, n=int(mask.sum()),
                            measure=measure, flagged=k, rate=round(r, 4), lo=round(lo, 4), hi=round(hi, 4)))
    return out


def main():
    d = load()
    assert (d.y == 1).sum() == 21600 and (d.y == 0).sum() == 19981, d.y.value_counts()
    res = []
    res += evaluate("word 1-2 grams", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), d)
    res += evaluate("character 3-5 grams", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                                           sublinear_tf=True), d)
    res += cross_source(d)
    t = pd.DataFrame(res)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
