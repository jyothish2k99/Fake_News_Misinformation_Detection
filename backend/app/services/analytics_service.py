from functools import lru_cache
import json
from pathlib import Path
import pandas as pd
import re
from scipy.stats import mannwhitneyu
from textstat import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "analytics_cache.json"
_SENTIMENT = SentimentIntensityAnalyzer()

def load_labeled_data():
    frames=[]
    for filename, label in [("Fake.csv", 1), ("True.csv", 0)]:
        p=DATA_DIR / filename
        if p.exists():
            try:
                df=pd.read_csv(p, on_bad_lines="skip")
                if not {"title", "text"}.issubset(df.columns):
                    df=pd.read_csv(p, header=None, on_bad_lines="skip")
                    if len(df.columns)>=2:
                        cols=["title","text","subject","date"][:len(df.columns)]
                        df.columns=cols
                df["label"]=label
                frames.append(df)
            except Exception:
                continue
    if not frames:
        corpus_path = Path(__file__).resolve().parents[2] / "artifacts" / "evidence_corpus.csv"
        if corpus_path.exists():
            try:
                df = pd.read_csv(corpus_path)
                for c in ["title","text","subject","date"]:
                    if c not in df: df[c] = ""
                    df[c] = df[c].fillna("").astype(str)
                if "label" not in df:
                    return pd.DataFrame()
                df["label"] = pd.to_numeric(df["label"], errors="coerce")
                return df[df["label"].isin([0, 1])].copy()
            except Exception:
                pass
        return pd.DataFrame()
    out=pd.concat(frames, ignore_index=True)
    for c in ["title","text","subject","date"]:
        if c not in out: out[c]=""
        out[c]=out[c].fillna("").astype(str)
    return out


def _topic_category(subject):
    value = str(subject).strip().lower()
    if any(term in value for term in ("health", "medical", "covid", "disease")):
        return "Health"
    if any(term in value for term in ("science", "technology", "tech", "climate", "environment")):
        return "Science & technology"
    if any(term in value for term in ("politic", "election", "government", "worldnews", "left", "right")):
        return "Politics & government"
    return "Other / uncategorized"


def _emotional_language_test(df):
    real = df.loc[df["label_name"] == "Real", "emotional_language_score"].dropna()
    fake = df.loc[df["label_name"] == "Fake", "emotional_language_score"].dropna()
    result = {
        "metric": "Absolute VADER compound sentiment score (0-1); a lexical emotional-language proxy, not a truth measure",
        "method": "Two-sided Mann–Whitney U test for independent REAL- and FAKE-labeled articles",
        "real_sample_size": int(len(real)),
        "fake_sample_size": int(len(fake)),
        "real_median": round(float(real.median()), 4) if len(real) else None,
        "fake_median": round(float(fake.median()), 4) if len(fake) else None,
        "p_value": None,
        "rank_biserial_effect_size": None,
        "effect_magnitude": "unavailable",
        "interpretation": "Insufficient labeled articles to compare emotional-language scores.",
        "assumptions": "Independent observations; ties are handled by the Mann–Whitney implementation. The result is descriptive of this labeled corpus and does not imply causation.",
    }
    if not len(real) or not len(fake):
        return result
    statistic = mannwhitneyu(real, fake, alternative="two-sided", method="auto")
    effect = 2 * float(statistic.statistic) / (len(real) * len(fake)) - 1
    magnitude = abs(effect)
    effect_label = "negligible" if magnitude < 0.147 else "small" if magnitude < 0.33 else "medium" if magnitude < 0.474 else "large"
    p_value = float(statistic.pvalue)
    comparison = "higher" if effect > 0 else "lower" if effect < 0 else "similar"
    significance = "statistically distinguishable" if p_value < 0.05 else "not statistically distinguishable"
    result.update({
        "p_value": p_value,
        "rank_biserial_effect_size": round(effect, 4),
        "effect_magnitude": effect_label,
        "interpretation": f"REAL-labeled articles have {comparison} emotional-language proxy scores than FAKE-labeled articles; the distributions are {significance} at alpha=0.05. Effect magnitude: {effect_label}.",
    })
    return result


@lru_cache(maxsize=1)
def summary():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("available"):
                return cached
        except Exception:
            pass
    df=load_labeled_data()
    if df.empty:
        return {"available":False,"message":"Add Fake.csv and True.csv to backend/data to enable analytics.","class_distribution":[],"subjects":[],"topics":[],"monthly_trends":[],"linguistic_comparison":[],"emotional_language_test":None}
    df["label"]=pd.to_numeric(df["label"], errors="coerce")
    df=df[df["label"].isin([0,1])].copy()
    df["label_name"]=df.label.map({1:"Fake",0:"Real"})
    df["text_key"]=(df["title"].str.cat(df["text"],sep=" ").str.lower().str.replace(r"\s+"," ",regex=True).str.strip())
    df=df[df["text_key"].str.len()>0]
    conflicting=df.groupby("text_key")["label_name"].nunique()
    conflict_keys=set(conflicting[conflicting>1].index)
    conflict_count=int(df["text_key"].isin(conflict_keys).sum())
    before_dedup=len(df)
    df=df[~df["text_key"].isin(conflict_keys)].drop_duplicates("text_key").copy()
    duplicate_count=before_dedup-conflict_count-len(df)
    if df.empty:
        return {"available":False,"message":"No complete, unambiguous labeled articles are available for analytics.","class_distribution":[],"subjects":[],"topics":[],"monthly_trends":[],"linguistic_comparison":[],"emotional_language_test":None}
    dist=df.label_name.value_counts().reindex(["Real","Fake"],fill_value=0)
    subject=df.groupby(["subject","label_name"]).size().unstack(fill_value=0)
    subject["total"]=subject.sum(axis=1)
    subject=subject.sort_values("total",ascending=False).head(10).reset_index()
    subjects=[]
    for _,r in subject.iterrows():
        subjects.append({"subject":str(r["subject"]),"fake":int(r.get("Fake",0)),"real":int(r.get("Real",0))})
    df["topic_category"]=df["subject"].map(_topic_category)
    topic_table=df.groupby(["topic_category","label_name"]).size().unstack(fill_value=0)
    topics=[]
    for category in ["Politics & government","Health","Science & technology","Other / uncategorized"]:
        if category not in topic_table.index:
            continue
        row=topic_table.loc[category]
        topic_total=int(row.sum())
        topics.append({"category":category,"fake":int(row.get("Fake",0)),"real":int(row.get("Real",0)),"total":topic_total,"fake_share":round(float(row.get("Fake",0))/topic_total,4) if topic_total else None})
    df["parsed_date"]=pd.to_datetime(df.date, errors="coerce", utc=True)
    monthly=[]
    valid=df.dropna(subset=["parsed_date"]).copy()
    if len(valid):
        valid["month"]=valid.parsed_date.dt.strftime("%Y-%m")
        pivot=valid.groupby(["month","label_name"]).size().unstack(fill_value=0).sort_index().tail(36)
        monthly=[{"month":idx,"fake":int(row.get("Fake",0)),"real":int(row.get("Real",0))} for idx,row in pivot.iterrows()]
    def words(s): return len(re.findall(r"\b\w+\b",s))
    def exclaims(s): return s.count("!")
    def caps(s):
        letters=[x for x in s if x.isalpha()]
        return (sum(x.isupper() for x in letters)/len(letters)*100) if letters else 0
    df["word_count"]=df["title"].str.cat(df["text"],sep=" ").map(words)
    df["exclamation_count"]=df["title"].str.cat(df["text"],sep=" ").map(exclaims)
    df["capital_ratio"]=df["title"].str.cat(df["text"],sep=" ").map(caps)
    combined=df["title"].str.cat(df["text"],sep=" ").str.strip()
    df["readability_score"]=combined.map(lambda value: float(textstat.flesch_reading_ease(value)) if value else float("nan"))
    df["emotional_language_score"]=combined.map(lambda value: abs(float(_SENTIMENT.polarity_scores(value)["compound"])) if value else float("nan"))
    ling=[]
    for label, grp in df.groupby("label_name"):
        ling.append({"label":label,"avg_words":round(float(grp.word_count.mean()),1),"avg_exclamations":round(float(grp.exclamation_count.mean()),2),"avg_capital_ratio":round(float(grp.capital_ratio.mean()),2),"avg_flesch_reading_ease":round(float(grp.readability_score.mean()),2) if grp.readability_score.notna().any() else None})
    return {"available":True,"total_articles":int(len(df)),"duplicate_rows_removed":duplicate_count,"conflicting_duplicate_rows_excluded":conflict_count,"class_distribution":[{"label":k,"count":int(v)} for k,v in dist.items()],"subjects":subjects,"topics":topics,"monthly_trends":monthly,"linguistic_comparison":ling,"emotional_language_test":_emotional_language_test(df)}
