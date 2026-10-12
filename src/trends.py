"""Step 4: initial trend charts - how many fake / real articles per month.

Run: python src/trends.py
Input : data/features_news.csv
Output: results/trends_monthly.csv, results/figures/trend_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from settings import FEATURES_CSV, FIGURES_DIR, RESULTS_DIR

data = pd.read_csv(FEATURES_CSV)
data = data.dropna(subset=["year_month"])
print("Articles with a valid date:", len(data))

monthly = (
    data.groupby(["year_month", "label"]).size().unstack(fill_value=0).sort_index()
)
for col in ("FAKE", "REAL"):
    if col not in monthly:
        monthly[col] = 0
monthly["total"] = monthly["FAKE"] + monthly["REAL"]
monthly["fake_share"] = (monthly["FAKE"] / monthly["total"]).round(4)
monthly.to_csv(RESULTS_DIR / "trends_monthly.csv")

x = range(len(monthly))
ticks = list(range(0, len(monthly), max(1, len(monthly) // 12)))

# Chart 1: article volume per month
plt.figure(figsize=(11, 4.5))
plt.plot(x, monthly["FAKE"], label="FAKE", color="#d62728", marker="o", markersize=3)
plt.plot(x, monthly["REAL"], label="REAL", color="#1f77b4", marker="o", markersize=3)
plt.xticks(ticks, [monthly.index[i] for i in ticks], rotation=45)
plt.title("Fake vs Real articles per month")
plt.ylabel("Number of articles")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "trend_monthly_volume.png", dpi=150)
plt.close()

# Chart 2: share of fake articles per month
plt.figure(figsize=(11, 4.5))
plt.plot(x, monthly["fake_share"] * 100, color="#d62728", marker="o", markersize=3)
plt.xticks(ticks, [monthly.index[i] for i in ticks], rotation=45)
plt.title("Share of fake articles per month (%)")
plt.ylabel("% fake")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "trend_monthly_fake_share.png", dpi=150)
plt.close()

print(monthly.head(12))
print("\nTrend charts saved in", FIGURES_DIR)
