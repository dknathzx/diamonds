dir
# ============================================================
#  Diamonds Dataset - Data Cleaning & Analysis
#  MSc Big Data Analytics Project
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ── Setup ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs", exist_ok=True)

# ============================================================
# 1. EXTRACT — Load Data
# ============================================================
print("=" * 55)
print("  DIAMONDS DATA CLEANING & ANALYSIS PIPELINE")
print("=" * 55)

df = pd.read_csv("data/diamonds_sample.csv")
print(f"\n✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 2. EXPLORE — Understand the Raw Data
# ============================================================
print("\n📋 STEP 1 — Raw Data Overview")
print("-" * 40)
print(df.head())
print(f"\nColumns : {list(df.columns)}")
print(f"Dtypes  :\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic stats:\n{df.describe()}")

# ============================================================
# 3. CLEAN — Data Cleaning Steps
# ============================================================
print("\n🧹 STEP 2 — Data Cleaning")
print("-" * 40)

original_count = len(df)

# Remove duplicates
df = df.drop_duplicates()
print(f"✅ Removed duplicates        : {original_count - len(df)} rows dropped")

# Drop rows with nulls
df = df.dropna()
print(f"✅ Removed null rows         : now {len(df)} rows remain")

# Remove invalid prices and carats
df = df[df["price"] > 0]
df = df[df["carat"] > 0]
print(f"✅ Removed invalid prices    : now {len(df)} rows remain")

# Standardise cut column (strip spaces, title case)
df["cut"] = df["cut"].str.strip().str.title()
print(f"✅ Standardised 'cut' column")

# ============================================================
# 4. TRANSFORM — Feature Engineering
# ============================================================
print("\n⚙️  STEP 3 — Feature Engineering")
print("-" * 40)

# Price per carat
df["price_per_carat"] = (df["price"] / df["carat"]).round(2)
print("✅ Added column: price_per_carat")

# Value category
def categorise(ppc):
    if ppc < 2000:
        return "Budget"
    elif ppc < 5000:
        return "Mid-Range"
    else:
        return "Premium"

df["value_category"] = df["price_per_carat"].apply(categorise)
print("✅ Added column: value_category")

# Price band
df["price_band"] = pd.cut(
    df["price"],
    bins=[0, 500, 1000, 2000, 5000],
    labels=["<$500", "$500-$1k", "$1k-$2k", "$2k+"]
)
print("✅ Added column: price_band")

print(f"\nCleaned dataset shape: {df.shape}")
print(df[["carat", "cut", "color", "price", "price_per_carat", "value_category"]].head(8))

# ============================================================
# 5. ANALYSE — Key Insights
# ============================================================
print("\n📊 STEP 4 — Analysis Insights")
print("-" * 40)

print("\n🔹 Average price by cut:")
print(df.groupby("cut")["price"].mean().round(2).sort_values(ascending=False))

print("\n🔹 Average price per carat by color:")
print(df.groupby("color")["price_per_carat"].mean().round(2).sort_values(ascending=False))

print("\n🔹 Value category distribution:")
print(df["value_category"].value_counts())

print("\n🔹 Correlation (carat vs price):")
corr = df["carat"].corr(df["price"])
print(f"   Pearson r = {corr:.4f}  ({'Strong' if abs(corr) > 0.7 else 'Moderate'} positive correlation)")

# ============================================================
# 6. VISUALISE — 4 Charts in one figure
# ============================================================
print("\n🎨 STEP 5 — Generating Visualisations...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("Diamonds Dataset — Analysis Dashboard", fontsize=18, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Chart 1 — Carat vs Price scatter
ax1 = fig.add_subplot(gs[0, 0])
colors_map = {"Budget": "#4CAF50", "Mid-Range": "#FF9800", "Premium": "#F44336"}
for cat, grp in df.groupby("value_category"):
    ax1.scatter(grp["carat"], grp["price"], label=cat,
                color=colors_map[cat], alpha=0.7, edgecolors="white", linewidth=0.5, s=60)
ax1.set_title("Carat vs Price by Value Category", fontweight="bold")
ax1.set_xlabel("Carat")
ax1.set_ylabel("Price ($)")
ax1.legend(title="Category")

# Chart 2 — Avg price by cut (bar)
ax2 = fig.add_subplot(gs[0, 1])
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
cut_avg = df.groupby("cut")["price"].mean().reindex(
    [c for c in cut_order if c in df["cut"].unique()]
)
bars = ax2.bar(cut_avg.index, cut_avg.values,
               color=sns.color_palette("muted", len(cut_avg)), edgecolor="white")
ax2.set_title("Average Price by Cut Quality", fontweight="bold")
ax2.set_xlabel("Cut")
ax2.set_ylabel("Average Price ($)")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 20, f"${bar.get_height():,.0f}",
             ha="center", va="bottom", fontsize=9)

# Chart 3 — Value category pie
ax3 = fig.add_subplot(gs[1, 0])
vc = df["value_category"].value_counts()
ax3.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
        colors=[colors_map[c] for c in vc.index],
        startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
ax3.set_title("Value Category Distribution", fontweight="bold")

# Chart 4 — Price distribution histogram
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(df["price"], bins=15, color="#5C6BC0", edgecolor="white", alpha=0.85)
ax4.axvline(df["price"].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: ${df['price'].mean():,.0f}")
ax4.axvline(df["price"].median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median: ${df['price'].median():,.0f}")
ax4.set_title("Price Distribution", fontweight="bold")
ax4.set_xlabel("Price ($)")
ax4.set_ylabel("Frequency")
ax4.legend()

plt.savefig("outputs/diamonds_dashboard.png", dpi=150, bbox_inches="tight")
print("✅ Dashboard saved → outputs/diamonds_dashboard.png")

# ============================================================
# 7. EXPORT — Save cleaned data
# ============================================================
df.to_csv("outputs/diamonds_cleaned.csv", index=False)
print("✅ Cleaned data saved → outputs/diamonds_cleaned.csv")

print("\n" + "=" * 55)
print("  ✅ PIPELINE COMPLETE!")
print("=" * 55)