"""Lab 3 solution: Exploratory Data Analysis (EDA) for Netflix titles dataset.

This script mirrors the EDA flow shown in class and applies it to:
    Lab3/netflix_titles.csv

Run:
    python3 Lab3/lab3_solution.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Styling for cleaner plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


DATA_PATH = Path(__file__).with_name("netflix_titles.csv")
OUTPUT_DIR = Path(__file__).with_name("plots")
OUTPUT_DIR.mkdir(exist_ok=True)


# 1) Load dataset
df = pd.read_csv(DATA_PATH)
print("\n=== Dataset Preview ===")
print(df.head())


# 2) Basic structure checks
print("\n=== Shape (rows, columns) ===")
print(df.shape)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n=== Column Data Types ===")
print(df.dtypes)


# 3) Missing values and duplicates
print("\n=== Missing Values (count) ===")
print(df.isna().sum().sort_values(ascending=False))

print("\n=== Duplicate Rows ===")
dup_count = df.duplicated().sum()
print(f"Duplicate rows: {dup_count}")


# 4) Light preprocessing for EDA
# Convert date_added for time-based analysis.
df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

# Extract duration number for numeric summaries and relations.
duration_num = df["duration"].str.extract(r"(\d+)")
df["duration_num"] = pd.to_numeric(duration_num[0], errors="coerce")

print("\n=== Descriptive Statistics ===")
print(df.describe(include="all"))


# 5) Univariate analysis
# Missing value heatmap
plt.figure(figsize=(12, 5))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_missing_values_heatmap.png")
plt.close()

# Content type distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="type", palette="Set2")
plt.title("Count of Titles by Type")
plt.xlabel("Type")
plt.ylabel("Count")
for container in ax.containers:
    ax.bar_label(container)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_type_distribution.png")
plt.close()

# Release year distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["release_year"], bins=30, kde=True)
plt.title("Distribution of Release Year")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_release_year_distribution.png")
plt.close()

# Rating distribution (top ratings)
rating_counts = df["rating"].value_counts().head(10)
plt.figure(figsize=(10, 5))
rating_counts.plot(kind="bar", color="teal")
plt.title("Top 10 Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_top_ratings.png")
plt.close()


# 6) Bivariate analysis
# Titles by country (top 10)
country_counts = (
    df["country"].dropna().str.split(", ").explode().value_counts().head(10)
)
plt.figure(figsize=(10, 5))
country_counts.plot(kind="bar", color="cornflowerblue")
plt.title("Top 10 Countries by Number of Titles")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_top_countries.png")
plt.close()

# Top genres/categories from listed_in
genre_counts = (
    df["listed_in"].dropna().str.split(", ").explode().value_counts().head(10)
)
plt.figure(figsize=(10, 5))
genre_counts.plot(kind="bar", color="darkorange")
plt.title("Top 10 Genres/Categories")
plt.xlabel("Genre")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_top_genres.png")
plt.close()

# Content type over years (release year)
content_by_year = df.groupby(["release_year", "type"]).size().reset_index(name="count")
plt.figure(figsize=(12, 6))
sns.lineplot(data=content_by_year, x="release_year", y="count", hue="type")
plt.title("Movies vs TV Shows by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_type_by_release_year.png")
plt.close()


# 7) Correlation analysis (numeric columns)
numeric_cols = [col for col in ["release_year", "duration_num"] if col in df.columns]
if len(numeric_cols) >= 2:
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Matrix (Numeric Features)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_correlation_matrix.png")
    plt.close()


# 8) Time-based analysis
# Number of titles added per month
df["month_added"] = df["date_added"].dt.to_period("M")
monthly_added = df.groupby("month_added").size().dropna()

plt.figure(figsize=(12, 5))
monthly_added.plot()
plt.title("Monthly Trend of Titles Added to Netflix")
plt.xlabel("Month")
plt.ylabel("Number of Titles Added")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_monthly_titles_added.png")
plt.close()

print("\n=== Key EDA Findings (Quick Summary) ===")
print(f"- Total titles: {len(df)}")
print(f"- Movies: {(df['type'] == 'Movie').sum()}")
print(f"- TV Shows: {(df['type'] == 'TV Show').sum()}")
print(
    f"- Unique countries: {df['country'].dropna().str.split(', ').explode().nunique()}"
)
print(
    f"- Unique genres/categories: {df['listed_in'].dropna().str.split(', ').explode().nunique()}"
)
print("- Plots saved to Lab3/plots/")

print("\nEDA complete.")
