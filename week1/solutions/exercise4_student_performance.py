"""Exercise 4: Student Performance Analysis — solution.

Expects scores.csv in this directory (or the current working directory).
A copy is provided next to this script as solutions/scores.csv.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Prefer scores.csv beside this script; fall back to cwd
csv_path = Path(__file__).resolve().parent / "scores.csv"
if not csv_path.is_file():
    csv_path = Path("scores.csv")

# Task 1: Load and inspect data
df = pd.read_csv(csv_path)
print("First 3 rows:\n", df.head(3))
print("\nData types:\n", df.dtypes)
print("\nSummary stats:\n", df.describe())

# Task 2: Data cleaning
print("\nMissing values:\n", df.isnull().sum())
df["attendance"] = (df["attendance"] * 100).round(2)  # Convert to %

# Task 3: Calculations
df["average_score"] = (
    df[["math_score", "physics_score", "chemistry_score"]].mean(axis=1).round(2)
)
# Match the task rules: <70, 70–<85, ≥85
df["performance_rating"] = pd.cut(
    df["average_score"],
    bins=[-float("inf"), 70, 85, float("inf")],
    right=False,
    labels=["Needs Improvement", "Good", "Excellent"],
)

# Task 4: Analysis
print("\nHighest math score:", df.loc[df["math_score"].idxmax(), "name"])
print(
    "\nClass averages:\n",
    df[["math_score", "physics_score", "chemistry_score"]].mean(),
)
print("\nPerformance rating counts:\n", df["performance_rating"].value_counts())

# Task 5: Visualisation (Bonus)
df.plot(x="name", y="average_score", kind="bar", title="Average Scores by Student")
plt.show()
